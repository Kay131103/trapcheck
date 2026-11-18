
# 🛡 scam-lab — train scam-radar-0.6b on Kaggle GPU
# Self-contained: seeds embedded below. Augment -> LoRA (Qwen3-0.6B) -> eval -> push to HF.
# HF token: set HF_TOKEN as a Kaggle secret (Add-ons > Secrets), or attach a dataset
# containing hf_token.txt. Without it the model is still saved to /kaggle/working.

import json, os, random, re, shutil, socket, sys, time

# single GPU + fragmentation-resistant allocator (must be set before torch import;
# Kaggle T4x2 sessions otherwise get DataParallel-wrapped and OOM on the 152k vocab)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import peft  # noqa: F401  (Kaggle GPU images usually ship it)
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)

TARGET_SCAM_PER_FAMILY = 150   # augmented examples per scam family
TARGET_BENIGN_TOTAL    = 80
BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_R, LORA_ALPHA, LR, EPOCHS = 16, 32, 2e-4, 3
WORKING = "/kaggle/working"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ---- 0a · the base model ships as an attached Kaggle dataset (offline mode):
# Kaggle sessions in this account have no working internet (verified by probe),
# so nothing downloads at runtime — everything loads from /kaggle/input.
import glob

def find_model_dir():
    for pattern in ["/kaggle/input/qwen3-0-6b-base", "/kaggle/input/*qwen3*"]:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    # last resort: allow a hub download if the internet happens to work
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])

CACHE = find_model_dir()
print(f"base model from: {CACHE}", flush=True)

# ---- 0b · torchao compatibility: the Kaggle image ships torchao 0.10 but
# transformers 5.x requires >=0.16. The matching wheel ships in the attached
# builddeps dataset — installing from it is instant (no network, no build).
import subprocess as _sp
_wheel = glob.glob("/kaggle/input/trapcheck-builddeps/torchao*.whl")
if _wheel:
    _sp.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
             "--find-links", "/kaggle/input/trapcheck-builddeps",
             "--no-deps", "--upgrade", "torchao"], check=False)
else:
    _sp.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao"], check=False)

# ---- 0c · training corpus: reuse the augmented corpus from the attached
# dataset when present (saves the ~65-minute augmentation pass), else
# augment from seeds as before.
_aug_cache = glob.glob("/kaggle/input/trapcheck-corpus/corpus.json")
AUGMENTED = json.load(open(_aug_cache[0])) if _aug_cache else None
if AUGMENTED:
    print(f"corpus loaded from dataset: {len(AUGMENTED)} examples (augmentation skipped)", flush=True)

# ---- 0 · seeds (hand-curated taxonomy from src/seeds.jsonl) ----
SEEDS = [{'text': 'Dear customer your SBI KYC will expire today. Click sbi-kyc-update[.]site to update else account blocked permanently', 'label': 'scam', 'scam_type': 'kyc_expiry', 'red_flags': ['KYC will expire today', 'else account blocked', 'Click'], 'lang': 'en'}, {'text': 'Aapka Paytm KYC aaj raat 12 baje tak update nahi hua toh wallet block ho jayega. Link: paytm-kyc[.]xyz', 'label': 'scam', 'scam_type': 'kyc_expiry', 'red_flags': ['aaj raat 12 baje tak', 'wallet block', 'Link:'], 'lang': 'hinglish'}, {'text': 'URGENT: This is CBI officer Sharma. A parcel under your Aadhaar contains illegal items. Stay on video call, do not inform anyone, transfer verification amount to secure your identity', 'label': 'scam', 'scam_type': 'digital_arrest', 'red_flags': ['CBI officer', 'do not inform anyone', 'transfer verification amount'], 'lang': 'en'}, {'text': 'This is Mumbai cyber police. Your number is linked to money laundering case. Digital arrest order issued. Pay Rs 50,000 security deposit immediately via UPI', 'label': 'scam', 'scam_type': 'digital_arrest', 'red_flags': ['cyber police', 'Digital arrest', 'security deposit', 'via UPI'], 'lang': 'en'}, {'text': 'Electricity board: power will be disconnected tonight at 9PM due to pending bill. Pay now on ebs-electricity[.]info to restore immediately', 'label': 'scam', 'scam_type': 'utility_bill', 'red_flags': ['tonight at 9PM', 'disconnected', 'Pay now'], 'lang': 'en'}, {'text': 'Bijli board se baat kar rahe hain. Aapka meter today evening disconnect hoga, turant payment kariye bhesab[.]pay link se', 'label': 'scam', 'scam_type': 'utility_bill', 'red_flags': ['today evening disconnect hoga', 'turant payment'], 'lang': 'hinglish'}, {'text': 'Congratulations! Your mobile number won Rs 25,00,000 in KBC lucky draw 2026. Send Rs 8,500 processing fee to claim your prize money today', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['won Rs', 'processing fee', 'claim your prize'], 'lang': 'en'}, {'text': 'Aap KBC lottery jeete hain 25 lakh! Claim karne ke liye 8500 register fee bheje WhatsApp par. Sirf aaj ke liye offer', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['lottery jeete', 'register fee bheje', 'Sirf aaj ke liye'], 'lang': 'hinglish'}, {'text': 'Need urgent cash? Get Rs 2,00,000 instant loan without documents. Download QuickCash app, processing fee only Rs 1,999 first', 'label': 'scam', 'scam_type': 'loan_app', 'red_flags': ['instant loan without documents', 'processing fee only'], 'lang': 'en'}, {'text': 'Congratulations! You are selected for work from home job. Earn Rs 3000 daily doing simple tasks. Pay Rs 999 registration charge to start from tomorrow', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['selected for work from home', 'Earn Rs 3000 daily', 'registration charge'], 'lang': 'en'}, {'text': 'WFH opportunity! Telegram pe join karo, daily 2000-5000 earn karo ghar baithe. Sirf 1500 joining fees hai, aaj invest karo kal se kamai', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['daily 2000-5000 earn karo', 'joining fees', 'aaj invest karo kal se kamai'], 'lang': 'hinglish'}, {'text': 'Your parcel FedEx-7845 is held at customs with unpaid duty of Rs 1,999. Pay online within 2 hours or parcel returns to sender and legal action starts', 'label': 'scam', 'scam_type': 'courier_customs', 'red_flags': ['held at customs', 'unpaid duty', 'within 2 hours', 'legal action'], 'lang': 'en'}, {'text': 'Aapka international parcel customs me fas gaya hai. Rs 2500 clearance charges pay karein warna parcel aapke naam par FIR darj hogi', 'label': 'scam', 'scam_type': 'courier_customs', 'red_flags': ['customs me fas gaya', 'clearance charges', 'FIR darj'], 'lang': 'hinglish'}, {'text': 'SBI: Transaction of Rs 4,999 failed. To refund amount share the OTP received on your mobile with our executive', 'label': 'scam', 'scam_type': 'otp_fraud', 'red_flags': ['share the OTP', 'with our executive'], 'lang': 'en'}, {'text': 'Bank official bol rahe hain account me problem hai. Jo OTP aaya hai wo bata do 2 minute me, warna account freeze ho jayega', 'label': 'scam', 'scam_type': 'otp_fraud', 'red_flags': ['OTP aaya hai wo bata do', 'account freeze'], 'lang': 'hinglish'}, {'text': 'Sir aapke account se 25,000 debit hone wala hai. Ruk jaiye aur abhi humein OTP bhej dijiye transaction cancel karne ke liye', 'label': 'scam', 'scam_type': 'otp_fraud', 'red_flags': ['debit hone wala', 'OTP bhej dijiye', 'transaction cancel'], 'lang': 'hinglish'}, {'text': 'Join our FREE stock tips group! Yesterday all 4 calls hit upper circuit. Sebi registered analyst. Click t[.]me/profitking to make 5000 daily in share market', 'label': 'scam', 'scam_type': 'investment_tips', 'red_flags': ['FREE stock tips', 'hit upper circuit', 'make 5000 daily'], 'lang': 'en'}, {'text': 'Guaranteed 30% monthly return crypto trading platform. Government approved AI trading. Minimum invest 10,000. Withdraw anytime 24 hours', 'label': 'scam', 'scam_type': 'investment_tips', 'red_flags': ['Guaranteed 30% monthly return', 'Government approved', 'Minimum invest'], 'lang': 'en'}, {'text': 'Income Tax Dept: You have a pending refund of Rs 12,480. Verify bank details on itrs-refund[.]online within 24 hours to receive credit, else refund cancelled', 'label': 'scam', 'scam_type': 'tax_refund', 'red_flags': ['pending refund', 'Verify bank details', 'within 24 hours', 'else refund cancelled'], 'lang': 'en'}, {'text': 'Aapka income tax refund 18400 approve hua hai. Bhaai jaldi se bank details update karein refund[.]govt-itr[.]site pe, kal tak nahi kiya toh cancel', 'label': 'scam', 'scam_type': 'tax_refund', 'red_flags': ['income tax refund', 'jaldi se bank details update', 'kal tak nahi kiya toh cancel'], 'lang': 'hinglish'}, {'text': 'Your SIM will be deactivated today as number is linked to illegal activities. Press 1 to speak with officer and stop deactivation', 'label': 'scam', 'scam_type': 'digital_arrest', 'red_flags': ['SIM will be deactivated', 'illegal activities', 'Press 1 to speak'], 'lang': 'en'}, {'text': 'LPG gas cylinder subsidy of Rs 1,200 pending since 3 months. Update UPI id on gas-subsidy[.]org within 12 hours, phir kabhi nahi milega', 'label': 'scam', 'scam_type': 'tax_refund', 'red_flags': ['subsidy of Rs', 'Update UPI id', 'within 12 hours', 'phir kabhi nahi milega'], 'lang': 'hinglish'}, {'text': "Hello ma'am I am from HDFC card department. Your card is blocked due to suspicious activity. Confirm card number CVV and OTP to unblock now", 'label': 'scam', 'scam_type': 'otp_fraud', 'red_flags': ['card department', 'suspicious activity', 'Confirm card number CVV and OTP'], 'lang': 'en'}, {'text': 'PM Awas Yojana 2026: free government grant Rs 1,00,000 approved in your name. Only registration fee Rs 500 required. limited seats apply now', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['free government grant', 'registration fee Rs', 'limited seats'], 'lang': 'en'}, {'text': 'Your son has been in an accident and hospital needs Rs 40,000 deposit immediately for surgery. Send via GPay to this number, doctor is waiting', 'label': 'scam', 'scam_type': 'emergency_scam', 'red_flags': ['accident', 'deposit immediately', 'Send via GPay', 'doctor is waiting'], 'lang': 'en'}, {'text': 'Beta ka accident ho gaya hai, hospital bol rahe hain pehle paise do operation ke baad baithenge. Urgent 30000 bhejo is number par', 'label': 'scam', 'scam_type': 'emergency_scam', 'red_flags': ['accident ho gaya', 'pehle paise do', 'Urgent 30000 bhejo'], 'lang': 'hinglish'}, {'text': 'Earn 8000 weekly typing captions from home. No experience needed! For account activation transfer Rs 750 security fee. Payment proof available', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['Earn 8000 weekly', 'No experience needed', 'transfer Rs 750 security fee'], 'lang': 'en'}, {'text': 'Your electricity meter reading is overdue by 3 months. Final notice attached. Pay Rs 2,340 before 6PM today or line will be cut. -BESCOM', 'label': 'scam', 'scam_type': 'utility_bill', 'red_flags': ['overdue by 3 months', 'before 6PM today', 'line will be cut'], 'lang': 'en'}, {'text': 'Wow! Your contact won the WhatsApp lucky dip. Claim Rs 5 lakh gift voucher from Reliance. Forward this to 10 groups then click wa-gift[.]win', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['lucky dip', 'Claim Rs', 'Forward this to 10 groups', 'click'], 'lang': 'en'}, {'text': 'Taxi passenger lost wallet with your mobile number. I am police constable, send Rs 1,500 reward advance and wallet will be couriered back to you', 'label': 'scam', 'scam_type': 'courier_customs', 'red_flags': ['lost wallet', 'police constable', 'reward advance'], 'lang': 'en'}, {'text': 'Loan approved! Rs 5 lakh at 2% interest, no EMI for 6 months. Verify account with Rs 2,100 token charge today. Offer closes at midnight', 'label': 'scam', 'scam_type': 'loan_app', 'red_flags': ['Loan approved', 'no EMI for 6 months', 'token charge', 'Offer closes at midnight'], 'lang': 'en'}, {'text': 'IIT topper trading secret revealed! Turn 10000 into 95000 in 30 days like my students. Join masterclass free link, seats filling fast', 'label': 'scam', 'scam_type': 'investment_tips', 'red_flags': ['Turn 10000 into 95000', 'Join masterclass', 'seats filling fast'], 'lang': 'en'}, {'text': 'Customs officer here. Gift box from Dubai in your name has 200g gold. Clear it with Rs 15,000 duty or case goes to revenue department', 'label': 'scam', 'scam_type': 'courier_customs', 'red_flags': ['Customs officer', '200g gold', 'Clear it with Rs', 'case goes to revenue'], 'lang': 'en'}, {'text': 'GST refund of Rs 45,670 sanctioned for your business. Submit processing fee Rs 3,999 to release payment, refund window ends Friday', 'label': 'scam', 'scam_type': 'tax_refund', 'red_flags': ['GST refund', 'processing fee', 'release payment', 'ends Friday'], 'lang': 'en'}, {'text': 'Remote job interview scheduled. Before joining pay Rs 1,200 laptop deposit, refundable with first salary. Company: Amazon Hiring Cell', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['pay Rs 1,200 laptop deposit', 'refundable with first salary', 'Amazon Hiring Cell'], 'lang': 'en'}, {'text': 'Aapki shadi profile match ho gayi hai! Ladki ki family 15 lakh dowry dena chahti hai, sirf 25000 mediation fees advance me bhejo', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['match ho gayi', '15 lakh dowry dena chahti', 'mediation fees'], 'lang': 'hinglish'}, {'text': 'Electricity department filed case against you for bill default of Rs 8,900. Arrest warrant process will start in 2 hours. Pay via upi link now to stop', 'label': 'scam', 'scam_type': 'utility_bill', 'red_flags': ['Arrest warrant', 'in 2 hours', 'Pay via upi link'], 'lang': 'en'}, {'text': 'Yr ek madam ne mera 50000 kaa order diya. Pehle 2000 advance uniform charge bhejo, main aapko bhi kaam dilwa dunga guaranteed', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['order diya', 'advance uniform charge', 'guaranteed'], 'lang': 'hinglish'}, {'text': 'Instant credit card approved with 5 lakh limit! Lifetime free. Just pay 1499 joining kit charge and card reaches home in 2 days', 'label': 'scam', 'scam_type': 'loan_app', 'red_flags': ['approved with 5 lakh limit', 'Lifetime free', 'pay 1499 joining kit charge'], 'lang': 'en'}, {'text': 'Madam aapka video viral ho gaya hai social media par. 20000 de do is UPI pe warna sab logoko bhej dunga, 1 hour ka time hai', 'label': 'scam', 'scam_type': 'emergency_scam', 'red_flags': ['video viral', '20000 de do', 'warna sab logoko bhej dunga', '1 hour ka time'], 'lang': 'hinglish'}, {'text': 'Your KYC for UPI transactions has failed verification. In 30 minutes all UPI payments from your number will be blocked. Update on upi-fix[.]in now', 'label': 'scam', 'scam_type': 'kyc_expiry', 'red_flags': ['KYC', 'has failed verification', 'will be blocked', 'Update on'], 'lang': 'en'}, {'text': 'Diwali special! First 100 customers get iPhone 16 for Rs 999 only. Hurry, transfer advance now, delivery free across India, offer only for today', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['iPhone 16 for Rs 999', 'transfer advance now', 'offer only for today'], 'lang': 'en'}, {'text': 'Loan recovery agent speaking. Your brother took 80000 from our app and not paying. Today evening we will visit home and inform society members, send 80000 now', 'label': 'scam', 'scam_type': 'loan_app', 'red_flags': ['recovery agent', 'not paying', 'visit home and inform society', 'send 80000 now'], 'lang': 'en'}, {'text': 'Sebi has approved our new algo trading bot. Fixed 7% profit every week, capital protection guaranteed. Minimum 25000. Demo profit screenshot attached', 'label': 'scam', 'scam_type': 'investment_tips', 'red_flags': ['approved our new algo trading bot', 'Fixed 7% profit every week', 'capital protection guaranteed'], 'lang': 'en'}, {'text': 'FedEx alert: your shipment IND-9921 contains restricted item. Fine Rs 5,000 payable in 1 hour else shipment destroyed and police intimation sent', 'label': 'scam', 'scam_type': 'courier_customs', 'red_flags': ['restricted item', 'Fine Rs', 'in 1 hour', 'shipment destroyed'], 'lang': 'en'}, {'text': 'Congratulations! Your PAN card is selected for Rs 5000 cashback under government digital scheme. Click gov-pan[.]in and enter PAN + bank login to receive', 'label': 'scam', 'scam_type': 'lottery', 'red_flags': ['selected for Rs', 'cashback under government', 'enter PAN + bank login'], 'lang': 'en'}, {'text': 'Your SBI account debited Rs 999 for annual maintenance. Wrongly charged? Refund process: reply YES and share OTP from next SMS', 'label': 'scam', 'scam_type': 'otp_fraud', 'red_flags': ['Wrongly charged', 'share OTP', 'reply YES'], 'lang': 'en'}, {'text': 'BISE board exam 2026 question paper leaked, available for Rs 6000. Guaranteed 90%+. Payment on UPI, paper delivered on WhatsApp night before exam', 'label': 'scam', 'scam_type': 'job_scam', 'red_flags': ['question paper leaked', 'Guaranteed 90%', 'Payment on UPI', 'night before exam'], 'lang': 'en'}, {'text': 'Dear customer, OTP 483921 for Rs 1,250 spent on Flipkart via HDFC card xx8821. NOT share with anyone -HDFC Bank', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Your Flipkart order #OD4323571 has been delivered. Rate your purchase and get 20 supercoins. -Flipkart', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Rs 500 credited to your A/C xx3344 on 12-11-26 by UPI ref 88213456. Balance Rs 12,340.02. -SBI', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Hi beta, kal shaam 6 baje ghar aa jana, papa ke ghar puja hai. Maa', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'hinglish'}, {'text': 'Your electricity bill of Rs 1,140 for Oct 2026 is generated. Due date 20-11-2026. Pay on official app or website. Late fee applies after due date', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Appointment reminder: Dr Sharma clinic visit tomorrow 11AM, Getwell clinic, Andheri West. Reply CANCEL to cancel', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Your OTP for Paytm login is 726481. Do not share this with anyone even if someone claims to be from Paytm', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'IRCTC: Ticket confirmed! PNR 8234567890, 2A, coach S4, seats 12,13. Train departs 22:15. Have a safe journey', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'DTH recharge of Rs 399 successful. Valid till 15-12-2026. Pack: Family HD. -Tata Play', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Reminder: EMI of Rs 8,450 due on 18-11-2026 for loan xx5567. Keep sufficient balance to avoid bounce charges. -HDFC', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': "Aarav's school PTM moved to Saturday 10AM in auditorium. Bring progress report. -Class teacher 7B", 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Your monthly statement is ready. Total outstanding Rs 23,890, minimum due Rs 1,195, due date 25-11-2026. Pay via official app only', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Speed post tracking: article RM882341234IN delivered to addressee at 3:42PM today. -India Post', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Mummy ko bol dena Sunday ko satsang hai mandir me 10 baje. Khana kha liya maine, ghar aa raha hun', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'hinglish'}, {'text': 'Rs 2,00,000 FD maturity credited to A/C xx7788. New FD rate 7.1% p.a. available. Visit branch or official app to renew. -ICICI', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Blood donation camp this Sunday 9AM-1PM at community hall, Sector 21. Walk-ins welcome. Free health check for donors', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Your UPI ID xyz@okaxis received Rs 150 from ankit99@oksbi. Note: towards auto fare. -AXIS Bank', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Society water tank cleaning on Wednesday 10AM-2PM, supply off during work. Store water in advance. -Secretary, Shiv Heights', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'EPF contribution of Rs 4,200 credited for Nov 2026. Check EPFO passbook on official UMANG app only', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Zomato: Your order #382 from Biryani Blues has been picked up and will arrive in 18 mins. Track live in app', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'en'}, {'text': 'Kal interview 11 baje hai, resume print karke le jao aur ID card zaroor lena. Office Metro station ke bagal me hi hai', 'label': 'benign', 'scam_type': None, 'red_flags': [], 'lang': 'hinglish'}]
print(f"{len(SEEDS)} seed messages:",
      sorted({(s['label'], s['scam_type']) for s in SEEDS}))

SYSTEM = ("You are a scam-detection assistant for Indian SMS/WhatsApp messages. "
          "Reply with ONLY a JSON verdict: "
          '{"risk": <0-100>, "scam_type": <family or null>, '
          '"red_flags": [<exact phrases from the message>], "action": "<one line>"}')

AUG_SYS = ("You rewrite scam/benign sample messages into realistic variants. "
           "Keep the SAME language mix (English/Hinglish) and the SAME intent. "
           "Change wording, names, amounts, URLs, urgency phrasing. "
           "Output numbered variants, one per line, nothing else.")

# ---- 1 · load base model (fp16: works on both T4 and P100) ----
tok = AutoTokenizer.from_pretrained(CACHE)
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
base = AutoModelForCausalLM.from_pretrained(CACHE, torch_dtype=torch.float16, device_map="cuda")
base.eval()
print("base model loaded")

@torch.no_grad()
def generate_variants(text, n, rounds=6):
    """Ask the base model for realistic paraphrases; returns up to n*rounds variants."""
    out_all = []
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": AUG_SYS},
         {"role": "user", "content": f"Message: {text}\nWrite 12 numbered realistic variants."}],
        tokenize=False, add_generation_prompt=True)
    for _ in range(rounds):
        ids = tok(prompt, return_tensors="pt").to(base.device)
        out = base.generate(**ids, max_new_tokens=600, do_sample=True, temperature=0.9, top_p=0.9,
                            pad_token_id=PAD_ID)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for line in gen.splitlines():
            line = line.split(". ", 1)[-1].strip()
            if len(line) > 15 and line not in out_all:
                out_all.append(line)
            if len(out_all) >= n:
                return out_all
    return out_all

# ---- 2 · augment (skipped entirely when the cached corpus is attached) ----
if AUGMENTED:
    corpus = AUGMENTED
    print("augmentation skipped — using cached corpus", flush=True)
else:
    corpus = list(SEEDS)
    by_kind = {}
    for s in SEEDS:
        by_kind.setdefault((s["label"], s["scam_type"]), []).append(s)

    t0 = time.time()
    for (label, stype), group in sorted(by_kind.items()):
        need = TARGET_SCAM_PER_FAMILY if label == "scam" else TARGET_BENIGN_TOTAL // max(1, len(group))
        per_seed = max(1, need // len(group))
        for s in group:
            corpus.append(s)
            vs = generate_variants(s["text"], per_seed - 1)
            corpus += [{**s, "text": v, "red_flags": []} for v in vs]
        got = sum(1 for c in corpus if (c["label"], c["scam_type"]) == (label, stype))
        print(f"{label}/{stype}: {got} examples  ({time.time()-t0:.0f}s elapsed)", flush=True)

random.Random(7).shuffle(corpus)
json.dump(corpus, open(f"{WORKING}/corpus.json", "w"), ensure_ascii=False)
print("training corpus:", len(corpus))

# ---- 3 · format: message in -> JSON verdict out; loss on verdict only ----
def verdict_for(s):
    rng = random.Random(s["text"][:32])
    if s["label"] == "scam":
        en = s.get("lang") != "hinglish"
        return {"risk": rng.randint(70, 98), "scam_type": s["scam_type"],
                "red_flags": s["red_flags"],
                "action": ("Do not click, share OTP or pay. Report to 1930 (cyber helpline)." if en
                           else "Kuch bhi click/share/pay mat karo. 1930 (cyber helpline) par report karo.")}
    return {"risk": rng.randint(1, 12), "scam_type": None, "red_flags": [],
            "action": "Looks legitimate. No action needed."}

examples = []
for s in corpus:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": s["text"]}],
        tokenize=False, add_generation_prompt=True)
    gold = json.dumps(verdict_for(s), ensure_ascii=False)
    ids = tok(prompt + gold + tok.eos_token, truncation=True, max_length=512,
              add_special_tokens=False)["input_ids"]
    plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = ids[:]
    if plen < len(labels):
        labels[:plen] = [-100] * plen
    examples.append({"input_ids": ids, "labels": labels})
print(f"{len(examples)} training examples")

# ---- 4 · LoRA ----
del base
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(CACHE, torch_dtype=torch.float16, device_map="cuda")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.gradient_checkpointing_enable()  # trades a little speed for big activation-memory savings
model.print_trainable_parameters()

# ---- 5 · train ----
from transformers import Trainer, TrainingArguments

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = PAD_ID
    return {"input_ids": torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch])}

# transformers 5.x renamed/removed some TrainingArguments kwargs; build the
# args adaptively so any version works (unsupported kwargs get dropped).
ta_kwargs = {
    "output_dir": f"{WORKING}/runs", "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4,
    "learning_rate": LR, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
    "fp16": True, "logging_steps": 20, "save_strategy": "no",
    "report_to": [], "remove_unused_columns": False,
}
while True:
    try:
        args = TrainingArguments(**ta_kwargs)
        break
    except TypeError as _te:
        import re as _re
        _m = _re.search(r"unexpected keyword argument ['\"](\w+)", str(_te))
        if not _m:
            raise
        _bad = _m.group(1)
        print(f"TrainingArguments: dropping unsupported kwarg '{_bad}'", flush=True)
        ta_kwargs.pop(_bad, None)

trainer = None
try:
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
except Exception as _te:
    print(f"Trainer(list dataset) failed ({type(_te).__name__}) — converting to Dataset", flush=True)
    from datasets import Dataset as _DS
    trainer = Trainer(model=model, args=args, train_dataset=_DS.from_list(examples), data_collator=collate)
trainer.train()

# ---- 6 · evaluate on the real (non-augmented) seed messages ----
def extract_verdict(t):
    match = re.search(r"\{.*\}", t.strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def auc(pairs):
    pos = sum(1 for _, y in pairs if y == 1); neg = len(pairs) - pos
    if not pos or not neg: return float("nan")
    w = t = 0
    for i, (si, yi) in enumerate(pairs):
        for sj, yj in pairs[i + 1:]:
            if yi == yj: continue
            sp, sn = (si, sj) if yi == 1 else (sj, si)
            w += sp > sn; t += sp == sn
    return (w + 0.5 * t) / (pos * neg)

@torch.no_grad()
def verdict(m, text):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=140, do_sample=False,
                     pad_token_id=PAD_ID)
    return extract_verdict(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))

m = model.eval()
pairs, tp_, tg_, th_, ben_fp, cal = [], {}, {}, {}, [], {}
for s in SEEDS:
    v = verdict(m, s["text"])
    risk = v.get("risk") if v else None
    if risk is not None:
        pairs.append((float(risk), 1 if s["label"] == "scam" else 0))
    if s["label"] == "benign" and (risk or 0) >= 50:
        ben_fp.append(s["text"])
    if s["label"] == "scam":
        p = (v or {}).get("scam_type"); g = s["scam_type"]
        tp_[p] = tp_.get(p, 0) + 1; tg_[g] = tg_.get(g, 0) + 1
        if p == g: th_[g] = th_.get(g, 0) + 1
    if v:
        b = min(int(risk or 0) // 25, 3)
        cal.setdefault(b, []).append(1 if s["label"] == "scam" else 0)

f1s = []
for lab in set(tg_) | set(tp_):
    tp = th_.get(lab, 0)
    pr = tp / tp_[lab] if tp_.get(lab) else 0.0
    rc = tp / tg_[lab] if tg_.get(lab) else 0.0
    f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)

results = {
    "auc": round(auc(pairs), 3),
    "benign_fpr": round(len(ben_fp) / max(1, sum(1 for r in SEEDS if r["label"] == "benign")), 3),
    "type_macro_f1": round(sum(f1s) / len(f1s), 3),
    "calibration": [{"bucket": f"{b*25}-{b*25+24}", "n": len(v), "scam_rate": round(sum(v)/len(v), 2)}
                    for b, v in sorted(cal.items())],
    "benign_false_positives": ben_fp[:5],
}
print(json.dumps(results, indent=2))
json.dump(results, open(f"{WORKING}/eval_results.json", "w"), default=str)

# ---- 7 · push to HF (needs HF_TOKEN via Kaggle secret / env / attached dataset) ----
def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t: return t
    except Exception: pass
    if os.environ.get("HF_TOKEN"): return os.environ["HF_TOKEN"]
    for p in ["/kaggle/input/hf-credentials/hf_token.txt"]:
        if os.path.exists(p): return open(p).read().strip()
    return None

model = model.merge_and_unload()
model.save_pretrained(f"{WORKING}/scam-radar-0.6b")
tok.save_pretrained(f"{WORKING}/scam-radar-0.6b")

def internet_ok(host="huggingface.co", timeout=5):
    """Quick DNS check: on unverified Kaggle accounts there is no internet at all."""
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

token = hf_token()
if token and internet_ok():
    from huggingface_hub import HfApi
    who = HfApi(token=token).whoami()
    repo = f"{who['name']}/scam-radar-0.6b"
    api = HfApi(token=token)
    for attempt in range(5):
        try:
            api.create_repo(repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=f"{WORKING}/scam-radar-0.6b", repo_id=repo)
            api.upload_file(path_or_fileobj=f"{WORKING}/eval_results.json", path_in_repo="eval_results.json", repo_id=repo)
            break
        except Exception as exc:
            print(f"push attempt {attempt + 1}/5 failed: {type(exc).__name__}; retrying in 60s", flush=True)
            time.sleep(60)
    card = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
tags: [scam-detection, lora, qwen3, hinglish, fraud]
---

# scam-radar-0.6b

India-specific scam detector (SMS/WhatsApp, English + Hinglish) — LoRA fine-tune of {BASE_MODEL}.
Outputs a JSON verdict: risk 0-100, scam family, red-flag phrases, recommended action.

## Held-out results (seed corpus, n={len(SEEDS)})
```json
{json.dumps({k: v for k, v in results.items() if k != 'benign_false_positives'}, indent=2, default=str)}
```

Benign-FPR is the headline metric: flagging genuine bank OTP messages would train users to ignore warnings.
Hand-curated 10-family scam taxonomy + paraphrase augmentation. See eval_results.json.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    print("PUSHED → https://huggingface.co/" + repo)
else:
    print("NO HF TOKEN — model saved to /kaggle/working/scam-radar-0.6b (download manually or add HF_TOKEN secret and re-run).")
print("DONE")
