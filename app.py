
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="IT Support AI",
    page_icon="🛠️",
    layout="centered"
)

# ---------- Professional UI ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1220, #111c30);
    color: #f1f5f9;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(135deg, #152942, #123b56);
    border: 1px solid #2c4b69;
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 24px;
}
.hero h1 {
    color: #f8fafc;
    margin-bottom: 6px;
}
.hero p {
    color: #b7c9dc;
    margin-bottom: 0;
}
.section-card {
    background: #142238;
    border: 1px solid #2b405a;
    border-radius: 16px;
    padding: 22px;
    margin: 14px 0;
}
div[data-testid="stTextArea"] textarea {
    background-color: #101f34 !important;
    color: #f8fafc !important;
    border: 1px solid #3c5c7d !important;
    border-radius: 12px !important;
    caret-color: #67e8f9 !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border: 2px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px #38bdf833 !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #91a7bf !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0284c7, #0369a1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover {
    background: #0ea5e9;
    color: white;
    border: none;
}
.footer {
    text-align: center;
    color: #94a3b8;
    padding-top: 24px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Training examples ----------
training_data = {
    "Network & Internet": [
        "My Wi-Fi is not working",
        "I have no internet connection",
        "The internet is very slow",
        "I cannot connect to the router",
        "Websites are not loading",
        "I have a DNS error",
        "My computer has no IP address",
        "The network connection keeps disconnecting",
        "I cannot connect to the company network",
        "Internet works on my phone but not my laptop"
    ],
    "Login & Accounts": [
        "I cannot login",
        "My password is not working",
        "My account is locked",
        "I forgot my password",
        "Access is denied",
        "I cannot access a shared folder",
        "My username is not recognized",
        "I cannot sign in to Windows",
        "I have no permission to open a file",
        "My work account is unavailable"
    ],
    "Windows & Operating System": [
        "My computer is very slow",
        "Windows update failed",
        "My PC shows a blue screen",
        "Windows will not start",
        "My computer keeps restarting",
        "Windows is frozen",
        "My laptop gets stuck during startup",
        "My taskbar is not responding",
        "Windows update is stuck",
        "My computer crashes frequently"
    ],
    "Software & Applications": [
        "My application will not open",
        "The program keeps crashing",
        "I cannot install software",
        "The application shows an error",
        "My browser is not responding",
        "The software installation failed",
        "An application freezes every time",
        "My program is incompatible",
        "Office applications are not working",
        "The application closes unexpectedly"
    ],
    "Printers & Scanners": [
        "My printer is not working",
        "The printer is offline",
        "I cannot print documents",
        "The printer has a paper jam",
        "My printer is not detected",
        "The scanner is not working",
        "My printer prints blank pages",
        "The print queue is stuck",
        "The printer driver is missing",
        "I cannot connect to the network printer"
    ],
    "Hardware & Storage": [
        "My hard drive is not detected",
        "My disk is full",
        "My laptop is overheating",
        "My computer has no sound",
        "My keyboard is not working",
        "My mouse is not responding",
        "My monitor shows no signal",
        "My USB device is not detected",
        "My computer runs out of memory",
        "My laptop battery drains quickly"
    ],
    "Email & Communication": [
        "I cannot send emails",
        "I am not receiving emails",
        "Outlook will not open",
        "Email attachments will not open",
        "My mailbox is full",
        "Emails are stuck in the outbox",
        "I cannot connect to Outlook",
        "My email keeps asking for a password",
        "Video calls have no sound",
        "My microphone is not working in meetings"
    ],
    "Security & VPN": [
        "My VPN will not connect",
        "I cannot access the company VPN",
        "My antivirus shows an alert",
        "I received a suspicious email",
        "My security certificate has expired",
        "The firewall blocks my application",
        "I cannot connect securely to the network",
        "My VPN connection keeps dropping",
        "Windows Defender detected a threat",
        "I think my account has been compromised"
    ]
}

solutions = {
    "Network & Internet": [
        "Check whether Wi-Fi is enabled and airplane mode is off.",
        "Test another device on the same network.",
        "Restart your router only if you are authorized to do so.",
        "If the issue continues, record any error message and contact IT Support."
    ],
    "Login & Accounts": [
        "Check your username and keyboard layout.",
        "Make sure Caps Lock is off.",
        "Use your organization's official password reset process.",
        "For locked accounts or missing permissions, contact IT Support."
    ],
    "Windows & Operating System": [
        "Save your work and restart the computer if possible.",
        "Check available disk space and pending Windows updates.",
        "Record any error code or blue-screen message.",
        "If startup fails repeatedly, contact IT Support before attempting recovery."
    ],
    "Software & Applications": [
        "Close the application and reopen it.",
        "Restart the computer if practical.",
        "Check for approved application updates.",
        "Record the exact error message and contact IT Support if the issue continues."
    ],
    "Printers & Scanners": [
        "Check power, cables, Wi-Fi and the printer display.",
        "Confirm that the correct printer is selected.",
        "Check paper, toner and the print queue.",
        "If the printer is still unavailable, ask IT Support to check the driver or connection."
    ],
    "Hardware & Storage": [
        "Check power, cables and device connections.",
        "Close unused applications and check available storage.",
        "Keep ventilation openings clear if the device is overheating.",
        "Back up important files when possible and contact IT Support for persistent hardware faults."
    ],
    "Email & Communication": [
        "Check your internet connection and email service status.",
        "Verify the recipient address and attachment size.",
        "Restart the email application and check for approved updates.",
        "Never share your password; contact IT Support if sign-in keeps failing."
    ],
    "Security & VPN": [
        "Check that you are using the organization's official VPN application.",
        "Verify your internet connection and follow company VPN instructions.",
        "Do not open suspicious links or attachments.",
        "Report suspected security incidents to IT immediately. Do not disable antivirus or firewall protection."
    ]
}

problems = []
categories = []

for category, examples in training_data.items():
    problems.extend(examples)
    categories.extend([category] * len(examples))

model = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])
model.fit(problems, categories)

# ---------- Session state ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "problem" not in st.session_state:
    st.session_state.problem = ""
if "category" not in st.session_state:
    st.session_state.category = ""
if "solution" not in st.session_state:
    st.session_state.solution = []

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🛠️ IT Support AI</h1>
    <p>Intelligent IT Troubleshooting Platform</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.page == "home":
    st.markdown("## How can we help you?")
    st.write(
        "Describe your technical issue. The machine-learning model "
        "will predict a category and suggest troubleshooting steps."
    )

    st.markdown("""
    <div class="section-card">
        <strong>Supported areas</strong><br><br>
        🌐 Network & Internet &nbsp; · &nbsp; 🔐 Login & Accounts<br><br>
        🖥️ Windows &nbsp; · &nbsp; 🧩 Software &nbsp; · &nbsp; 🖨️ Printers<br><br>
        💾 Hardware &nbsp; · &nbsp; ✉️ Email &nbsp; · &nbsp; 🛡️ VPN & Security
    </div>
    """, unsafe_allow_html=True)

    user_problem = st.text_area(
        "Describe your IT problem",
        placeholder="Example: My Wi-Fi is not working...",
        height=150,
        key="problem_input"
    )

    if st.button("Analyze Problem", type="primary"):
        if not user_problem.strip():
            st.warning("Please describe your IT problem first.")
        else:
            probabilities = model.predict_proba([user_problem])[0]
            best_index = probabilities.argmax()
            confidence = float(probabilities[best_index])
            category = model.classes_[best_index]

            st.session_state.problem = user_problem.strip()
            st.session_state.category = category
            st.session_state.confidence = confidence
            st.session_state.solution = solutions[category]
            st.session_state.page = "solution"
            st.rerun()

else:
    st.markdown("## Problem Analysis")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.caption("YOUR REPORTED ISSUE")
    st.write(st.session_state.problem)
    st.markdown("**Predicted category**")
    st.subheader(st.session_state.category)
    st.markdown("</div>", unsafe_allow_html=True)

    confidence = st.session_state.confidence
    if confidence < 0.25:
        st.warning(
            "The model may not recognize this problem reliably. "
            "Try describing it with more detail."
        )
    else:
        st.caption(
            f"Model score: {confidence:.0%} "
            "(this is not a calibrated probability of correctness)."
        )

    st.markdown("## Recommended Troubleshooting Steps")
    for index, step in enumerate(st.session_state.solution, start=1):
        st.markdown(
            f'<div class="section-card"><strong>Step {index}</strong><br>{step}</div>',
            unsafe_allow_html=True
        )

    st.info(
        "These are general recommendations. Follow your organization's "
        "IT policies and contact IT Support for critical issues."
    )

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# ---------- Footer ----------
st.markdown("""
<div class="footer">
    <strong>Developed by Abdullah Aitah</strong><br>
    Built with Python, Streamlit and scikit-learn
</div>
""", unsafe_allow_html=True)
