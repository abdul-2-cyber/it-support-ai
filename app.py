import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="IT Support AI",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL UI
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1120;
    color: #e5edf8;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] {
    background: rgba(11,17,32,0.9);
}
[data-testid="stSidebar"] {
    background: #101a2d;
    border-right: 1px solid #23334d;
}
[data-testid="stSidebar"] * {
    color: #dce8f7;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #14243d, #123e5b);
    border: 1px solid #294b6c;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 25px;
}
.hero h1 {
    color: #ffffff;
    font-size: 2.25rem;
    font-weight: 800;
    margin: 0;
}
.hero p {
    color: #b8cce2;
    font-size: 1rem;
    margin-top: 10px;
}
.eyebrow {
    color: #67e8f9;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.panel {
    background: #121e32;
    border: 1px solid #263a55;
    border-radius: 18px;
    padding: 24px;
    margin: 12px 0;
}
.metric-card {
    background: #14243a;
    border: 1px solid #2a4562;
    border-radius: 14px;
    padding: 18px;
    min-height: 110px;
}
.metric-label {
    color: #9eb4ce;
    font-size: 0.85rem;
}
.metric-value {
    color: #f8fafc;
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 8px;
}
.step-card {
    background: #14243a;
    border: 1px solid #2a405b;
    border-radius: 13px;
    padding: 18px;
    margin: 10px 0;
}
.step-number {
    color: #67e8f9;
    font-weight: 800;
}
div[data-testid="stTextArea"] textarea {
    background: #101d31 !important;
    color: #f8fafc !important;
    border: 1px solid #3a5675 !important;
    border-radius: 12px !important;
    caret-color: #67e8f9 !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border: 2px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px #38bdf833 !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #8fa7c1 !important;
}
.stButton > button {
    background: linear-gradient(90deg, #0284c7, #0369a1);
    color: white;
    border: 1px solid #168bc5;
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
    transition: 0.2s;
}
.stButton > button:hover {
    background: #0ea5e9;
    color: white;
    border-color: #38bdf8;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #8094ae;
    padding-top: 35px;
    font-size: 0.85rem;
}
hr {
    border-color: #263a55;
}
@media (max-width: 700px) {
    .hero { padding: 22px; }
    .hero h1 { font-size: 1.7rem; }
    .panel { padding: 17px; }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# KNOWLEDGE BASE: TRAINING EXAMPLES
# ==================================================
training_data = {
    "Network & Internet": [
        "My Wi-Fi is not working",
        "I have no internet connection",
        "The internet is very slow",
        "I cannot connect to the router",
        "Websites are not loading",
        "I have a DNS error",
        "My computer has no IP address",
        "The network keeps disconnecting",
        "I cannot connect to the company network",
        "Internet works on my phone but not my laptop",
        "My Ethernet cable is connected but there is no internet",
        "I get a network connection error",
        "Some websites are not accessible",
        "My Wi-Fi says connected without internet",
        "I cannot reach the network gateway",
        "The connection drops every few minutes"
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
        "My work account is unavailable",
        "I keep getting an incorrect password message",
        "My account has been disabled",
        "I cannot access a company resource",
        "I am asked to sign in repeatedly",
        "My login session keeps expiring",
        "I cannot access my user profile"
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
        "My computer crashes frequently",
        "My PC takes too long to boot",
        "Windows shows an error after startup",
        "My desktop icons disappeared",
        "File Explorer is not responding",
        "Windows settings will not open",
        "My computer freezes randomly"
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
        "The application closes unexpectedly",
        "The app says a required file is missing",
        "My software update failed",
        "An application is running very slowly",
        "I cannot uninstall a program",
        "The application needs administrator permission",
        "My browser keeps crashing"
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
        "I cannot connect to the network printer",
        "My printer prints very slowly",
        "The printer says there is no paper",
        "My scanned document is blank",
        "The printer is showing an error code",
        "My printer is connected but not printing",
        "Windows cannot find my printer"
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
        "My laptop battery drains quickly",
        "My external hard drive is not showing",
        "My computer makes unusual noises",
        "My laptop will not charge",
        "My microphone is not detected",
        "My computer has low disk space",
        "My second monitor is not working"
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
        "My microphone is not working in meetings",
        "Emails are going to spam",
        "I cannot download email attachments",
        "My calendar is not synchronizing",
        "Outlook keeps crashing",
        "My emails are delayed",
        "I cannot join a video meeting"
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
        "I think my account has been compromised",
        "My VPN says authentication failed",
        "I received a suspicious attachment",
        "My browser shows a certificate warning",
        "I cannot access an internal website through VPN",
        "My antivirus blocked a file",
        "I think I clicked a phishing link"
    ]
}

# ==================================================
# TROUBLESHOOTING GUIDES
# ==================================================
solutions = {
    "Network & Internet": [
        "Check whether Wi-Fi is enabled and airplane mode is off.",
        "Check if other devices can access the same network.",
        "If possible, disconnect and reconnect to the network.",
        "On Windows, open Command Prompt and run ipconfig to inspect the network configuration.",
        "If the problem continues, record any error message and contact IT Support."
    ],
    "Login & Accounts": [
        "Check your username and keyboard layout, and make sure Caps Lock is off.",
        "Check whether the sign-in page or service is available.",
        "Use your organization's official password-reset process if needed.",
        "Do not share your password or verification codes with anyone.",
        "For locked accounts or missing permissions, contact IT Support."
    ],
    "Windows & Operating System": [
        "Save your work and restart the computer if it is safe to do so.",
        "Check available disk space and pending Windows updates.",
        "Write down any error code or blue-screen message.",
        "Avoid changing system settings or attempting recovery if you are unsure.",
        "If startup fails repeatedly, contact IT Support."
    ],
    "Software & Applications": [
        "Close the application completely and open it again.",
        "Check whether the issue affects only this application or other applications too.",
        "Check for approved software updates.",
        "Record the exact error message and when it appears.",
        "If the problem continues, contact IT Support with the error details."
    ],
    "Printers & Scanners": [
        "Check power, cables, Wi-Fi and the printer display.",
        "Confirm that the correct printer is selected.",
        "Check paper, toner and whether the print queue is paused.",
        "Try a test page if your organization permits it.",
        "If the device remains unavailable, ask IT Support to check the driver and connection."
    ],
    "Hardware & Storage": [
        "Check power, cables and device connections.",
        "Close unused applications and inspect available storage space.",
        "Keep ventilation openings clear if the device is overheating.",
        "Back up important files when possible.",
        "Do not open hardware or continue using a device that smells burnt or appears damaged; contact IT Support."
    ],
    "Email & Communication": [
        "Check your internet connection and whether the email service is available.",
        "Verify the recipient address and attachment size.",
        "Restart the email application and check for approved updates.",
        "For microphone or meeting audio problems, check the selected input and output devices.",
        "Never share your password; contact IT Support if sign-in keeps failing."
    ],
    "Security & VPN": [
        "Use only your organization's official VPN application and instructions.",
        "Check your internet connection and whether your account is authorized for VPN access.",
        "Do not open suspicious links or attachments.",
        "Never disable antivirus or firewall protection to bypass an error.",
        "Report suspected phishing or account compromise to IT immediately."
    ]
}

# ==================================================
# TRAIN SEARCH AND CLASSIFICATION MODELS
# ==================================================
problems = []
categories = []

for category, examples in training_data.items():
    problems.extend(examples)
    categories.extend([category] * len(examples))

# Search model: finds the closest known problem
search_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    strip_accents="unicode",
    sublinear_tf=True,
    stop_words="english"
)
problem_matrix = search_vectorizer.fit_transform(problems)

# Classification model: predicts a broad IT category
classifier = LogisticRegression(max_iter=1500)
classifier.fit(problem_matrix, categories)

# ==================================================
# SESSION STATE
# ==================================================
defaults = {
    "page": "home",
    "problem": "",
    "category": "",
    "solution": [],
    "matched_problem": "",
    "similarity": 0.0,
    "low_match": False
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🛠️ IT Support AI")
    st.caption("Technical troubleshooting assistant")
    st.divider()

    if st.button("⌂  Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("### Supported areas")
    st.markdown("""
    🌐 Network & Internet

    🔐 Login & Accounts

    🖥️ Windows & OS

    🧩 Software

    🖨️ Printers & Scanners

    💾 Hardware & Storage

    ✉️ Email & Communication

    🛡️ Security & VPN
    """)
    st.divider()
    st.caption("Powered by Python, Streamlit and scikit-learn")

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">IT SERVICE DESK · AI ASSISTANT</div>
    <h1>🛠️ IT Support AI</h1>
    <p>Describe your technical issue and get relevant troubleshooting guidance.</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# HOME PAGE
# ==================================================
if st.session_state.page == "home":
    st.markdown("## How can we help you?")
    st.write(
        "Describe the problem in English. The system searches its examples "
        "for a similar issue and predicts the most relevant support category."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Knowledge areas</div>
            <div class="metric-value">8 categories</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Search examples</div>
            <div class="metric-value">128 examples</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Running cost</div>
            <div class="metric-value">Free tools</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🔎 Describe your IT problem")

    with st.form("problem_form"):
        user_problem = st.text_area(
            "Problem description",
            placeholder=(
                "Example: My laptop connects to Wi-Fi, "
                "but I cannot open any websites..."
            ),
            height=150
        )
        submitted = st.form_submit_button(
            "Analyze problem →",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if len(user_problem.strip()) < 5:
            st.warning("Please describe the problem in a little more detail.")
        else:
            query_vector = search_vectorizer.transform([user_problem.strip()])
            similarities = cosine_similarity(
                query_vector, problem_matrix
            ).flatten()

            best_index = int(similarities.argmax())
            best_similarity = float(similarities[best_index])

            predicted_category = classifier.predict(query_vector)[0]

            # Prefer the category of the closest example when the match
            # is meaningful; otherwise use the classifier prediction.
            if best_similarity >= 0.08:
                category = categories[best_index]
                matched_problem = problems[best_index]
            else:
                category = predicted_category
                matched_problem = ""

            st.session_state.problem = user_problem.strip()
            st.session_state.category = category
            st.session_state.solution = solutions[category]
            st.session_state.matched_problem = matched_problem
            st.session_state.similarity = best_similarity
            st.session_state.low_match = best_similarity < 0.08
            st.session_state.page = "solution"
            st.rerun()

# ==================================================
# RESULTS PAGE
# ==================================================
elif st.session_state.page == "solution":
    st.markdown("## 🔍 Problem analysis")
    st.caption("Review the predicted category and recommended next steps.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">YOUR REPORTED ISSUE</div>',
                unsafe_allow_html=True)
    st.write(st.session_state.problem)

    st.markdown("### Predicted category")
    st.subheader(st.session_state.category)

    if st.session_state.matched_problem:
        st.markdown("**Closest example in the knowledge base**")
        st.info(st.session_state.matched_problem)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.low_match:
        st.warning(
            "The search did not find a strong match in the current examples. "
            "The category is only a best-effort prediction. Try adding the "
            "device name, the exact error message, and what you were doing."
        )
    else:
        st.caption(
            f"Text similarity: {st.session_state.similarity:.0%}. "
            "This measures similarity to a stored example; it is not the "
            "probability that the diagnosis is correct."
        )

    st.markdown("## Recommended troubleshooting steps")
    for index, step in enumerate(st.session_state.solution, start=1):
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-number">STEP {index:02d}</span><br><br>
                {step}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "These are general troubleshooting suggestions, not a confirmed "
        "diagnosis. Follow your organization's IT policies."
    )

    if st.button("← Analyze another problem", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    <strong>Developed by Abdullah Aitah</strong><br>
    Built with Python, Streamlit and scikit-learn
</div>
""", unsafe_allow_html=True)
