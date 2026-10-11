
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="IT Support | Help Center",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL CUSTOMER-FACING DESIGN
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1120;
    --panel: #111d30;
    --panel2: #15243a;
    --border: #263b55;
    --text: #edf4fc;
    --muted: #a7bad0;
    --accent: #38bdf8;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(11,17,32,0.9);
}

[data-testid="stSidebar"] {
    background: #0f192b;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: #dce8f7;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(125deg, #142944 0%, #104464 100%);
    border: 1px solid #285273;
    border-radius: 24px;
    padding: 38px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.eyebrow {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 12px 0;
}

.hero p {
    color: #c3d7ea;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
    margin-bottom: 0;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    margin: 16px 0;
}

.area-card {
    background: var(--panel2);
    border: 1px solid #2b405c;
    border-radius: 14px;
    padding: 17px;
    min-height: 90px;
    margin-bottom: 10px;
}

.area-card strong {
    color: #f1f7ff;
    font-size: 14px;
}

.area-card p {
    color: var(--muted);
    font-size: 12px;
    margin: 7px 0 0 0;
}

.result-card {
    background: linear-gradient(135deg, #13253b, #102033);
    border: 1px solid #2b4561;
    border-radius: 18px;
    padding: 24px;
    margin: 15px 0;
}

.step-card {
    background: #111e31;
    border: 1px solid #293e58;
    border-radius: 13px;
    padding: 19px;
    margin: 11px 0;
    line-height: 1.8;
}

.step-label {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

div[data-testid="stTextArea"] textarea {
    background: #0e1a2d !important;
    color: #f8fafc !important;
    border: 1px solid #3b5675 !important;
    border-radius: 12px !important;
    caret-color: #7dd3fc !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border: 2px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px #38bdf833 !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #8fa6c0 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0284c7, #0369a1);
    color: white;
    border: 1px solid #168bc5;
    border-radius: 11px;
    min-height: 46px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0ea5e9;
    color: white;
    border-color: #7dd3fc;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

h1, h2, h3, h4 {
    color: #f1f7ff;
}

.footer {
    text-align: center;
    color: #91a5bd;
    border-top: 1px solid #24354c;
    margin-top: 42px;
    padding-top: 24px;
    font-size: 13px;
}

@media (max-width: 700px) {
    .hero { padding: 24px; }
    .panel { padding: 17px; }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TRAINING EXAMPLES
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
        "Wi-Fi connected but no internet access",
        "My Ethernet connection is not working",
        "Some websites cannot be reached",
        "My network connection keeps dropping",
        "I cannot access the network gateway",
        "The internet stopped working after an update",
        "My laptop cannot find any Wi-Fi networks",
        "I get a network connection error"
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
        "I cannot access my user profile",
        "My account requires verification",
        "I get an authentication error"
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
        "My computer freezes randomly",
        "Windows update is downloading forever",
        "My PC shuts down unexpectedly"
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
        "My browser keeps crashing",
        "The software shows a license error",
        "My application is not responding"
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
        "Windows cannot find my printer",
        "My printer prints the wrong pages",
        "The scanner cannot connect to my computer"
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
        "My second monitor is not working",
        "My laptop fan is very loud",
        "My computer cannot detect a USB drive"
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
        "I cannot join a video meeting",
        "My email signature is missing",
        "I cannot open a shared mailbox"
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
        "I think I clicked a phishing link",
        "I received a fake password reset email",
        "The VPN says connection timed out"
    ]
}

# ==================================================
# TROUBLESHOOTING GUIDES
# ==================================================
solutions = {
    "Network & Internet": [
        "Check whether Wi-Fi is enabled and airplane mode is off.",
        "Check whether other devices can access the same network.",
        "Disconnect and reconnect to the network if possible.",
        "On Windows, open Command Prompt and run ipconfig to inspect your network configuration.",
        "If the issue continues, record the error message and contact IT Support."
    ],
    "Login & Accounts": [
        "Check your username, keyboard layout and Caps Lock.",
        "Confirm that the sign-in page or service is available.",
        "Use your organization's official password-reset process if necessary.",
        "Never share your password or verification codes.",
        "For locked accounts or missing permissions, contact IT Support."
    ],
    "Windows & Operating System": [
        "Save your work and restart the computer if it is safe to do so.",
        "Check available disk space and pending Windows updates.",
        "Record any error code or blue-screen message.",
        "Avoid recovery operations if you are unsure of their effects.",
        "If startup fails repeatedly, contact IT Support."
    ],
    "Software & Applications": [
        "Close the application completely and open it again.",
        "Check whether the issue affects only this application.",
        "Check for approved application updates.",
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
        "Close unused applications and inspect available storage.",
        "Keep ventilation openings clear if the device is overheating.",
        "Back up important files when possible.",
        "For persistent hardware problems, contact IT Support."
    ],
    "Email & Communication": [
        "Check your internet connection and whether the email service is available.",
        "Verify the recipient address and attachment size.",
        "Restart the email application and check for approved updates.",
        "For meeting audio issues, check the selected microphone and speakers.",
        "Never share your password; contact IT Support if sign-in keeps failing."
    ],
    "Security & VPN": [
        "Use only your organization's official VPN application and instructions.",
        "Check your internet connection and VPN account authorization.",
        "Do not open suspicious links or attachments.",
        "Never disable antivirus or firewall protection to bypass an error.",
        "Report suspected phishing or account compromise to IT immediately."
    ]
}

# ==================================================
# SEARCH AND CLASSIFICATION
# ==================================================
problems = []
categories = []

for category, examples in training_data.items():
    problems.extend(examples)
    categories.extend([category] * len(examples))

search_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    strip_accents="unicode",
    sublinear_tf=True,
    stop_words="english"
)
problem_matrix = search_vectorizer.fit_transform(problems)

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
    st.markdown("## 🛠️ IT Support")
    st.caption("Customer Help Center")
    st.divider()

    if st.button("⌂  Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("### Support services")
    st.markdown("""
    🌐 **Network & Internet**

    🔐 **Account Access**

    🖥️ **Windows & Software**

    🖨️ **Printers & Scanners**

    💾 **Hardware**

    ✉️ **Email & Communication**

    🛡️ **Security & VPN**
    """)
    st.divider()
    st.caption("Need more help? Contact your IT department.")

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">CUSTOMER SUPPORT CENTER</div>
    <h1>IT Support Assistant</h1>
    <p>
        Technical support made simpler. Describe your issue and
        receive practical guidance to help you resolve it.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# HOME PAGE
# ==================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="panel">
        <h3 style="color:#ffffff;">How can we help you today?</h3>
        <p style="color:#a7bad0; line-height:1.8;">
            Describe the technical issue you are experiencing.
            Include any error message if one appears.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="area-card">
            <strong>🌐 Connectivity</strong>
            <p>Wi-Fi, internet and network access</p>
        </div>
        <div class="area-card">
            <strong>🔐 Account access</strong>
            <p>Login, passwords and permissions</p>
        </div>
        <div class="area-card">
            <strong>🖨️ Office equipment</strong>
            <p>Printers, scanners and peripherals</p>
        </div>
        <div class="area-card">
            <strong>🛡️ Security</strong>
            <p>VPN access and suspicious activity</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="area-card">
            <strong>🖥️ Windows</strong>
            <p>Startup, updates and system errors</p>
        </div>
        <div class="area-card">
            <strong>🧩 Applications</strong>
            <p>Software errors and crashes</p>
        </div>
        <div class="area-card">
            <strong>💾 Hardware</strong>
            <p>Storage, displays and devices</p>
        </div>
        <div class="area-card">
            <strong>✉️ Communication</strong>
            <p>Email, calendars and online meetings</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Tell us what happened")

    with st.form("problem_form"):
        user_problem = st.text_area(
            "Problem description",
            placeholder=(
                "Example: My laptop connects to Wi-Fi, "
                "but I cannot open any websites..."
            ),
            height=145,
            help="Describe what happened and include any error message."
        )
        submitted = st.form_submit_button(
            "Find a solution  →",
            use_container_width=True
        )

    if submitted:
        if len(user_problem.strip()) < 5:
            st.warning("Please provide a little more detail about the issue.")
        else:
            query_vector = search_vectorizer.transform(
                [user_problem.strip()]
            )
            similarities = cosine_similarity(
                query_vector, problem_matrix
            ).flatten()

            best_index = int(similarities.argmax())
            best_similarity = float(similarities[best_index])
            predicted_category = classifier.predict(query_vector)[0]

            if best_similarity >= 0.08:
                category = categories[best_index]
            else:
                category = predicted_category

            st.session_state.problem = user_problem.strip()
            st.session_state.category = category
            st.session_state.solution = solutions[category]
            st.session_state.similarity = best_similarity
            st.session_state.low_match = best_similarity < 0.08
            st.session_state.page = "solution"
            st.rerun()

# ==================================================
# RESULTS PAGE
# ==================================================
elif st.session_state.page == "solution":
    st.markdown("## Your Support Results")
    st.caption("Review the suggested steps below to troubleshoot your issue.")

    st.markdown("""
    <div class="result-card">
        <div class="eyebrow">ISSUE REPORTED</div>
    </div>
    """, unsafe_allow_html=True)
    st.write(st.session_state.problem)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### Issue category")
    st.subheader(st.session_state.category)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.low_match:
        st.warning(
            "We could not find a strong match for this issue. "
            "The suggested category may not be correct. Try adding "
            "the device name and the exact error message."
        )

    st.markdown("## Recommended next steps")

    for index, step in enumerate(st.session_state.solution, start=1):
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-label">STEP {index:02d}</div>
                <div style="color:#edf4fc; margin-top:8px;">{step}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "These steps are general guidance, not a confirmed diagnosis. "
        "Follow your organization's IT policies."
    )

    if st.button("← Report another issue", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    <strong>Developed by Abdullah Aitah</strong><br>
    Customer IT Support
</div>
""", unsafe_allow_html=True)
