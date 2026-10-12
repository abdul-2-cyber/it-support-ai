


import html
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="IT Support Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# PROFESSIONAL CUSTOMER INTERFACE
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0b1120;
    color: #edf4fc;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"],
#MainMenu,
[data-testid="stDecoration"],
[data-testid="stAppDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

.block-container {
    max-width: 920px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(125deg, #142944, #104464);
    border: 1px solid #285273;
    border-radius: 22px;
    padding: 38px;
    margin-bottom: 28px;
}

.eyebrow {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(30px, 4vw, 42px);
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #c3d7ea;
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 0;
}

.panel {
    background: #111d30;
    border: 1px solid #263b55;
    border-radius: 18px;
    padding: 26px;
    margin: 18px 0;
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
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0ea5e9;
    color: white;
    border-color: #7dd3fc;
}

.result-card {
    background: #13243a;
    border: 1px solid #2b4561;
    border-radius: 16px;
    padding: 23px;
    margin: 15px 0;
}

.step-card {
    background: #111e31;
    border: 1px solid #293e58;
    border-radius: 13px;
    padding: 19px;
    margin: 12px 0;
    line-height: 1.8;
}

.step-number {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.footer {
    text-align: center;
    color: #91a5bd;
    border-top: 1px solid #24354c;
    margin-top: 40px;
    padding-top: 24px;
    font-size: 13px;
}

h1, h2, h3, h4 {
    color: #f1f7ff;
}

@media (max-width: 700px) {
    .hero {
        padding: 24px;
    }

    .panel {
        padding: 17px;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TRAINING DATA
# ==================================================
training_data = {
    "Network & Internet": [
        "My Wi-Fi is not working",
        "I have no internet connection",
        "The internet is very slow",
        "Websites are not loading",
        "I have a DNS error",
        "My computer has no IP address",
        "The network keeps disconnecting",
        "Wi-Fi connected but no internet",
        "My Ethernet connection is not working",
        "My laptop cannot find Wi-Fi networks",
        "I cannot connect to the company network",
        "Internet works on my phone but not my laptop",
        "I cannot access the network gateway",
        "My connection drops every few minutes"
    ],
    "Login & Accounts": [
        "I cannot log in",
        "My password is not working",
        "My account is locked",
        "I forgot my password",
        "Access is denied",
        "I cannot access a shared folder",
        "My username is not recognized",
        "I cannot sign in to Windows",
        "I have no permission to open a file",
        "My account has been disabled",
        "I keep getting an incorrect password message",
        "I am asked to sign in repeatedly"
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
        "File Explorer is not responding",
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
        "I cannot uninstall a program",
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
        "My laptop will not charge",
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
        "My calendar is not synchronizing",
        "I cannot join a video meeting"
    ],
    "Security & VPN": [
        "My VPN will not connect",
        "I cannot access the company VPN",
        "My antivirus shows an alert",
        "I received a suspicious email",
        "My security certificate has expired",
        "The firewall blocks my application",
        "My VPN connection keeps dropping",
        "Windows Defender detected a threat",
        "I think my account has been compromised",
        "My VPN says authentication failed",
        "I received a suspicious attachment",
        "I think I clicked a phishing link"
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
        "On Windows, run ipconfig in Command Prompt to inspect network settings.",
        "If the issue continues, record the error message and contact IT Support."
    ],
    "Login & Accounts": [
        "Check your username, keyboard layout and Caps Lock.",
        "Use your organization's official password-reset process if needed.",
        "Check whether the service is available.",
        "Never share your password or verification codes.",
        "Contact IT Support for locked accounts or permission problems."
    ],
    "Windows & Operating System": [
        "Save your work and restart the computer if safe.",
        "Check available disk space and pending Windows updates.",
        "Record any error code or blue-screen message.",
        "Avoid recovery operations if you are unsure of their effects.",
        "Contact IT Support if startup failures or crashes continue."
    ],
    "Software & Applications": [
        "Close the application completely and reopen it.",
        "Check whether the issue affects only this application.",
        "Check for approved application updates.",
        "Record the exact error message.",
        "Contact IT Support if the problem continues."
    ],
    "Printers & Scanners": [
        "Check power, cables, Wi-Fi and the printer display.",
        "Confirm that the correct printer is selected.",
        "Check paper, toner and the print queue.",
        "Try a test page if permitted by your organization.",
        "Contact IT Support if the driver or connection may be faulty."
    ],
    "Hardware & Storage": [
        "Check power, cables and device connections.",
        "Inspect available storage and close unused applications.",
        "Keep ventilation openings clear if the device is overheating.",
        "Back up important files when possible.",
        "Contact IT Support for persistent hardware problems."
    ],
    "Email & Communication": [
        "Check your internet connection and email service availability.",
        "Verify the recipient address and attachment size.",
        "Restart the email application and check approved updates.",
        "For meeting audio issues, check the selected microphone and speakers.",
        "Never share your password; contact IT Support if sign-in keeps failing."
    ],
    "Security & VPN": [
        "Use only your organization's official VPN application.",
        "Check your internet connection and VPN account authorization.",
        "Do not open suspicious links or attachments.",
        "Never disable antivirus or firewall protection to bypass an error.",
        "Report suspected phishing or account compromise to IT immediately."
    ]
}

# ==================================================
# BUILD SEARCH MODEL
# ==================================================
problems = []
categories = []

for category, examples in training_data.items():
    problems.extend(examples)
    categories.extend([category] * len(examples))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    strip_accents="unicode",
    sublinear_tf=True,
    stop_words="english"
)

problem_matrix = vectorizer.fit_transform(problems)

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
    "low_match": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

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
# HOME PAGE — NO CATEGORY CARDS
# ==================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="panel">
        <h3 style="color:#ffffff;">How can we help you today?</h3>
        <p style="color:#a7bad0;line-height:1.8;">
            Describe the technical issue you are experiencing.
            Include any error message if one appears.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("problem_form"):
        user_problem = st.text_area(
            "Problem description",
            placeholder=(
                "Example: My laptop connects to Wi-Fi, "
                "but websites do not load..."
            ),
            height=160
        )

        submitted = st.form_submit_button(
            "Find a solution  →",
            use_container_width=True
        )

    if submitted:
        if len(user_problem.strip()) < 5:
            st.warning("Please describe your issue in a little more detail.")
        else:
            query = vectorizer.transform([user_problem.strip()])
            similarities = cosine_similarity(
                query, problem_matrix
            ).flatten()

            best_index = int(similarities.argmax())
            best_score = float(similarities[best_index])
            predicted_category = classifier.predict(query)[0]

            category = (
                categories[best_index]
                if best_score >= 0.08
                else predicted_category
            )

            st.session_state.problem = user_problem.strip()
            st.session_state.category = category
            st.session_state.solution = solutions[category]
            st.session_state.low_match = best_score < 0.08
            st.session_state.page = "solution"
            st.rerun()

# ==================================================
# RESULTS PAGE
# ==================================================
elif st.session_state.page == "solution":
    st.markdown("## Your Support Results")
    st.caption("Review the recommended steps for your issue.")

    st.markdown(f"""
    <div class="panel">
        <div class="eyebrow">YOUR REPORTED ISSUE</div>
        <p style="color:#edf4fc;">{html.escape(st.session_state.problem)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Issue category")
    st.subheader(st.session_state.category)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.low_match:
        st.warning(
            "We could not find a strong match. The suggested category may "
            "not be correct. Try adding your device name and exact error."
        )

    st.markdown("## Recommended next steps")

    for index, step in enumerate(st.session_state.solution, start=1):
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">STEP {index:02d}</div>
            <div style="margin-top:8px;">{html.escape(step)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "These suggestions are general guidance, not a confirmed diagnosis. "
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

