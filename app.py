
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="IT Support AI",
    page_icon="🛠️",
    layout="centered"
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111d30);
    color: #f1f5f9;
}
.block-container {
    max-width: 850px;
    padding-top: 2.5rem;
}
.hero {
    padding: 1.5rem;
    border: 1px solid #334155;
    border-radius: 18px;
    background: #142238;
    margin-bottom: 1.5rem;
}
.brand {
    color: #67e8f9;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 2px;
}
.subtitle {
    color: #a8b8cc;
}
.result-card {
    padding: 1.3rem;
    background: #142238;
    border: 1px solid #334155;
    border-radius: 16px;
    margin: 1rem 0;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 3rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- Machine Learning Model ----------
problems = [
    "My Wi-Fi is not working",
    "I have no internet connection",
    "The network is very slow",
    "I cannot connect to the router",
    "I cannot login",
    "My password is not working",
    "My account is locked",
    "I forgot my password",
    "My printer is not working",
    "The printer is offline",
    "I cannot print documents",
    "The printer has a paper jam"
]

categories = [
    "Network", "Network", "Network", "Network",
    "Login", "Login", "Login", "Login",
    "Printer", "Printer", "Printer", "Printer"
]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])
model.fit(problems, categories)

solutions = {
    "Network": [
        "Check that Wi-Fi is enabled on your device.",
        "Check whether other devices can connect.",
        "Restart the router if you are authorized to do so.",
        "If the problem continues, contact IT Support."
    ],
    "Login": [
        "Check your username and keyboard layout.",
        "Make sure Caps Lock is not enabled.",
        "If your account is locked, contact IT Support.",
        "Never share your password with anyone."
    ],
    "Printer": [
        "Make sure the printer is powered on.",
        "Check paper, toner, and any error messages.",
        "Verify the printer connection and selected printer.",
        "Clear a paper jam only by following the printer instructions."
    ]
}

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "problem" not in st.session_state:
    st.session_state.problem = ""
if "category" not in st.session_state:
    st.session_state.category = ""

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="brand">INTELLIGENT IT TROUBLESHOOTING</div>
    <h1>🛠️ IT Support AI</h1>
    <p class="subtitle">
        Describe your technical issue and get a recommended solution.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- Home Page ----------
if st.session_state.page == "home":
    st.subheader("How can we help you?")
    st.write("Enter your IT problem below to begin the analysis.")

    with st.form("problem_form"):
        problem = st.text_area(
            "Describe your problem",
            placeholder="Example: My Wi-Fi is not working",
            height=140
        )
        submitted = st.form_submit_button(
            "🔍 Analyze Problem",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not problem.strip():
            st.warning("Please describe your IT problem first.")
        else:
            category = model.predict([problem.strip()])[0]
            st.session_state.problem = problem.strip()
            st.session_state.category = category
            st.session_state.page = "result"
            st.rerun()

# ---------- Results Page ----------
else:
    st.subheader("Analysis Results")

    st.markdown(
        f'<div class="result-card">'
        f'<p class="subtitle">YOUR REPORTED ISSUE</p>'
        f'<strong>{st.session_state.problem}</strong>'
        f'</div>',
        unsafe_allow_html=True
    )

    category = st.session_state.category
    icons = {
        "Network": "🌐",
        "Login": "🔐",
        "Printer": "🖨️"
    }

    st.success(f"{icons[category]} Predicted Category: {category}")
    st.markdown("### Recommended troubleshooting steps")

    for index, step in enumerate(solutions[category], start=1):
        st.markdown(f"**{index}.** {step}")

    st.info(
        "This is an educational machine-learning prototype. "
        "Its predictions may be incorrect. Verify the steps before acting."
    )

    if st.button("← Back to Problem Input"):
        st.session_state.page = "home"
        st.rerun()

# ---------- Footer ----------
st.divider()
st.caption("Developed by Abdullah Aitah")
st.caption("Built with Python · Streamlit · scikit-learn")
