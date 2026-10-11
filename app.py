import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="IT Support AI",
    page_icon="🛠️",
    layout="centered"
)

# Page header
st.title("🛠️ IT Support AI")
st.caption("Machine Learning for IT Troubleshooting")
st.markdown("**Developed by Abdullah Aitah**")
st.divider()

# Training data
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

# Train model
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(problems, categories)

# Suggested solutions
solutions = {
    "Network": (
        "Check your Wi-Fi connection and router. "
        "Test whether other devices can connect."
    ),
    "Login": (
        "Check your username and keyboard layout. "
        "If your account is locked, contact IT Support."
    ),
    "Printer": (
        "Check the printer power, paper, and connection. "
        "Make sure the correct printer is selected."
    )
}

# User input
st.subheader("Describe your IT problem")

user_problem = st.text_area(
    "Enter your problem in English:",
    placeholder="Example: My internet connection is very slow",
    height=120
)

if st.button("Analyze Problem", type="primary"):
    if not user_problem.strip():
        st.warning("Please describe your IT problem first.")
    else:
        prediction = model.predict([user_problem])[0]

        st.divider()
        st.subheader("Analysis Results")

        st.success(f"Predicted Category: {prediction}")

        st.markdown("### Suggested Solution")
        st.info(solutions[prediction])

        st.caption(
            "Educational prototype: predictions may be incorrect. "
            "Please verify the suggested steps."
        )

# Footer
st.divider()
st.caption("Built with Python, Streamlit and scikit-learn")