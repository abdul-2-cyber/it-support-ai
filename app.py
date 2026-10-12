
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="IT Support AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Design --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #101e35 55%, #0a1424 100%);
    color: #f1f5f9;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(135deg, #142b49, #102039);
    border: 1px solid #294363;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 26px;
}
.hero h1 {
    color: white;
    font-size: 2.5rem;
}
.hero p {
    color: #c2d1e3;
    font-size: 1.05rem;
}
.stTextArea textarea {
    background-color: #0d192a !important;
    color: #f1f5f9 !important;
    border: 1px solid #334b69 !important;
    border-radius: 10px !important;
}
div.stButton > button {
    background: #2477e8;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #3989f5;
    color: white;
}
.result {
    background: #111f33;
    border-left: 4px solid #2477e8;
    padding: 18px 22px;
    margin: 15px 0;
    border-radius: 5px;
}
.footer {
    text-align: center;
    color: #8295ad;
    border-top: 1px solid #26364b;
    padding-top: 20px;
    margin-top: 40px;
    font-size: 0.85rem;
}
h2, h3 {
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Language --------------------

if "language" not in st.session_state:
    st.session_state.language = "English"

language = st.selectbox(
    "Language / Sprache",
    ["English", "Deutsch"],
    key="language",
)
de = language == "Deutsch"

T = {
    "description": (
        "Beschreibe dein IT-Problem, um eine Kategorie und Lösungsvorschläge zu erhalten."
        if de else
        "Describe your IT problem to receive a category and troubleshooting suggestions."
    ),
    "title": "Wie können wir helfen?" if de else "How can we help you?",
    "categories": "Verfügbare Support-Bereiche" if de else "Available Support Categories",
    "network": "Netzwerk & Internet" if de else "Network & Internet",
    "login": "Anmeldung & Konten" if de else "Login & Accounts",
    "windows": "Windows & Betriebssystem" if de else "Windows & Operating System",
    "software": "Software & Anwendungen" if de else "Software & Applications",
    "printer": "Drucker & Scanner" if de else "Printers & Scanners",
    "hardware": "Hardware & Speicher" if de else "Hardware & Storage",
    "email": "E-Mail & Kommunikation" if de else "Email & Communication",
    "security": "Sicherheit & VPN" if de else "Security & VPN",
    "input": "Beschreibe dein Problem" if de else "Describe Your Problem",
    "placeholder": "Zum Beispiel: Mein WLAN funktioniert nicht."
                  if de else "For example: My Wi-Fi is not working.",
    "analyze": "Problem analysieren" if de else "Analyze Problem",
    "empty": "Bitte beschreibe zuerst dein Problem."
             if de else "Please describe your problem first.",
    "result": "Analyseergebnis" if de else "Analysis Result",
    "predicted": "Erkannte Kategorie" if de else "Predicted Category",
    "confidence": "Modell-Konfidenz" if de else "Model Confidence",
    "solutions": "Lösungsvorschläge" if de else "Suggested Solutions",
    "back": "Zurück" if de else "Back",
    "footer": "Developed by Abdullah Aitah",
}

# -------------------- Training data --------------------

data = {
    "Network & Internet": [
        "wifi not working", "no internet connection",
        "internet is slow", "network connection lost",
        "cannot connect to wifi", "dns error",
        "WLAN funktioniert nicht", "kein Internet",
        "Internet ist langsam", "Netzwerkverbindung unterbrochen",
        "keine WLAN Verbindung", "DNS Fehler",
    ],
    "Login & Accounts": [
        "cannot log in", "forgot password", "account locked",
        "login failed", "password not working",
        "Ich kann mich nicht anmelden", "Passwort vergessen",
        "Konto gesperrt", "Anmeldung fehlgeschlagen",
        "Passwort funktioniert nicht",
    ],
    "Windows & Operating System": [
        "computer is slow", "windows keeps crashing",
        "blue screen", "windows update failed",
        "computer freezes", "windows will not start",
        "Computer ist langsam", "Windows stürzt ab",
        "Bluescreen", "Windows Update fehlgeschlagen",
        "Computer friert ein", "Windows startet nicht",
    ],
    "Software & Applications": [
        "application will not open", "software keeps crashing",
        "cannot install a program", "application error",
        "program not responding", "software installation failed",
        "Programm öffnet sich nicht", "Software stürzt ab",
        "Programm lässt sich nicht installieren",
        "Anwendungsfehler", "Programm reagiert nicht",
    ],
    "Printers & Scanners": [
        "printer not working", "printer offline",
        "printer will not print", "paper jam",
        "scanner not detected", "printer driver error",
        "Drucker funktioniert nicht", "Drucker offline",
        "Drucker druckt nicht", "Papierstau",
        "Scanner wird nicht erkannt",
    ],
    "Hardware & Storage": [
        "computer will not turn on", "hard drive not detected",
        "usb device not recognized", "computer overheating",
        "keyboard not working", "mouse not working",
        "disk space full", "monitor no signal",
        "Computer startet nicht", "Festplatte wird nicht erkannt",
        "USB Gerät wird nicht erkannt", "Computer überhitzt",
        "Tastatur funktioniert nicht", "Maus funktioniert nicht",
        "Speicherplatz voll", "Monitor zeigt kein Signal",
    ],
    "Email & Communication": [
        "cannot send emails", "not receiving emails",
        "outlook will not open", "email not syncing",
        "cannot access mailbox", "microphone not working",
        "Ich kann keine E-Mails senden", "keine E-Mails erhalten",
        "Outlook öffnet sich nicht", "E-Mail synchronisiert nicht",
        "Kein Zugriff auf Postfach", "Mikrofon funktioniert nicht",
    ],
    "Security & VPN": [
        "vpn will not connect", "antivirus detected threat",
        "suspicious email", "account hacked",
        "firewall blocking application", "vpn disconnecting",
        "VPN verbindet sich nicht", "Antivirus hat Bedrohung erkannt",
        "verdächtige E-Mail", "Konto gehackt",
        "Firewall blockiert Programm", "VPN Verbindung bricht ab",
    ],
}

names_de = {
    "Network & Internet": "Netzwerk & Internet",
    "Login & Accounts": "Anmeldung & Konten",
    "Windows & Operating System": "Windows & Betriebssystem",
    "Software & Applications": "Software & Anwendungen",
    "Printers & Scanners": "Drucker & Scanner",
    "Hardware & Storage": "Hardware & Speicher",
    "Email & Communication": "E-Mail & Kommunikation",
    "Security & VPN": "Sicherheit & VPN",
}

solutions_en = {
    "Network & Internet": [
        "Restart your router and wait 30 seconds.",
        "Check that Wi-Fi is enabled and airplane mode is off.",
        "Restart your computer and reconnect.",
        "Check whether other devices can access the internet.",
    ],
    "Login & Accounts": [
        "Check your username and password.",
        "Check Caps Lock and your keyboard language.",
        "Use the official password-reset process.",
        "Contact your IT administrator if your account is locked.",
    ],
    "Windows & Operating System": [
        "Restart your computer.",
        "Install pending Windows updates if possible.",
        "Check your available disk space.",
        "Record any error messages if the issue continues.",
    ],
    "Software & Applications": [
        "Close the application completely and reopen it.",
        "Restart your computer.",
        "Check for application updates.",
        "Record the exact error message if the problem continues.",
    ],
    "Printers & Scanners": [
        "Check that the printer is powered on and connected.",
        "Make sure the correct printer is selected.",
        "Check for paper jams.",
        "Clear the print queue and restart the printer.",
    ],
    "Hardware & Storage": [
        "Check the power cable and connections.",
        "Restart the computer and reconnect the device.",
        "Check available storage space.",
        "Contact IT support if you suspect hardware failure.",
    ],
    "Email & Communication": [
        "Check your internet connection.",
        "Verify your email address and account settings.",
        "Restart your email application.",
        "Contact your email administrator if the issue continues.",
    ],
    "Security & VPN": [
        "Check your internet connection before reconnecting to VPN.",
        "Use the official VPN application and current credentials.",
        "Do not open suspicious links or attachments.",
        "Contact IT security immediately if you suspect a compromised account.",
    ],
}

solutions_de = {
    "Network & Internet": [
        "Starte deinen Router neu und warte 30 Sekunden.",
        "Prüfe, ob WLAN aktiviert und der Flugmodus ausgeschaltet ist.",
        "Starte deinen Computer neu und verbinde dich erneut.",
        "Prüfe, ob andere Geräte Internetzugriff haben.",
    ],
    "Login & Accounts": [
        "Prüfe deinen Benutzernamen und dein Passwort.",
        "Kontrolliere die Feststelltaste und die Tastatursprache.",
        "Nutze das offizielle Verfahren zum Zurücksetzen des Passworts.",
        "Kontaktiere bei einem gesperrten Konto die IT-Abteilung.",
    ],
    "Windows & Operating System": [
        "Starte deinen Computer neu.",
        "Installiere ausstehende Windows-Updates, sofern möglich.",
        "Prüfe den verfügbaren Speicherplatz.",
        "Notiere Fehlermeldungen, falls das Problem weiterhin besteht.",
    ],
    "Software & Applications": [
        "Schließe das Programm vollständig und öffne es erneut.",
        "Starte deinen Computer neu.",
        "Prüfe, ob Updates für das Programm verfügbar sind.",
        "Notiere die genaue Fehlermeldung, falls das Problem bleibt.",
    ],
    "Printers & Scanners": [
        "Prüfe, ob der Drucker eingeschaltet und angeschlossen ist.",
        "Stelle sicher, dass der richtige Drucker ausgewählt ist.",
        "Kontrolliere, ob ein Papierstau vorliegt.",
        "Leere die Druckwarteschlange und starte den Drucker neu.",
    ],
    "Hardware & Storage": [
        "Prüfe das Stromkabel und alle Anschlüsse.",
        "Starte den Computer neu und verbinde das Gerät erneut.",
        "Prüfe den verfügbaren Speicherplatz.",
        "Kontaktiere den IT-Support bei vermutetem Hardwaredefekt.",
    ],
    "Email & Communication": [
        "Prüfe deine Internetverbindung.",
        "Überprüfe deine E-Mail-Adresse und Kontoeinstellungen.",
        "Starte dein E-Mail-Programm neu.",
        "Kontaktiere die E-Mail-Administration, falls das Problem bleibt.",
    ],
    "Security & VPN": [
        "Prüfe deine Internetverbindung, bevor du VPN erneut verbindest.",
        "Verwende die offizielle VPN-Anwendung und aktuelle Zugangsdaten.",
        "Öffne keine verdächtigen Links oder Anhänge.",
        "Kontaktiere bei einem möglichen Kontomissbrauch sofort den IT-Support.",
    ],
}

# -------------------- Model --------------------

@st.cache_resource
def train_model():
    texts, labels = [], []

    for category, examples in data.items():
        texts.extend(examples)
        labels.extend([category] * len(examples))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(max_iter=1500)),
    ])

    pipeline.fit(texts, labels)
    return pipeline

model = train_model()

# -------------------- Header --------------------

st.markdown(f"""
<div class="hero">
    <h1>💻 IT Support AI</h1>
    <p>{T["description"]}</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Home page --------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown(f"## {T['title']}")

    st.markdown(f"### {T['input']}")

    problem = st.text_area(
        T["input"],
        placeholder=T["placeholder"],
        height=130,
        key="problem_input",
        label_visibility="collapsed",
    )

    if st.button(T["analyze"], use_container_width=True):
        if not problem.strip():
            st.warning(T["empty"])
        else:
            probabilities = model.predict_proba([problem])[0]
            best_index = probabilities.argmax()

            st.session_state.problem = problem.strip()
            st.session_state.predicted_category = model.classes_[best_index]
            st.session_state.confidence = float(probabilities[best_index])
            st.session_state.page = "results"
            st.rerun()

    st.markdown("---")
    st.markdown(f"### {T['categories']}")

    category_labels = [
        T["network"], T["login"], T["windows"], T["software"],
        T["printer"], T["hardware"], T["email"], T["security"],
    ]

    # Plain text list: no cards
    for label in category_labels:
        st.markdown(f"- {label}")

# -------------------- Results page --------------------

elif st.session_state.page == "results":
    category = st.session_state.predicted_category
    display_name = names_de[category] if de else category
    solutions = solutions_de if de else solutions_en

    st.markdown(f"## {T['result']}")
    st.markdown(f"### {T['predicted']}")
    st.markdown(f"## {display_name}")
    st.caption(f"{T['confidence']}: {st.session_state.confidence:.0%}")

    st.markdown(f"### {T['solutions']}")

    for number, solution in enumerate(solutions[category], start=1):
        st.markdown(f"{number}. {solution}")

    st.markdown("---")

    if st.button(T["back"], use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# -------------------- Footer --------------------

st.markdown(
    f'<div class="footer">{T["footer"]}</div>',
    unsafe_allow_html=True,
)
