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
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu {
    visibility: hidden !important;
    height: 0 !important;
}
.stApp {
    background: radial-gradient(ellipse at 12% 0%, rgba(37,99,235,.16), transparent 38%),
                linear-gradient(135deg, #07111f 0%, #0b1728 52%, #101e35 100%);
    color: #edf4ff;
}
.block-container { max-width: 1000px; padding-top: 1rem !important; padding-bottom: 2.5rem; }
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(115deg, rgba(20,43,73,.98), rgba(12,28,49,.98));
    border: 1px solid rgba(105,155,220,.28); border-radius: 22px;
    padding: 30px 32px; margin: 12px 0 25px 0;
    box-shadow: 0 16px 42px rgba(0,0,0,.18);
}
.hero h1 { color: #fff; font-size: clamp(2rem,5vw,2.7rem); letter-spacing: -.04em; margin: 0 0 8px 0; }
.hero p { color: #c3d3e8; font-size: 1.04rem; line-height: 1.65; margin: 0; max-width: 680px; }
h2, h3 { color: #f1f6ff; letter-spacing: -.02em; }
p, li, label { color: #d3deec; }
hr { border-color: rgba(122,151,190,.22); margin: 1.6rem 0; }
.stTextArea textarea {
    background: rgba(8,20,36,.92) !important; color: #f1f5f9 !important;
    border: 1px solid #385575 !important; border-radius: 14px !important;
    padding: 15px !important; line-height: 1.6 !important;
}
.stTextArea textarea:focus { border-color: #4b9aff !important; box-shadow: 0 0 0 2px rgba(59,130,246,.16) !important; }
div.stButton > button {
    background: linear-gradient(100deg,#2477e8,#368df5); color: #fff;
    border: 1px solid rgba(139,190,255,.25); border-radius: 11px;
    min-height: 48px; padding: .7rem 1.2rem; font-weight: 650; transition: .18s ease;
}
div.stButton > button:hover { background: linear-gradient(100deg,#3286f1,#55a0ff); color: #fff; border-color: rgba(180,215,255,.5); }
div[data-baseweb="select"] > div { background-color: #0c1a2c; border-color: #314b69; border-radius: 10px; }
div[data-baseweb="select"] * { color: #eaf2ff; }
.result-panel {
    background: rgba(15,31,52,.88); border: 1px solid rgba(92,137,190,.35);
    border-left: 4px solid #368df5; border-radius: 14px; padding: 22px 24px; margin: 14px 0 20px 0;
}
.result-panel .eyebrow { color: #8fbfff; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 8px; }
.result-panel .category { color: #fff; font-size: 1.5rem; font-weight: 700; }
.footer { text-align: center; color: #8295ad; border-top: 1px solid rgba(122,151,190,.2); padding-top: 18px; margin-top: 38px; font-size: .85rem; }
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
    st.caption("✦  IT help, made simpler" if not de else "✦  IT-Hilfe, einfach erklärt")

    # Search box appears near the beginning of the page
    st.markdown(f"### {T['input']}")
    problem = st.text_area(
        T["input"],
        placeholder=T["placeholder"],
        height=125,
        key="problem_input",
        label_visibility="collapsed",
    )

    if st.button("⌕  " + T["analyze"], use_container_width=True):
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

    # Simple text list in two columns — no cards
    left, right = st.columns(2)
    for index, label in enumerate(category_labels):
        with (left if index % 2 == 0 else right):
            st.markdown(f"- {label}")

# -------------------- Results page --------------------

elif st.session_state.page == "results":
    category = st.session_state.predicted_category
    display_name = names_de[category] if de else category
    solutions = solutions_de if de else solutions_en

    st.markdown(f"## {T['result']}")
    st.markdown(f"""
    <div class="result-panel">
        <div class="eyebrow">{T['predicted']}</div>
        <div class="category">{display_name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"{T['confidence']}: {st.session_state.confidence:.0%}")
    st.progress(st.session_state.confidence)
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
