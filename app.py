


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IT Support AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

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
    color: #ffffff;
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.hero p {
    color: #c2d1e3;
    font-size: 1.05rem;
}

.section-card {
    background: #111f33;
    border: 1px solid #263b56;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 12px;
}

.section-card h3 {
    color: #eaf2ff;
    font-size: 1.05rem;
    margin-bottom: 8px;
}

.section-card p {
    color: #a9bad0;
    font-size: 0.92rem;
}

.result-card {
    background: #111f33;
    border: 1px solid #294363;
    border-radius: 14px;
    padding: 24px;
    margin-top: 16px;
    margin-bottom: 16px;
}

.footer {
    text-align: center;
    color: #8295ad;
    border-top: 1px solid #26364b;
    padding-top: 20px;
    margin-top: 40px;
    font-size: 0.85rem;
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
    border: none;
}

h2, h3 {
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Language selection
# --------------------------------------------------

if "language" not in st.session_state:
    st.session_state.language = "English"

language = st.selectbox(
    "Language / Sprache",
    ["English", "Deutsch"],
    index=0 if st.session_state.language == "English" else 1,
    key="language_selector",
)

st.session_state.language = language
is_german = language == "Deutsch"

TEXT = {
    "title": "Wie können wir helfen?" if is_german
             else "How can we help you?",
    "description": (
        "Beschreibe dein IT-Problem. Die KI ordnet es einer Kategorie zu "
        "und schlägt mögliche Lösungen vor."
        if is_german else
        "Describe your technical issue. The AI will predict a category "
        "and suggest possible troubleshooting steps."
    ),
    "categories": "IT-Support-Kategorien" if is_german
                  else "IT Support Categories",
    "input_title": "Beschreibe dein Problem" if is_german
                   else "Describe Your Problem",
    "placeholder": "Zum Beispiel: Ich kann keine Verbindung zum WLAN herstellen."
                   if is_german else
                   "For example: I cannot connect to the Wi-Fi.",
    "analyze": "Problem analysieren" if is_german else "Analyze Problem",
    "empty": "Bitte beschreibe zuerst dein Problem." if is_german
             else "Please describe your problem first.",
    "result": "Analyseergebnis" if is_german else "Analysis Result",
    "category": "Erkannte Kategorie" if is_german else "Predicted Category",
    "solutions": "Lösungsvorschläge" if is_german else "Suggested Solutions",
    "confidence": "Modell-Konfidenz" if is_german else "Model Confidence",
    "back": "Zurück zur Startseite" if is_german else "Back to Home",
    "footer": "Entwickelt von Abdullah Aitah" if is_german
              else "Developed by Abdullah Aitah",
}

# --------------------------------------------------
# Training examples
# --------------------------------------------------

training_data = {
    "Network & Internet": [
        "my wifi is not working",
        "I cannot connect to the internet",
        "internet connection keeps dropping",
        "network is unavailable",
        "my computer cannot connect to wifi",
        "wifi says connected but there is no internet",
        "the ethernet connection is not working",
        "my internet is very slow",
        "I have a DNS error",
        "the network adapter is not working",
        "Ich habe keine WLAN-Verbindung",
        "Mein Internet funktioniert nicht",
        "Die Internetverbindung bricht ständig ab",
        "Mein WLAN ist sehr langsam",
        "Mein Computer findet das Netzwerk nicht",
        "Ich habe einen DNS-Fehler",
    ],
    "Login & Accounts": [
        "I cannot log in to my account",
        "my password is not working",
        "I forgot my password",
        "my account is locked",
        "login failed",
        "I cannot access my user account",
        "I am getting an authentication error",
        "my account credentials are incorrect",
        "I cannot sign in to Windows",
        "my account is disabled",
        "Ich kann mich nicht anmelden",
        "Mein Passwort funktioniert nicht",
        "Ich habe mein Passwort vergessen",
        "Mein Konto wurde gesperrt",
        "Die Anmeldung ist fehlgeschlagen",
        "Ich kann nicht auf mein Benutzerkonto zugreifen",
    ],
    "Windows & Operating System": [
        "my computer is running very slowly",
        "Windows keeps crashing",
        "my computer has a blue screen",
        "Windows update failed",
        "my computer freezes randomly",
        "Windows will not start",
        "my system keeps restarting",
        "I have an operating system error",
        "my computer takes too long to boot",
        "Windows is not responding",
        "Mein Computer ist sehr langsam",
        "Windows stürzt ständig ab",
        "Mein Computer zeigt einen Bluescreen",
        "Das Windows-Update ist fehlgeschlagen",
        "Mein Computer friert ständig ein",
        "Windows startet nicht",
    ],
    "Software & Applications": [
        "an application will not open",
        "my software keeps crashing",
        "I cannot install a program",
        "an application is not responding",
        "the software installation failed",
        "my application shows an error message",
        "a program is running slowly",
        "I cannot uninstall an application",
        "my software is incompatible",
        "an application closes unexpectedly",
        "Ein Programm lässt sich nicht öffnen",
        "Meine Software stürzt ständig ab",
        "Ich kann kein Programm installieren",
        "Eine Anwendung reagiert nicht",
        "Die Softwareinstallation ist fehlgeschlagen",
        "Ein Programm zeigt eine Fehlermeldung",
    ],
    "Printers & Scanners": [
        "my printer is not working",
        "the printer is offline",
        "my printer will not print",
        "the printer has a paper jam",
        "my scanner is not detected",
        "printer driver installation failed",
        "my printer prints blank pages",
        "documents are stuck in the print queue",
        "the printer is not connected",
        "my printer has a connection error",
        "Mein Drucker funktioniert nicht",
        "Der Drucker ist offline",
        "Mein Drucker druckt nicht",
        "Das Papier steckt im Drucker fest",
        "Mein Scanner wird nicht erkannt",
        "Die Druckertreiberinstallation ist fehlgeschlagen",
    ],
    "Hardware & Storage": [
        "my computer does not turn on",
        "my hard drive is not detected",
        "my USB device is not recognized",
        "my computer is overheating",
        "my keyboard is not working",
        "my mouse is not working",
        "I am running out of disk space",
        "my monitor shows no signal",
        "my laptop battery is not charging",
        "my computer makes strange noises",
        "Mein Computer lässt sich nicht einschalten",
        "Meine Festplatte wird nicht erkannt",
        "Mein USB-Gerät wird nicht erkannt",
        "Mein Computer überhitzt",
        "Meine Tastatur funktioniert nicht",
        "Meine Maus funktioniert nicht",
        "Mein Speicherplatz ist fast voll",
        "Mein Monitor zeigt kein Signal",
    ],
    "Email & Communication": [
        "I cannot send emails",
        "I am not receiving emails",
        "my email account is not syncing",
        "Outlook will not open",
        "my email attachments will not download",
        "my emails are going to spam",
        "I cannot connect to the mail server",
        "video calls have no sound",
        "my microphone is not working in meetings",
        "I cannot access my mailbox",
        "Ich kann keine E-Mails senden",
        "Ich empfange keine E-Mails",
        "Mein E-Mail-Konto synchronisiert nicht",
        "Outlook lässt sich nicht öffnen",
        "E-Mail-Anhänge lassen sich nicht herunterladen",
        "Ich kann keine Verbindung zum Mailserver herstellen",
    ],
    "Security & VPN": [
        "my VPN will not connect",
        "I cannot access the company VPN",
        "my antivirus detected a threat",
        "I received a suspicious email",
        "I think my account was hacked",
        "my firewall is blocking an application",
        "I cannot connect securely to the network",
        "my VPN connection keeps disconnecting",
        "I received a phishing message",
        "my security software is not working",
        "Mein VPN verbindet sich nicht",
        "Ich kann keine Verbindung zum Firmen-VPN herstellen",
        "Mein Antivirenprogramm hat eine Bedrohung erkannt",
        "Ich habe eine verdächtige E-Mail erhalten",
        "Ich glaube, mein Konto wurde gehackt",
        "Meine Firewall blockiert ein Programm",
    ],
}

# --------------------------------------------------
# Troubleshooting solutions
# --------------------------------------------------

solutions_en = {
    "Network & Internet": [
        "Restart your router and wait 30 seconds before reconnecting.",
        "Check that Wi-Fi is enabled and airplane mode is turned off.",
        "Restart your computer and try connecting again.",
        "If the issue continues, check whether other devices can access the internet.",
    ],
    "Login & Accounts": [
        "Check that your username and password are entered correctly.",
        "Check Caps Lock and your keyboard language.",
        "Use the official password-reset process if you forgot your password.",
        "If the account is locked, contact your IT administrator.",
    ],
    "Windows & Operating System": [
        "Restart your computer and check whether the issue persists.",
        "Install pending Windows updates if the computer starts normally.",
        "Check available disk space.",
        "If blue screens continue, record the error message and contact IT support.",
    ],
    "Software & Applications": [
        "Close the application completely and open it again.",
        "Restart your computer and try again.",
        "Check whether an update is available for the application.",
        "If the problem continues, record the exact error message.",
    ],
    "Printers & Scanners": [
        "Check that the printer is powered on and connected.",
        "Make sure the correct printer is selected.",
        "Check for paper jams and sufficient paper.",
        "Clear the print queue and restart the printer if necessary.",
    ],
    "Hardware & Storage": [
        "Check the power cable and connections.",
        "Restart the computer and reconnect the device.",
        "Check available storage space.",
        "If you hear unusual drive noises or suspect hardware failure, stop using the device and contact IT support.",
    ],
    "Email & Communication": [
        "Check your internet connection.",
        "Verify the email address and account settings.",
        "Restart the email application and check for updates.",
        "If the problem continues, contact your email administrator.",
    ],
    "Security & VPN": [
        "Check your internet connection before reconnecting to the VPN.",
        "Verify that you are using the official VPN application and current credentials.",
        "Do not open suspicious links or attachments.",
        "If you suspect a compromised account, contact IT or security support immediately.",
    ],
}

solutions_de = {
    "Network & Internet": [
        "Starte deinen Router neu und warte 30 Sekunden, bevor du dich erneut verbindest.",
        "Prüfe, ob WLAN aktiviert und der Flugmodus ausgeschaltet ist.",
        "Starte deinen Computer neu und versuche erneut, eine Verbindung herzustellen.",
        "Prüfe, ob andere Geräte auf das Internet zugreifen können.",
    ],
    "Login & Accounts": [
        "Prüfe, ob Benutzername und Passwort korrekt eingegeben wurden.",
        "Kontrolliere die Feststelltaste und die Tastatursprache.",
        "Nutze das offizielle Verfahren zum Zurücksetzen des Passworts.",
        "Wenn dein Konto gesperrt ist, kontaktiere die IT-Abteilung.",
    ],
    "Windows & Operating System": [
        "Starte deinen Computer neu und prüfe, ob das Problem weiterhin besteht.",
        "Installiere ausstehende Windows-Updates, sofern der Computer normal startet.",
        "Prüfe den verfügbaren Speicherplatz.",
        "Notiere bei wiederholten Bluescreens die Fehlermeldung und kontaktiere den IT-Support.",
    ],
    "Software & Applications": [
        "Schließe das Programm vollständig und öffne es erneut.",
        "Starte deinen Computer neu und versuche es noch einmal.",
        "Prüfe, ob ein Update für die Anwendung verfügbar ist.",
        "Notiere die genaue Fehlermeldung, falls das Problem weiterhin besteht.",
    ],
    "Printers & Scanners": [
        "Prüfe, ob der Drucker eingeschaltet und angeschlossen ist.",
        "Stelle sicher, dass der richtige Drucker ausgewählt ist.",
        "Kontrolliere Papierstau und Papiervorrat.",
        "Leere bei Bedarf die Druckwarteschlange und starte den Drucker neu.",
    ],
    "Hardware & Storage": [
        "Prüfe das Stromkabel und alle Anschlüsse.",
        "Starte den Computer neu und schließe das Gerät erneut an.",
        "Kontrolliere den verfügbaren Speicherplatz.",
        "Bei ungewöhnlichen Festplattengeräuschen oder vermutetem Hardwaredefekt solltest du das Gerät nicht weiter verwenden und den IT-Support kontaktieren.",
    ],
    "Email & Communication": [
        "Prüfe deine Internetverbindung.",
        "Überprüfe die E-Mail-Adresse und die Kontoeinstellungen.",
        "Starte das E-Mail-Programm neu und suche nach Updates.",
        "Wenn das Problem weiterhin besteht, kontaktiere deine E-Mail-Administration.",
    ],
    "Security & VPN": [
        "Prüfe deine Internetverbindung, bevor du dich erneut mit dem VPN verbindest.",
        "Stelle sicher, dass du die offizielle VPN-Anwendung und aktuelle Zugangsdaten verwendest.",
        "Öffne keine verdächtigen Links oder Anhänge.",
        "Wenn du einen Kontomissbrauch vermutest, kontaktiere sofort den IT- oder Sicherheits-Support.",
    ],
}

category_names_de = {
    "Network & Internet": "Netzwerk & Internet",
    "Login & Accounts": "Anmeldung & Konten",
    "Windows & Operating System": "Windows & Betriebssystem",
    "Software & Applications": "Software & Anwendungen",
    "Printers & Scanners": "Drucker & Scanner",
    "Hardware & Storage": "Hardware & Speicher",
    "Email & Communication": "E-Mail & Kommunikation",
    "Security & VPN": "Sicherheit & VPN",
}

category_descriptions_en = {
    "Network & Internet": "Wi-Fi, internet and network connection issues.",
    "Login & Accounts": "Passwords, sign-in problems and locked accounts.",
    "Windows & Operating System": "Windows errors, crashes and system performance.",
    "Software & Applications": "Applications, installations and software errors.",
    "Printers & Scanners": "Printing, scanning and device connection problems.",
    "Hardware & Storage": "Computer hardware, USB devices and storage.",
    "Email & Communication": "Email delivery, Outlook and communication tools.",
    "Security & VPN": "VPN connectivity and common security concerns.",
}

category_descriptions_de = {
    "Network & Internet": "Probleme mit WLAN, Internet und Netzwerkverbindungen.",
    "Login & Accounts": "Passwörter, Anmeldeprobleme und gesperrte Konten.",
    "Windows & Operating System": "Windows-Fehler, Abstürze und Systemleistung.",
    "Software & Applications": "Anwendungen, Installationen und Softwarefehler.",
    "Printers & Scanners": "Drucken, Scannen und Geräteverbindungen.",
    "Hardware & Storage": "Computerhardware, USB-Geräte und Speicher.",
    "Email & Communication": "E-Mail-Versand, Outlook und Kommunikation.",
    "Security & VPN": "VPN-Verbindungen und allgemeine Sicherheitsprobleme.",
}

category_icons = {
    "Network & Internet": "🌐",
    "Login & Accounts": "🔐",
    "Windows & Operating System": "💻",
    "Software & Applications": "🧩",
    "Printers & Scanners": "🖨️",
    "Hardware & Storage": "⚙️",
    "Email & Communication": "📧",
    "Security & VPN": "🛡️",
}

# --------------------------------------------------
# Train the model
# --------------------------------------------------

@st.cache_resource
def train_model():
    texts = []
    labels = []

    for category, examples in training_data.items():
        texts.extend(examples)
        labels.extend([category] * len(examples))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(max_iter=1500)),
    ])

    model.fit(texts, labels)
    return model

model = train_model()

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "problem" not in st.session_state:
    st.session_state.problem = ""

if "predicted_category" not in st.session_state:
    st.session_state.predicted_category = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(f"""
<div class="hero">
    <h1>💻 IT Support AI</h1>
    <p>{TEXT["description"]}</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Home page
# --------------------------------------------------

if st.session_state.page == "home":

    st.markdown(f"## {TEXT['title']}")
    st.markdown(f"### {TEXT['categories']}")

    descriptions = (
        category_descriptions_de if is_german
        else category_descriptions_en
    )

    categories = list(training_data.keys())

    for start in range(0, len(categories), 2):
        cols = st.columns(2)

        for col, category in zip(cols, categories[start:start + 2]):
            with col:
                display_name = (
                    category_names_de[category]
                    if is_german else category
                )

                st.markdown(f"""
                <div class="section-card">
                    <h3>{category_icons[category]} {display_name}</h3>
                    <p>{descriptions[category]}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {TEXT['input_title']}")

    problem = st.text_area(
        TEXT["input_title"],
        placeholder=TEXT["placeholder"],
        height=130,
        key="problem_input",
        label_visibility="collapsed",
    )

    if st.button(TEXT["analyze"], use_container_width=True):
        if not problem.strip():
            st.warning(TEXT["empty"])
        else:
            probabilities = model.predict_proba([problem])[0]
            best_index = probabilities.argmax()

            st.session_state.problem = problem.strip()
            st.session_state.predicted_category = model.classes_[best_index]
            st.session_state.confidence = float(probabilities[best_index])
            st.session_state.page = "results"
            st.rerun()

# --------------------------------------------------
# Results page
# --------------------------------------------------

elif st.session_state.page == "results":

    category = st.session_state.predicted_category

    display_name = (
        category_names_de[category] if is_german else category
    )

    solution_data = solutions_de if is_german else solutions_en

    st.markdown(f"## {TEXT['result']}")

    st.markdown(f"""
    <div class="result-card">
        <h3>{TEXT['category']}</h3>
        <h2>{category_icons[category]} {display_name}</h2>
        <p>{TEXT['confidence']}: {st.session_state.confidence:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### {TEXT['solutions']}")

    for index, solution in enumerate(solution_data[category], start=1):
        st.markdown(f"""
        <div class="section-card">
            <p><strong>{index}.</strong> {solution}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    if st.button(TEXT["back"], use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    f'<div class="footer">{TEXT["footer"]}</div>',
    unsafe_allow_html=True,
)
