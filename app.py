


import html
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="IT Support Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# TRANSLATIONS
# =====================================================
TEXT = {
    "en": {
        "language": "Language / Sprache",
        "title": "IT Support Assistant",
        "eyebrow": "CUSTOMER SUPPORT CENTER",
        "tagline": "Technical support made simpler.",
        "description": (
            "Describe your technical issue and receive practical "
            "guidance to help you resolve it."
        ),
        "question": "How can we help you today?",
        "instructions": (
            "Describe the technical issue you are experiencing. "
            "Include any error message if one appears."
        ),
        "problem_label": "Problem description",
        "placeholder": (
            "Example: My laptop connects to Wi-Fi, "
            "but websites do not load..."
        ),
        "button": "Find a solution →",
        "short_error": "Please describe your issue in a little more detail.",
        "results": "Your Support Results",
        "results_desc": "Review the recommended steps for your issue.",
        "reported": "YOUR REPORTED ISSUE",
        "category": "Issue category",
        "steps": "Recommended next steps",
        "low_match": (
            "We could not find a strong match. The suggested category "
            "may not be correct. Try adding your device name and exact error."
        ),
        "disclaimer": (
            "These suggestions are general guidance, not a confirmed "
            "diagnosis. Follow your organization's IT policies."
        ),
        "back": "← Report another issue",
        "footer": "Customer IT Support",
        "home": "Home",
        "invalid": "Please enter a description of your problem.",
        "categories": {
            "Network & Internet": "Network & Internet",
            "Login & Accounts": "Login & Accounts",
            "Windows & Operating System": "Windows & Operating System",
            "Software & Applications": "Software & Applications",
            "Printers & Scanners": "Printers & Scanners",
            "Hardware & Storage": "Hardware & Storage",
            "Email & Communication": "Email & Communication",
            "Security & VPN": "Security & VPN"
        }
    },
    "de": {
        "language": "Sprache / Language",
        "title": "IT-Support-Assistent",
        "eyebrow": "KUNDENSUPPORT",
        "tagline": "Technischer Support – einfach und verständlich.",
        "description": (
            "Beschreiben Sie Ihr technisches Problem und erhalten Sie "
            "praktische Hinweise zur Fehlerbehebung."
        ),
        "question": "Wie können wir Ihnen heute helfen?",
        "instructions": (
            "Beschreiben Sie Ihr technisches Problem. "
            "Geben Sie, wenn möglich, auch die Fehlermeldung an."
        ),
        "problem_label": "Problembeschreibung",
        "placeholder": (
            "Beispiel: Mein Laptop verbindet sich mit dem WLAN, "
            "aber Webseiten werden nicht geladen ..."
        ),
        "button": "Lösung finden →",
        "short_error": "Bitte beschreiben Sie Ihr Problem etwas genauer.",
        "results": "Ihre Support-Ergebnisse",
        "results_desc": "Hier finden Sie empfohlene Schritte zur Fehlerbehebung.",
        "reported": "BESCHRIEBENES PROBLEM",
        "category": "Problemkategorie",
        "steps": "Empfohlene nächste Schritte",
        "low_match": (
            "Es wurde keine ausreichend passende Beschreibung gefunden. "
            "Die vorgeschlagene Kategorie kann falsch sein. "
            "Ergänzen Sie den Gerätenamen und die genaue Fehlermeldung."
        ),
        "disclaimer": (
            "Diese Hinweise sind allgemeine Empfehlungen und keine "
            "bestätigte Diagnose. Beachten Sie die IT-Richtlinien "
            "Ihres Unternehmens."
        ),
        "back": "← Weiteres Problem melden",
        "footer": "IT-Kundensupport",
        "home": "Startseite",
        "invalid": "Bitte beschreiben Sie Ihr technisches Problem.",
        "categories": {
            "Network & Internet": "Netzwerk & Internet",
            "Login & Accounts": "Anmeldung & Konten",
            "Windows & Operating System": "Windows & Betriebssystem",
            "Software & Applications": "Software & Anwendungen",
            "Printers & Scanners": "Drucker & Scanner",
            "Hardware & Storage": "Hardware & Speicher",
            "Email & Communication": "E-Mail & Kommunikation",
            "Security & VPN": "Sicherheit & VPN"
        }
    }
}

# =====================================================
# BILINGUAL TRAINING EXAMPLES
# =====================================================
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
        "I cannot connect to the company network",
        "Internet works on my phone but not my laptop",
        "Mein WLAN funktioniert nicht",
        "Ich habe keine Internetverbindung",
        "Das Internet ist sehr langsam",
        "Webseiten werden nicht geladen",
        "Ich habe einen DNS-Fehler",
        "Mein Computer hat keine IP-Adresse",
        "Die Netzwerkverbindung bricht ständig ab",
        "WLAN verbunden, aber kein Internet",
        "Ich kann mich nicht mit dem Firmennetzwerk verbinden",
        "Das Internet funktioniert auf dem Handy, aber nicht am Laptop"
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
        "My account has been disabled",
        "I am asked to sign in repeatedly",
        "Ich kann mich nicht anmelden",
        "Mein Passwort funktioniert nicht",
        "Mein Konto ist gesperrt",
        "Ich habe mein Passwort vergessen",
        "Zugriff verweigert",
        "Ich kann nicht auf einen freigegebenen Ordner zugreifen",
        "Mein Benutzername wird nicht erkannt",
        "Ich kann mich nicht bei Windows anmelden",
        "Mein Konto wurde deaktiviert",
        "Ich muss mich ständig neu anmelden"
    ],
    "Windows & Operating System": [
        "My computer is very slow",
        "Windows update failed",
        "My PC shows a blue screen",
        "Windows will not start",
        "My computer keeps restarting",
        "Windows is frozen",
        "My taskbar is not responding",
        "Windows update is stuck",
        "My computer crashes frequently",
        "File Explorer is not responding",
        "Mein Computer ist sehr langsam",
        "Das Windows-Update ist fehlgeschlagen",
        "Mein PC zeigt einen Bluescreen",
        "Windows startet nicht",
        "Mein Computer startet ständig neu",
        "Windows ist eingefroren",
        "Meine Taskleiste reagiert nicht",
        "Das Windows-Update hängt",
        "Mein Computer stürzt häufig ab",
        "Der Datei-Explorer reagiert nicht"
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
        "Meine Anwendung lässt sich nicht öffnen",
        "Das Programm stürzt ständig ab",
        "Ich kann keine Software installieren",
        "Die Anwendung zeigt einen Fehler",
        "Mein Browser reagiert nicht",
        "Die Softwareinstallation ist fehlgeschlagen",
        "Eine Anwendung friert ständig ein",
        "Mein Programm ist nicht kompatibel",
        "Office-Anwendungen funktionieren nicht",
        "Die Anwendung schließt sich unerwartet"
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
        "Mein Drucker funktioniert nicht",
        "Der Drucker ist offline",
        "Ich kann keine Dokumente drucken",
        "Der Drucker hat einen Papierstau",
        "Mein Drucker wird nicht erkannt",
        "Der Scanner funktioniert nicht",
        "Mein Drucker druckt leere Seiten",
        "Die Druckwarteschlange hängt",
        "Der Druckertreiber fehlt",
        "Ich kann mich nicht mit dem Netzwerkdrucker verbinden"
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
        "My laptop battery drains quickly",
        "My laptop will not charge",
        "Meine Festplatte wird nicht erkannt",
        "Meine Festplatte ist voll",
        "Mein Laptop überhitzt",
        "Mein Computer hat keinen Ton",
        "Meine Tastatur funktioniert nicht",
        "Meine Maus reagiert nicht",
        "Mein Monitor zeigt kein Signal",
        "Mein USB-Gerät wird nicht erkannt",
        "Der Laptop-Akku ist schnell leer",
        "Mein Laptop lädt nicht"
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
        "Ich kann keine E-Mails senden",
        "Ich empfange keine E-Mails",
        "Outlook lässt sich nicht öffnen",
        "E-Mail-Anhänge lassen sich nicht öffnen",
        "Mein Postfach ist voll",
        "E-Mails bleiben im Postausgang hängen",
        "Ich kann keine Verbindung zu Outlook herstellen",
        "Mein E-Mail-Programm fragt ständig nach einem Passwort",
        "Bei Videokonferenzen gibt es keinen Ton",
        "Mein Mikrofon funktioniert in Besprechungen nicht"
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
        "I think I clicked a phishing link",
        "Mein VPN verbindet sich nicht",
        "Ich kann nicht auf das Firmen-VPN zugreifen",
        "Mein Virenschutz zeigt eine Warnung",
        "Ich habe eine verdächtige E-Mail erhalten",
        "Mein Sicherheitszertifikat ist abgelaufen",
        "Die Firewall blockiert meine Anwendung",
        "Meine VPN-Verbindung bricht ständig ab",
        "Windows Defender hat eine Bedrohung erkannt",
        "Ich glaube, mein Konto wurde kompromittiert",
        "Ich glaube, ich habe auf einen Phishing-Link geklickt"
    ]
}

# =====================================================
# BILINGUAL SOLUTIONS
# =====================================================
solutions = {
    "Network & Internet": {
        "en": [
            "Check whether Wi-Fi is enabled and airplane mode is off.",
            "Check whether other devices can use the same network.",
            "Disconnect and reconnect to the network if possible.",
            "On Windows, run ipconfig in Command Prompt to inspect network settings.",
            "If the problem continues, record the error message and contact IT Support."
        ],
        "de": [
            "Prüfen Sie, ob WLAN aktiviert und der Flugmodus ausgeschaltet ist.",
            "Prüfen Sie, ob andere Geräte dasselbe Netzwerk nutzen können.",
            "Trennen Sie die Netzwerkverbindung und verbinden Sie sich erneut.",
            "Unter Windows können Sie mit ipconfig in der Eingabeaufforderung die Netzwerkkonfiguration prüfen.",
            "Notieren Sie bei anhaltenden Problemen die Fehlermeldung und wenden Sie sich an den IT-Support."
        ]
    },
    "Login & Accounts": {
        "en": [
            "Check your username, keyboard layout and Caps Lock.",
            "Use your organization's official password-reset process if needed.",
            "Check whether the sign-in service is available.",
            "Never share your password or verification codes.",
            "Contact IT Support for locked accounts or permission problems."
        ],
        "de": [
            "Prüfen Sie Benutzernamen, Tastaturlayout und Feststelltaste.",
            "Nutzen Sie bei Bedarf das offizielle Verfahren zum Zurücksetzen des Passworts.",
            "Prüfen Sie, ob der Anmeldedienst verfügbar ist.",
            "Geben Sie niemals Ihr Passwort oder Ihre Bestätigungscodes weiter.",
            "Wenden Sie sich bei gesperrten Konten oder fehlenden Berechtigungen an den IT-Support."
        ]
    },
    "Windows & Operating System": {
        "en": [
            "Save your work and restart the computer if safe.",
            "Check available disk space and pending Windows updates.",
            "Record any error code or blue-screen message.",
            "Avoid recovery operations if you are unsure of their effects.",
            "Contact IT Support if startup failures or crashes continue."
        ],
        "de": [
            "Speichern Sie Ihre Arbeit und starten Sie den Computer neu, sofern dies sicher ist.",
            "Prüfen Sie den verfügbaren Speicherplatz und ausstehende Windows-Updates.",
            "Notieren Sie Fehlercodes oder Meldungen auf einem Bluescreen.",
            "Führen Sie keine Wiederherstellungsmaßnahmen durch, deren Auswirkungen Sie nicht kennen.",
            "Wenden Sie sich bei wiederholten Startproblemen oder Abstürzen an den IT-Support."
        ]
    },
    "Software & Applications": {
        "en": [
            "Close the application completely and reopen it.",
            "Check whether the issue affects only this application.",
            "Check for approved application updates.",
            "Record the exact error message.",
            "Contact IT Support if the problem continues."
        ],
        "de": [
            "Schließen Sie die Anwendung vollständig und öffnen Sie sie erneut.",
            "Prüfen Sie, ob nur diese Anwendung betroffen ist.",
            "Prüfen Sie, ob freigegebene Updates verfügbar sind.",
            "Notieren Sie die genaue Fehlermeldung.",
            "Wenden Sie sich an den IT-Support, wenn das Problem weiterhin besteht."
        ]
    },
    "Printers & Scanners": {
        "en": [
            "Check power, cables, Wi-Fi and the printer display.",
            "Confirm that the correct printer is selected.",
            "Check paper, toner and the print queue.",
            "Try a test page if permitted by your organization.",
            "Contact IT Support if the driver or connection may be faulty."
        ],
        "de": [
            "Prüfen Sie Stromversorgung, Kabel, WLAN und die Druckeranzeige.",
            "Stellen Sie sicher, dass der richtige Drucker ausgewählt ist.",
            "Prüfen Sie Papier, Toner und die Druckwarteschlange.",
            "Drucken Sie eine Testseite, sofern dies in Ihrem Unternehmen erlaubt ist.",
            "Wenden Sie sich bei Treiber- oder Verbindungsproblemen an den IT-Support."
        ]
    },
    "Hardware & Storage": {
        "en": [
            "Check power, cables and device connections.",
            "Inspect available storage and close unused applications.",
            "Keep ventilation openings clear if the device is overheating.",
            "Back up important files when possible.",
            "Contact IT Support for persistent hardware problems."
        ],
        "de": [
            "Prüfen Sie Stromversorgung, Kabel und Geräteverbindungen.",
            "Prüfen Sie den verfügbaren Speicherplatz und schließen Sie nicht benötigte Anwendungen.",
            "Halten Sie Lüftungsöffnungen frei, wenn das Gerät überhitzt.",
            "Sichern Sie wichtige Dateien, sofern möglich.",
            "Wenden Sie sich bei anhaltenden Hardwareproblemen an den IT-Support."
        ]
    },
    "Email & Communication": {
        "en": [
            "Check your internet connection and email service availability.",
            "Verify the recipient address and attachment size.",
            "Restart the email application and check approved updates.",
            "For meeting audio issues, check the selected microphone and speakers.",
            "Never share your password; contact IT Support if sign-in keeps failing."
        ],
        "de": [
            "Prüfen Sie Ihre Internetverbindung und die Verfügbarkeit des E-Mail-Dienstes.",
            "Überprüfen Sie die Empfängeradresse und die Größe des Anhangs.",
            "Starten Sie das E-Mail-Programm neu und prüfen Sie freigegebene Updates.",
            "Prüfen Sie bei Tonproblemen in Besprechungen das ausgewählte Mikrofon und die Lautsprecher.",
            "Geben Sie Ihr Passwort niemals weiter und wenden Sie sich bei Anmeldeproblemen an den IT-Support."
        ]
    },
    "Security & VPN": {
        "en": [
            "Use only your organization's official VPN application.",
            "Check your internet connection and VPN account authorization.",
            "Do not open suspicious links or attachments.",
            "Never disable antivirus or firewall protection to bypass an error.",
            "Report suspected phishing or account compromise to IT immediately."
        ],
        "de": [
            "Verwenden Sie ausschließlich die offizielle VPN-Anwendung Ihres Unternehmens.",
            "Prüfen Sie Ihre Internetverbindung und Ihre VPN-Zugriffsberechtigung.",
            "Öffnen Sie keine verdächtigen Links oder Anhänge.",
            "Deaktivieren Sie niemals Virenschutz oder Firewall, um eine Fehlermeldung zu umgehen.",
            "Melden Sie vermutetes Phishing oder kompromittierte Konten sofort der IT-Abteilung."
        ]
    }
}

# =====================================================
# MODEL TRAINING
# =====================================================
problems = []
categories = []

for category, examples in training_data.items():
    problems.extend(examples)
    categories.extend([category] * len(examples))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    strip_accents="unicode",
    sublinear_tf=True,
    stop_words=None
)

problem_matrix = vectorizer.fit_transform(problems)

classifier = LogisticRegression(max_iter=2000)
classifier.fit(problem_matrix, categories)

# =====================================================
# SESSION STATE
# =====================================================
if "language" not in st.session_state:
    st.session_state.language = "de"

if "page" not in st.session_state:
    st.session_state.page = "home"

if "problem" not in st.session_state:
    st.session_state.problem = ""

if "category" not in st.session_state:
    st.session_state.category = ""

if "low_match" not in st.session_state:
    st.session_state.low_match = False

# =====================================================
# LANGUAGE SELECTOR
# =====================================================
lang_choice = st.selectbox(
    "Language / Sprache",
    options=["Deutsch", "English"],
    index=0 if st.session_state.language == "de" else 1,
    key="language_selector"
)

selected_language = "de" if lang_choice == "Deutsch" else "en"

if selected_language != st.session_state.language:
    st.session_state.language = selected_language
    st.rerun()

lang = st.session_state.language
t = TEXT[lang]

# =====================================================
# STYLING HEADER
# =====================================================
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">{t["eyebrow"]}</div>
    <h1>{t["title"]}</h1>
    <p><strong>{t["tagline"]}</strong></p>
    <p>{t["description"]}</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":
    st.markdown(f"""
    <div class="panel">
        <h3 style="color:#ffffff;">{t["question"]}</h3>
        <p style="color:#a7bad0;line-height:1.8;">
            {t["instructions"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("problem_form"):
        user_problem = st.text_area(
            t["problem_label"],
            placeholder=t["placeholder"],
            height=160
        )

        submitted = st.form_submit_button(
            t["button"],
            use_container_width=True
        )

    if submitted:
        cleaned_problem = user_problem.strip()

        if len(cleaned_problem) < 5:
            st.warning(t["short_error"])
        else:
            query = vectorizer.transform([cleaned_problem])

            similarities = cosine_similarity(
                query, problem_matrix
            ).flatten()

            best_index = int(similarities.argmax())
            best_score = float(similarities[best_index])
            predicted_category = classifier.predict(query)[0]

            if best_score >= 0.08:
                category = categories[best_index]
            else:
                category = predicted_category

            st.session_state.problem = cleaned_problem
            st.session_state.category = category
            st.session_state.low_match = best_score < 0.08
            st.session_state.page = "solution"
            st.rerun()

# =====================================================
# RESULTS PAGE
# =====================================================
elif st.session_state.page == "solution":
    st.markdown(f"## {t['results']}")
    st.caption(t["results_desc"])

    safe_problem = html.escape(st.session_state.problem)

    st.markdown(f"""
    <div class="panel">
        <div class="eyebrow">{t["reported"]}</div>
        <p style="color:#edf4fc;">{safe_problem}</p>
    </div>
    """, unsafe_allow_html=True)

    category = st.session_state.category
    translated_category = t["categories"][category]

    st.markdown(f"""
    <div class="result-card">
        <div class="eyebrow">{t["category"]}</div>
        <h3 style="color:#ffffff;">{translated_category}</h3>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.low_match:
        st.warning(t["low_match"])

    st.markdown(f"## {t['steps']}")

    selected_solutions = solutions[category][lang]

    for index, step in enumerate(selected_solutions, start=1):
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">STEP {index:02d}</div>
            <div style="margin-top:8px;color:#edf4fc;">
                {html.escape(step)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info(t["disclaimer"])

    if st.button(t["back"], use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# =====================================================
# FOOTER
# =====================================================
st.markdown(f"""
<div class="footer">
    <strong>Developed by Abdullah Aitah</strong><br>
    {t["footer"]}
</div>
""", unsafe_allow_html=True)
