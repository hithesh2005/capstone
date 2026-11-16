import io

import re

import uuid

from datetime import datetime







from urllib.parse import quote_plus















import streamlit as st







import streamlit.components.v1 as components







import speech_recognition as sr















import mediassist as core























st.set_page_config(







    page_title="MediAssist",







    page_icon="🩺",







    layout="centered",







    initial_sidebar_state="expanded",







)















LANGUAGES = {







    "English": ("en", "en-IN"),







    "తెలుగు": ("te", "te-IN"),







    "हिन्दी": ("hi", "hi-IN"),







}















UI_TRANSLATIONS = {







    "te": {







        "AI-powered preliminary health assistance": "AI ఆధారిత ప్రాథమిక ఆరోగ్య సహాయం",







        "Chat language": "చాట్ భాష", "Speak assistant replies": "అసిస్టెంట్ సమాధానాలను వినండి",







        "Voice replies use your browser's available speech voice.": "వాయిస్ సమాధానాలు మీ బ్రౌజర్‌లో అందుబాటులో ఉన్న వాయిస్‌ను ఉపయోగిస్తాయి.",







        "＋ New chat": "＋ కొత్త చాట్", "About this model": "ఈ మోడల్ గురించి",







        "773 disease classes": "773 వ్యాధి వర్గాలు", "Model scores are not clinical probabilities.": "మోడల్ స్కోర్లు వైద్య నిర్ధారణ సంభావ్యతలు కావు.",







        "For emergencies, contact local emergency services or go to the nearest emergency department.": "అత్యవసర పరిస్థితుల్లో స్థానిక అత్యవసర సేవలను సంప్రదించండి లేదా సమీప అత్యవసర విభాగానికి వెళ్లండి.",







        "A conversational, multilingual health-assessment assistant": "బహుభాషా సంభాషణాత్మక ఆరోగ్య అంచనా సహాయకుడు",







        "MediAssist provides preliminary educational information only. It is not a doctor, does not provide a confirmed diagnosis, and does not prescribe treatment. Do not delay urgent care to use this chatbot.": "MediAssist ప్రాథమిక విద్యాపరమైన సమాచారాన్ని మాత్రమే అందిస్తుంది. ఇది వైద్యుడు కాదు; నిర్ధారిత రోగ నిర్ధారణ లేదా చికిత్సను అందించదు. అత్యవసర వైద్య సహాయాన్ని ఆలస్యం చేయవద్దు.",







        "Message MediAssist...": "MediAssist కు సందేశం పంపండి...", "Send": "పంపండి",







        "Sorry, I couldn't understand. Please try speaking again.": "క్షమించండి, మీ మాటలు అర్థం కాలేదు. దయచేసి మళ్లీ మాట్లాడండి.",







        "Speech recognition is unavailable. Check your internet connection.": "వాయిస్ గుర్తింపు అందుబాటులో లేదు. మీ ఇంటర్నెట్ కనెక్షన్‌ను తనిఖీ చేయండి.",







        "Your next step: connect with a healthcare professional": "తదుపరి చర్య: ఆరోగ్య నిపుణుడిని సంప్రదించండి",







        "Find care near you": "మీకు సమీపంలో వైద్య సేవలను కనుగొనండి",







        "Use your current device location or enter an area manually. Your browser will ask for location permission.": "మీ పరికరం ప్రస్తుత స్థానాన్ని ఉపయోగించండి లేదా ప్రాంతాన్ని నమోదు చేయండి. మీ బ్రౌజర్ స్థాన అనుమతిని అడుగుతుంది.",







        "Enter your city, area, or PIN code": "మీ నగరం, ప్రాంతం లేదా పిన్ కోడ్ నమోదు చేయండి",







        "e.g. Hanamkonda, Warangal or 506001": "ఉదా: హనుమకొండ, వరంగల్ లేదా 506001",







        "Your location is used only to create map-search links. It is not saved by this feature.": "మీ స్థానాన్ని మ్యాప్ శోధన లింక్‌ల కోసం మాత్రమే ఉపయోగిస్తాము. ఈ ఫీచర్ దాన్ని సేవ్ చేయదు.",







        "Explore locations on the map": "మ్యాప్‌లో ప్రదేశాలను చూడండి", "🏥 Hospitals": "🏥 ఆసుపత్రులు", "👨‍⚕️ Doctors": "👨‍⚕️ వైద్యులు", "➕ Clinics": "➕ క్లినిక్‌లు",







        "Open hospitals in Google Maps": "Google Maps లో ఆసుపత్రులను తెరవండి", "Open doctors in Google Maps": "Google Maps లో వైద్యులను తెరవండి", "Open clinics in Google Maps": "Google Maps లో క్లినిక్‌లను తెరవండి",







        "Need urgent help?": "తక్షణ సహాయం అవసరమా?",







        "MediAssist does not directly contact doctors, book appointments, or transmit your health information to hospitals.": "MediAssist నేరుగా వైద్యులను సంప్రదించదు, అపాయింట్‌మెంట్ బుక్ చేయదు లేదా మీ ఆరోగ్య సమాచారాన్ని ఆసుపత్రులకు పంపదు.",







    },







    "hi": {







        "AI-powered preliminary health assistance": "AI आधारित प्रारंभिक स्वास्थ्य सहायता",







        "Chat language": "चैट की भाषा", "Speak assistant replies": "सहायक के जवाब बोलकर सुनें",







        "Voice replies use your browser's available speech voice.": "आवाज़ वाले जवाब आपके ब्राउज़र में उपलब्ध आवाज़ का उपयोग करते हैं।",







        "＋ New chat": "＋ नई चैट", "About this model": "इस मॉडल के बारे में",







        "773 disease classes": "773 रोग श्रेणियाँ", "Model scores are not clinical probabilities.": "मॉडल स्कोर चिकित्सकीय संभावनाएँ नहीं हैं।",







        "For emergencies, contact local emergency services or go to the nearest emergency department.": "आपात स्थिति में स्थानीय आपातकालीन सेवाओं से संपर्क करें या नज़दीकी आपातकालीन विभाग जाएँ।",







        "A conversational, multilingual health-assessment assistant": "बहुभाषी संवादात्मक स्वास्थ्य मूल्यांकन सहायक",







        "MediAssist provides preliminary educational information only. It is not a doctor, does not provide a confirmed diagnosis, and does not prescribe treatment. Do not delay urgent care to use this chatbot.": "MediAssist केवल प्रारंभिक शैक्षिक जानकारी देता है। यह डॉक्टर नहीं है, निश्चित निदान या उपचार नहीं देता। आपातकालीन देखभाल में देरी न करें।",







        "Message MediAssist...": "MediAssist को संदेश भेजें...", "Send": "भेजें",







        "Sorry, I couldn't understand. Please try speaking again.": "क्षमा करें, मैं समझ नहीं पाया। कृपया फिर से बोलें।",







        "Speech recognition is unavailable. Check your internet connection.": "वाक् पहचान उपलब्ध नहीं है। अपना इंटरनेट कनेक्शन जाँचें।",







        "Your next step: connect with a healthcare professional": "अगला कदम: स्वास्थ्य विशेषज्ञ से संपर्क करें",







        "Find care near you": "अपने पास चिकित्सा सहायता खोजें",







        "Use your current device location or enter an area manually. Your browser will ask for location permission.": "अपने डिवाइस की वर्तमान लोकेशन का उपयोग करें या क्षेत्र दर्ज करें। आपका ब्राउज़र लोकेशन की अनुमति माँगेगा।",







        "Enter your city, area, or PIN code": "अपना शहर, क्षेत्र या पिन कोड दर्ज करें",







        "e.g. Hanamkonda, Warangal or 506001": "उदाहरण: हनमकोंडा, वारंगल या 506001",







        "Your location is used only to create map-search links. It is not saved by this feature.": "आपकी लोकेशन केवल मैप खोज लिंक बनाने के लिए उपयोग होती है। इसे इस सुविधा द्वारा सहेजा नहीं जाता।",







        "Explore locations on the map": "मैप पर स्थान देखें", "🏥 Hospitals": "🏥 अस्पताल", "👨‍⚕️ Doctors": "👨‍⚕️ डॉक्टर", "➕ Clinics": "➕ क्लिनिक",







        "Open hospitals in Google Maps": "Google Maps में अस्पताल खोलें", "Open doctors in Google Maps": "Google Maps में डॉक्टर खोलें", "Open clinics in Google Maps": "Google Maps में क्लिनिक खोलें",







        "Need urgent help?": "क्या आपको तुरंत मदद चाहिए?",







        "MediAssist does not directly contact doctors, book appointments, or transmit your health information to hospitals.": "MediAssist सीधे डॉक्टरों से संपर्क नहीं करता, अपॉइंटमेंट बुक नहीं करता और आपकी स्वास्थ्य जानकारी अस्पतालों को नहीं भेजता।",







    }







}























def tr(text):
    """Translate visible UI text while preserving Markdown and URLs."""
    language = st.session_state.get("language", "en")
    value = str(text)
    if language == "en" or not value.strip():
        return value

    exact = UI_TRANSLATIONS.get(language, {}).get(value)
    if exact:
        return exact

    # Protect URLs and Markdown markers without using a fragile regex.
    token_pattern = r"(https?://[^\s)]+|\*\*|\*|`|^#{1,6}\s)"
    try:
        parts = re.split(token_pattern, value)
        output = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(token_pattern, part):
                output.append(part)
            else:
                output.append(core.translate_text(part, language))
        return "".join(output)
    except Exception:
        return value


def model_language_text(text):







    """Convert Telugu/Hindi symptom text to English for the existing English model."""







    language = st.session_state.get("language", "en")







    if language == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(str(text))
        return translated.strip() if translated else str(text)
    except Exception as exc:
        st.warning(f"Could not translate the symptom into English: {exc}")
        return str(text)























st.markdown("""







<style>







    :root {







        --ma-ink: #f1f5f9; --ma-muted: #b6c2d1; --ma-teal: #2dd4bf;







        --ma-teal-dark: #0f766e; --ma-bg: #0b1220; --ma-card: #172235;







        --ma-border: #334155; --ma-input: #111c2e;







    }







    html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMain"] {







        background: var(--ma-bg) !important; color: var(--ma-ink) !important;







    }







    [data-testid="stHeader"] { background: #0b1220 !important; }







    [data-testid="stSidebar"] { background: #111c2e !important; border-right: 1px solid var(--ma-border); }







    [data-testid="stSidebar"] * { color: var(--ma-ink) !important; }







    .block-container { max-width: 900px; padding-top: 1.5rem; padding-bottom: 6rem; }







    .brand { color: var(--ma-teal) !important; font-size: 2rem; font-weight: 800; letter-spacing: -1px; }







    .subbrand { color: var(--ma-muted) !important; margin-top: -10px; }







    .disclaimer { background: #332b16 !important; border: 1px solid #8b6b27 !important; color: #ffe8a3 !important; padding: 12px 14px; border-radius: 12px; font-size: .92rem; }







    .disclaimer * { color: #ffe8a3 !important; }







    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,







    [data-testid="stMarkdownContainer"] li, [data-testid="stText"], .stCaption,







    label, [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] p { color: var(--ma-ink) !important; }







    [data-testid="stCaptionContainer"] p { color: var(--ma-muted) !important; }







    [data-testid="stChatMessage"] { background: var(--ma-card) !important; border: 1px solid var(--ma-border) !important; border-radius: 16px !important; color: var(--ma-ink) !important; padding: .8rem 1rem !important; }







    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color: var(--ma-ink) !important; }







    [data-testid="stChatInput"], [data-testid="stTextInput"] input,







    [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input {







        background: var(--ma-input) !important; border: 1px solid #52647c !important;







        border-radius: 12px !important; color: #ffffff !important;







        -webkit-text-fill-color: #ffffff !important; caret-color: var(--ma-teal) !important;







    }







    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea:focus,







    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {







        color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;







        background: var(--ma-input) !important;







    }







    input::placeholder, textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder {







        color: #aab8ca !important; opacity: 1 !important; -webkit-text-fill-color: #aab8ca !important;







    }







    [data-baseweb="select"] > div, [data-baseweb="popover"] [role="option"] { background: #172235 !important; color: #f1f5f9 !important; }







    [data-testid="stChatInput"] button, .stButton > button, [data-testid="stLinkButton"] a {







        background: #0f766e !important; color: #ffffff !important; border: 1px solid #2dd4bf !important; border-radius: 10px !important;







    }







    .stButton > button:hover, [data-testid="stLinkButton"] a:hover { background: #115e59 !important; color: #ffffff !important; }







    [data-testid="stAlert"] { background: #172235 !important; color: #f1f5f9 !important; }







    hr { border-color: var(--ma-border) !important; }







    @media (max-width: 640px) { .block-container { padding: 1rem .8rem 6rem .8rem; } .brand { font-size: 1.65rem; } }







</style>







""", unsafe_allow_html=True)























def speak_in_browser(text: str, language: str):







    """Speak translated text only when the browser has a matching language voice."""







    import json







    import re















    spoken_text = re.sub(r"[#*_`]", "", str(text))







    spoken_text = re.sub(r"\s+", " ", spoken_text).strip()[:1800]

    if not spoken_text:







        return















    locale = {"en": "en-IN", "te": "te-IN", "hi": "hi-IN"}.get(language, "en-IN")







    labels = {







        "en": ("▶ Play reply", "Speaking…", "No English voice is available in this browser."),







        "te": ("▶ సమాధానం వినండి", "మాట్లాడుతోంది…", "ఈ పరికరంలో తెలుగు వాయిస్ అందుబాటులో లేదు. Windows సెట్టింగ్స్‌లో Telugu speech voice జోడించండి."),







        "hi": ("▶ जवाब सुनें", "बोल रहा है…", "इस डिवाइस में हिंदी आवाज़ उपलब्ध नहीं है। Windows सेटिंग्स में Hindi speech voice जोड़ें।"),







    }







    play_label, speaking_label, missing_label = labels.get(language, labels["en"])







    text_json = json.dumps(spoken_text, ensure_ascii=False)







    locale_json = json.dumps(locale)







    play_json = json.dumps(play_label, ensure_ascii=False)







    speaking_json = json.dumps(speaking_label, ensure_ascii=False)







    missing_json = json.dumps(missing_label, ensure_ascii=False)















    components.html(







        f"""<!doctype html><html><head><meta charset="utf-8">







        <style>body{{margin:0;font:13px Arial;color:#cbd5e1}}button{{border:1px solid #2dd4bf;







        border-radius:8px;background:#0f766e;padding:6px 10px;color:#fff;cursor:pointer}}</style>







        </head><body><button id="play">{play_json}</button><span id="status" style="margin-left:8px"></span>







        <script>







        (() => {{







          const text={text_json}, locale={locale_json};







          const speakingLabel={speaking_json}, missingLabel={missing_json};







          const status=document.getElementById('status');







          function speak() {{







            if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {{







              status.textContent=missingLabel; return;







            }}







            const synth=window.speechSynthesis;







            const voices=synth.getVoices();







            const prefix=locale.split('-')[0].toLowerCase();







            const selected=voices.find(v=>v.lang.toLowerCase()===locale.toLowerCase())







              || voices.find(v=>v.lang.toLowerCase().startsWith(prefix+'-'))







              || voices.find(v=>v.lang.toLowerCase()===prefix);







            if (!selected) {{ status.textContent=missingLabel; return; }}







            synth.cancel();







            const utterance=new SpeechSynthesisUtterance(text);







            utterance.lang=locale;







            utterance.voice=selected;







            utterance.rate=0.92;







            utterance.onstart=()=>status.textContent=speakingLabel;







            utterance.onend=()=>status.textContent='';







            utterance.onerror=()=>status.textContent=missingLabel;







            synth.speak(utterance);







          }}







          document.getElementById('play').addEventListener('click', speak);







          setTimeout(() => {{







            if (window.speechSynthesis && window.speechSynthesis.getVoices().length) speak();







            else if (window.speechSynthesis) window.speechSynthesis.addEventListener('voiceschanged', speak, {{once:true}});







          }}, 300);







        }})();







        </script></body></html>""",







        height=42,







    )























def transcribe_audio(audio_file, language_code):







    recognizer = sr.Recognizer()







    audio_bytes = audio_file.getvalue()







    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:







        audio = recognizer.record(source)







    return recognizer.recognize_google(audio, language=language_code)























@st.cache_resource(show_spinner="Loading MediAssist's saved AI model...")







def load_model():







    if core.os.path.exists(core.MODEL_FILE):







        return core.load_saved_model()







    # Only trains on the first run when no compatible saved model exists.







    return core.train_and_save()























def add_message(role, content):







    st.session_state.messages.append({







        "id": str(uuid.uuid4()),







        "role": role,







        "content": content,







    })























def render_assistant(text):







    language = st.session_state.get("language", "en")







    localized = tr(text)







    # Store English source text; the selected language is applied to display and audio.







    add_message("assistant", text)







    if language != "en" and localized == text:







        st.warning("Translation did not complete. Check internet access and install deep-translator.")







    if st.session_state.voice_reply:







        speak_in_browser(localized, language)























def yes_no_value(text):







    """Accept natural short yes/no replies, including common speech transcripts."""







    import re







    value = re.sub(r"[^\w\s]", " ", str(text).strip().lower())







    tokens = value.split()







    if not tokens:







        return None















    yes_words = {"yes", "y", "yeah", "yep", "yup", "sure", "correct", "true", "absolutely",







                 "అవును", "అవున", "हाँ", "हां", "हा"}







    no_words = {"no", "n", "nope", "nah", "false", "లేదు", "కాదు", "नहीं", "नही", "नहि"}















    # Exact answers or a short phrase beginning with a clear yes/no token.







    if tokens[0] in yes_words:







        return True







    if tokens[0] in no_words:







        return False







    if len(tokens) <= 5:







        if any(token in yes_words for token in tokens) and not any(token in no_words for token in tokens):







            return True







        if any(token in no_words for token in tokens) and not any(token in yes_words for token in tokens):







            return False







    return None























def ask_next_question():







    a = st.session_state.assessment







    if a["question_count"] >= 6:







        a["stage"] = "duration"







        render_assistant("Thank you. How long have you had these symptoms? "







                         "For example: 2 days, 1 week, or since yesterday.")







        return















    results = core.get_predictions(a["model"], a["vector"], top_n=5)







    selected = core.choose_group_question(







        a["initial"],







        a["vector"],







        results,







        a["profiles"],







        a["symptom_cols"],







        a["asked_symptoms"],







        a["asked_groups"],







    )















    if selected is None:







        a["stage"] = "duration"







        render_assistant("I have asked the relevant follow-up questions available "







                         "for this assessment. How long have you had these symptoms?")







        return















    group_name, symptom, question = selected







    a["asked_symptoms"].add(symptom)







    a["asked_groups"].add(group_name)







    a["pending_symptom"] = symptom







    a["pending_group"] = group_name







    a["pending_question"] = question







    a["question_count"] += 1















    translated = core.translate_question(question)







    render_assistant(







        f"**Follow-up {a['question_count']} of 6**\n\n"







        f"{translated}\n\n{core.t('answer_yes_no')}"







    )























def start_assessment(user_text):



    model_package = st.session_state.model_package







    symptom_cols = model_package["symptom_columns"]







    english_symptoms = model_language_text(user_text)







    vector, matched = core.text_to_vector(english_symptoms, symptom_cols)







    flags = core.check_red_flags(english_symptoms)















    st.session_state.assessment = {







        "model": model_package["model"],







        "profiles": model_package["profiles"],







        "symptom_cols": symptom_cols,







        "initial": english_symptoms,







        "vector": vector,







        "matched": matched,







        "initial_flags": flags,







        "asked_symptoms": set(),







        "asked_groups": set(),







        "answers": [],







        "question_count": 0,







        "stage": "followup",







        "duration": "",







        "severity": "",







    }















    results = core.get_predictions(model_package["model"], vector, top_n=5)







    response = "I understood your symptoms. Here is the initial model assessment:\n\n"







    if matched:







        response += "**Symptoms recognized:** " + ", ".join(matched[:15]) + "\n\n"







    else:







        response += "**Symptoms recognized:** No exact dataset symptom matched. "







        response += "The model may not fully understand this description.\n\n"















    response += "**Initial model classifications**\n"







    for rank, (disease, probability) in enumerate(results, start=1):







        response += f"{rank}. {disease} — {probability * 100:.1f}%\n"















    response += "\nThese are model outputs, not confirmed diagnoses or clinical certainty."















    if flags:







        response += (







            "\n\n🚨 **Urgent warning:** I detected a possible red-flag symptom: "







            + ", ".join(flags)







            + ". Please seek urgent medical attention now. Do not delay care "







              "to complete this chat."







        )















    render_assistant(response)







    if flags:







        # Continue only as an optional record, without suggesting that chat is care.







        render_assistant("If you are safe and still wish to continue this educational "







                         "assessment, answer the follow-up question below.")







    ask_next_question()























def finish_assessment():







    a = st.session_state.assessment







    results = core.get_predictions(a["model"], a["vector"], top_n=5)







    top_disease, top_probability = results[0]







    second_probability = results[1][1] if len(results) > 1 else 0.0







    margin = top_probability - second_probability















    yes_symptoms = [







        item["symptom"] for item in a["answers"] if item["answer"] == "yes"







    ]







    final_context = a["initial"] + " " + " ".join(yes_symptoms)







    flags = core.check_red_flags(final_context)















    if top_probability >= 0.60 and margin >= 0.15:







        lead = (







            f"The leading model classification is **{top_disease}** "







            f"({top_probability * 100:.1f}% model score)."







        )







    else:







        lead = (







            "The model did not produce a sufficiently separated result. "







            "Several conditions overlap, so no single result should be treated as confirmed."







        )















    response = "**Assessment summary**\n\n"







    response += lead + "\n\n**Top model classifications**\n"







    for rank, (disease, probability) in enumerate(results, start=1):







        response += f"{rank}. {disease} — {probability * 100:.1f}%\n"















    response += (







        f"\n**Duration:** {a['duration'] or 'Not provided'}"







        f"\n\n**Severity:** {a['severity']}"







        f"\n\n**Symptoms recognized:** "







        f"{', '.join(a['matched']) if a['matched'] else 'No exact match'}"







    )















    if flags:







        response += (







            "\n\n🚨 **Urgent warning:** " + ", ".join(flags)







            + ". Please seek urgent medical attention."







        )







    elif a["severity"] == "severe":







        response += "\n\nPlease seek prompt professional medical evaluation."







    elif top_probability < 0.30 or margin < 0.10:







        response += (







            "\n\nConsider discussing these symptoms with a qualified healthcare professional."







        )







    else:







        response += (







            "\n\nYou can share this summary with a healthcare professional for evaluation."







        )















    response += (







        "\n\n*This is a preliminary educational symptom assessment, not a diagnosis "







        "or a prescription. Do not start, stop, or change medication based on this result.*"







    )















    session = {







        "timestamp": datetime.now().isoformat(timespec="seconds"),







        "initial_symptoms": a["initial"],







        "recognized_symptoms": a["matched"],







        "duration": a["duration"],







        "severity": a["severity"],







        "follow_up_answers": a["answers"],







        "top_classification": top_disease,







        "top_probability": top_probability,







        "top_5": [







            {"disease": disease, "probability": probability}







            for disease, probability in results







        ],







        "red_flags": flags,







        "language": st.session_state.language,







        "input_mode": "chat",







    }







    core.save_session(session)







    a["stage"] = "completed"







    render_assistant(response)























def handle_user_message(user_text):







    user_text = user_text.strip()







    if not user_text:







        return















    add_message("user", user_text)







    a = st.session_state.assessment















    if a is None or a["stage"] == "completed":







        lowered = user_text.lower()







        if lowered in {"hi", "hello", "hey", "నమస్కారం", "नमस्ते"}:







            render_assistant(







                "Hello! I’m MediAssist. Tell me what symptoms you are experiencing. "







                "You can type your message below."







            )







            return







        if any(word in lowered for word in ["help", "what can you do", "మీరు ఏమి"]):







            render_assistant(







                "I can collect symptom information, ask adaptive follow-up questions, "







                "and show preliminary model classifications. I cannot confirm a diagnosis, "







                "prescribe medicine, or replace a healthcare professional. "







                "Please describe your symptoms to begin."







            )







            return







        start_assessment(user_text)







        return















    if a["stage"] == "followup":







        answer = yes_no_value(user_text)







        if answer is None:







            render_assistant("Please answer **yes** or **no** so I can update the assessment.")







            return















        symptom = a["pending_symptom"]







        a["answers"].append({







            "group": a["pending_group"],







            "symptom": symptom,







            "question": a["pending_question"],







            "answer": "yes" if answer else "no",







        })







        index = a["symptom_cols"].index(symptom)







        a["vector"][index] = 1 if answer else 0







        ask_next_question()







        return















    if a["stage"] == "duration":







        a["duration"] = user_text







        a["stage"] = "severity"







        render_assistant(







            "How severe are your symptoms? Please reply **mild**, **moderate**, or **severe**."







        )







        return















    if a["stage"] == "severity":







        severity_raw = user_text.lower().strip()







        severity_map = {







            "తేలికపాటి": "mild", "మధ్యస్థ": "moderate", "తీవ్ర": "severe",







            "हल्का": "mild", "मध्यम": "moderate", "गंभीर": "severe",







        }







        severity = severity_map.get(severity_raw, severity_raw)







        if severity not in {"mild", "moderate", "severe"}:







            render_assistant("Please choose **mild**, **moderate**, or **severe**.")







            return







        a["severity"] = severity







        finish_assessment()







        return























def reset_chat():







    st.session_state.messages = []







    st.session_state.assessment = None







    add_message(







        "assistant",







        "Hello! I’m **MediAssist**, your multilingual preliminary health-assessment assistant. "







        "Tell me what symptoms you are experiencing. You can type your message below."







    )























# Sidebar controls







with st.sidebar:







    st.markdown(tr("## 🩺 MediAssist"))







    st.caption(tr("AI-powered preliminary health assistance"))







    language_name = st.selectbox(tr("Chat language"), list(LANGUAGES.keys()), index=0)







    language, speech_locale = LANGUAGES[language_name]







    st.session_state.language = language







    st.session_state.speech_locale = speech_locale







    core.LANGUAGE = language















    st.session_state.voice_reply = st.toggle(tr("Speak assistant replies"), value=True)







    st.caption(tr("Voice replies use your browser's available speech voice."))







    if st.button(tr("＋ New chat"), use_container_width=True):







        reset_chat()







        st.rerun()















    st.divider()







    st.markdown(tr("**About this model**"))







    st.write(tr("773 disease classes"))







    metrics = st.session_state.get("model_package", {}).get("metrics", {})







    if metrics:







        st.write(tr(f"Test accuracy: {metrics.get('accuracy', 0) * 100:.2f}%"))







    st.caption(tr("Model scores are not clinical probabilities."))







    st.divider()







    st.caption(tr("For emergencies, contact local emergency services or go to the nearest emergency department."))























st.markdown('<div class="brand">🩺 MediAssist</div>', unsafe_allow_html=True)







st.markdown(







    tr('<div class="subbrand">A conversational, multilingual health-assessment assistant</div>'),







    unsafe_allow_html=True,







)







st.markdown(







    tr('<div class="disclaimer">MediAssist provides preliminary educational information only. '







    'It is not a doctor, does not provide a confirmed diagnosis, and does not prescribe treatment. '







    'Do not delay urgent care to use this chatbot.</div>'),







    unsafe_allow_html=True,







)







st.write("")















if "messages" not in st.session_state:







    st.session_state.messages = []







if "assessment" not in st.session_state:







    st.session_state.assessment = None







if "voice_reply" not in st.session_state:







    st.session_state.voice_reply = True







if "language" not in st.session_state:







    st.session_state.language = "en"







if "speech_locale" not in st.session_state:







    st.session_state.speech_locale = "en-IN"















if "model_package" not in st.session_state:







    try:







        st.session_state.model_package = load_model()







    except Exception as exc:







        st.error(tr(f"Could not load MediAssist model: {exc}"))







        st.info(tr("Check that your saved_model folder and dataset are in the project directory."))







        st.stop()















if not st.session_state.messages:







    reset_chat()















for message in st.session_state.messages:







    with st.chat_message(message["role"]):







        visible_content = tr(message["content"]) if message["role"] == "assistant" else message["content"]







        st.markdown(visible_content)















# Original message input with a microphone button beside it.







# The existing theme and all other page sections remain unchanged.







message_col, mic_col = st.columns([0.88, 0.12], vertical_alignment="bottom")















with message_col:







    with st.form("mediassist_message_form", clear_on_submit=True):







        typed_text = st.text_input(







            tr("Message MediAssist..."),







            key="mediassist_typed_message",







            label_visibility="collapsed",







            placeholder=tr("Message MediAssist..."),







        )







        send_message = st.form_submit_button(tr("Send"), use_container_width=True)















with mic_col:







    audio_file = st.audio_input(







        "🎙️",







        key="mediassist_mic_input",







        sample_rate=16000,







        label_visibility="collapsed",







    )















if send_message and typed_text.strip():







    handle_user_message(typed_text.strip())







    st.rerun()















if audio_file is not None:







    audio_bytes = audio_file.getvalue()







    audio_fingerprint = __import__("hashlib").sha256(audio_bytes).hexdigest()















    if audio_fingerprint != st.session_state.get("last_processed_audio"):







        try:







            spoken_text = transcribe_audio(







                audio_file,







                st.session_state.get("speech_locale", "en-IN"),







            )







            st.session_state.last_processed_audio = audio_fingerprint







            if spoken_text and spoken_text.strip():







                handle_user_message(spoken_text.strip())







                st.rerun()







        except sr.UnknownValueError:







            st.warning(tr("Sorry, I couldn't understand. Please try speaking again."))







        except sr.RequestError:







            st.error(tr("Speech recognition is unavailable. Check your internet connection."))







        except Exception as exc:







            st.error(tr(f"Microphone error: {exc}"))















# Care navigation appears only after the symptom conversation is completed.







# It does not claim that the model has diagnosed a disease or booked an appointment.







assessment = st.session_state.get("assessment")







if assessment and assessment.get("stage") == "completed":







    st.markdown("---")







    st.markdown(tr("### 🩺 Your next step: connect with a healthcare professional"))







    st.write(tr(







        "MediAssist cannot confirm a disease. If you are concerned about your symptoms, "







        "you can use the options below to find a qualified doctor or nearby hospital."







    ))







    with st.container(border=True):







        st.markdown(tr("**Find care near you**"))







        st.markdown(tr("Use your current device location or enter an area manually. Your browser will ask for location permission."))







        live_location_html = """<!doctype html><html><body style="margin:0;background:#172235;font-family:Arial,sans-serif">







        <button id="liveLoc" style="background:#0f766e;color:#fff;border:1px solid #2dd4bf;border-radius:10px;padding:12px 16px;font-size:15px;font-weight:600;cursor:pointer;width:100%">📍 Use my live location</button>







        <div id="locStatus" style="color:#cbd5e1;font-size:13px;margin-top:8px"></div>







        <script>







        document.getElementById('liveLoc').addEventListener('click', function(){







          const status=document.getElementById('locStatus');







          if(!navigator.geolocation){status.textContent='Location is not supported by this browser.';return;}







          status.textContent='Requesting location permission…';







          navigator.geolocation.getCurrentPosition(function(pos){







            const lat=pos.coords.latitude, lon=pos.coords.longitude;







            status.textContent='Location found. Opening nearby hospitals in Google Maps…';







            window.open('https://www.google.com/maps/search/hospitals/@'+lat+','+lon+',14z','_blank');







          }, function(err){







            status.textContent=err.code===1?'Location permission was denied. Allow location access or enter your area below.':'Could not get your location. Check GPS and browser permissions, or enter your area below.';







          }, {enableHighAccuracy:true,timeout:15000,maximumAge:60000});







        });







        </script></body></html>"""







        location_labels = {







            "te": {







                "📍 Use my live location": "📍 నా ప్రస్తుత స్థానాన్ని ఉపయోగించండి",







                "Location is not supported by this browser.": "ఈ బ్రౌజర్‌లో స్థాన సేవ అందుబాటులో లేదు.",







                "Requesting location permission…": "స్థాన అనుమతి కోరుతోంది…",







                "Location found. Opening nearby hospitals in Google Maps…": "స్థానం కనుగొనబడింది. Google Maps లో సమీప ఆసుపత్రులను తెరుస్తోంది…",







                "Location permission was denied. Allow location access or enter your area below.": "స్థాన అనుమతి నిరాకరించబడింది. స్థాన అనుమతిని ఇవ్వండి లేదా మీ ప్రాంతాన్ని క్రింద నమోదు చేయండి.",







                "Could not get your location. Check GPS and browser permissions, or enter your area below.": "మీ స్థానాన్ని పొందలేకపోయాం. GPS మరియు బ్రౌజర్ అనుమతులను తనిఖీ చేయండి లేదా మీ ప్రాంతాన్ని క్రింద నమోదు చేయండి.",







            },







            "hi": {







                "📍 Use my live location": "📍 मेरी वर्तमान लोकेशन उपयोग करें",







                "Location is not supported by this browser.": "यह ब्राउज़र लोकेशन का समर्थन नहीं करता।",







                "Requesting location permission…": "लोकेशन की अनुमति माँगी जा रही है…",







                "Location found. Opening nearby hospitals in Google Maps…": "लोकेशन मिल गई। Google Maps में नज़दीकी अस्पताल खोले जा रहे हैं…",







                "Location permission was denied. Allow location access or enter your area below.": "लोकेशन की अनुमति नहीं मिली। अनुमति दें या नीचे अपना क्षेत्र दर्ज करें।",







                "Could not get your location. Check GPS and browser permissions, or enter your area below.": "लोकेशन नहीं मिल सकी। GPS और ब्राउज़र अनुमति जाँचें या नीचे अपना क्षेत्र दर्ज करें।",







            },







        }







        for source_label, translated_label in location_labels.get(st.session_state.language, {}).items():







            live_location_html = live_location_html.replace(source_label, translated_label)







        components.html(live_location_html, height=75)







        care_location = st.text_input(







            tr("Enter your city, area, or PIN code"),







            key="care_location",







            placeholder=tr("e.g. Hanamkonda, Warangal or 506001"),







            help=tr("Your location is used only to create map-search links. It is not saved by this feature."),







        )







        if care_location.strip():







            place = care_location.strip()







            hospital_query = quote_plus(f"hospitals near {place}")







            doctor_query = quote_plus(f"doctors near {place}")







            clinic_query = quote_plus(f"clinics near {place}")







            hospitals_url = f"https://www.google.com/maps/search/?api=1&query={hospital_query}"







            doctors_url = f"https://www.google.com/maps/search/?api=1&query={doctor_query}"







            clinics_url = f"https://www.google.com/maps/search/?api=1&query={clinic_query}"















            st.markdown(tr("**Explore locations on the map**"))







            map_tabs = st.tabs([tr("🏥 Hospitals"), tr("👨‍⚕️ Doctors"), tr("➕ Clinics")])







            for tab, query in zip(map_tabs, [hospital_query, doctor_query, clinic_query]):







                with tab:







                    embed_url = f"https://www.google.com/maps?q={query}&output=embed"







                    map_html = (







                        '<iframe title="Google Maps results" src="' + embed_url +







                        '" width="100%" height="430" style="border:0;border-radius:12px" '







                        'loading="lazy" referrerpolicy="no-referrer-when-downgrade" allowfullscreen></iframe>'







                    )







                    components.html(map_html, height=440)















            c1, c2, c3 = st.columns(3)







            with c1:







                st.link_button(tr("Open hospitals in Google Maps"), hospitals_url, use_container_width=True)







            with c2:







                st.link_button(tr("Open doctors in Google Maps"), doctors_url, use_container_width=True)







            with c3:







                st.link_button(tr("Open clinics in Google Maps"), clinics_url, use_container_width=True)







            st.caption(tr("Google Maps supplies the live map and listings. Confirm opening hours, services, doctor availability, and contact details directly. Results depend on the location you entered."))







        else:







            st.caption(tr("Enter your location to see nearby hospitals, doctors, and clinics on a map."))















    st.markdown(tr("**Need urgent help?**"))







    st.error(







        tr("For severe symptoms, chest pain, difficulty breathing, fainting, or heavy bleeding, "







        "seek emergency care immediately. In India, call **112** for emergency assistance. "







        "Do not wait for MediAssist or a map search.")







    )







    st.caption(tr("MediAssist does not directly contact doctors, book appointments, or transmit your health information to hospitals."))
