# ============================================================
# MEDIASSIST
# Rural AI Symptom Assessment & Health Assistance System
# ============================================================
# Features:
# - Uses an existing saved 773-disease model when available
# - Trains and saves the model only when no saved model exists
# - Group-based adaptive follow-up questions
# - One question per relevant symptom group
# - Avoids duplicate symptoms and duplicate groups
# - Prioritizes the user's initial complaint
# - Top-5 model classifications
# - Low-confidence handling
# - Red-flag / emergency symptom detection
# - Duration and severity collection
# - Symptom summary
# - Session history saved as JSON
# - Doctor/appointment request demo saved as JSON
# - English / Telugu / Hindi UI labels and common questions
# - Main menu for repeated assessments
#
# NOTE:
# MediAssist is an educational/research symptom-classification system.
# It is not a medical diagnosis and does not prescribe treatment.
# ============================================================

import os
import re
import json
import pickle
import warnings
from functools import lru_cache
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "archive (8)",
    "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
)

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_FILE = os.path.join(MODEL_DIR, "medisathi_773_model.pkl")

HISTORY_DIR = os.path.join(BASE_DIR, "mediassist_history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "sessions.json")
APPOINTMENT_FILE = os.path.join(HISTORY_DIR, "appointment_requests.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ============================================================
# 2. UI / LANGUAGE
# ============================================================

LANGUAGE = "en"

TEXT = {
    "en": {
        "app": "MEDIASSIST",
        "subtitle": "Rural AI Symptom Assessment & Health Assistance System",
        "symptoms": "Enter symptoms",
        "example": "Example: fever, cough and body pain",
        "yes_no": "(yes/no)",
        "answer_yes_no": "Please answer yes or no.",
        "invalid_yes_no": "Please enter yes or no.",
        "duration": "How long have you had these symptoms?",
        "severity": "How severe are your symptoms?",
        "severity_options": "(mild/moderate/severe)",
        "invalid_severity": "Please enter mild, moderate, or severe.",
        "goodbye": "Thank you for using MediAssist.",
    },
    "te": {
        "app": "MEDIASSIST",
        "subtitle": "AI లక్షణాల అంచనా మరియు ఆరోగ్య సహాయ వ్యవస్థ",
        "symptoms": "మీ లక్షణాలను నమోదు చేయండి",
        "example": "ఉదాహరణ: జ్వరం, దగ్గు మరియు శరీర నొప్పి",
        "yes_no": "(అవును/కాదు)",
        "answer_yes_no": "దయచేసి అవును లేదా కాదు అని సమాధానం ఇవ్వండి.",
        "invalid_yes_no": "దయచేసి అవును లేదా కాదు అని నమోదు చేయండి.",
        "duration": "ఈ లక్షణాలు ఎంత కాలంగా ఉన్నాయి?",
        "severity": "లక్షణాల తీవ్రత ఎంత?",
        "severity_options": "(తేలికపాటి/మధ్యస్థ/తీవ్ర)",
        "invalid_severity": "దయచేసి mild, moderate లేదా severe నమోదు చేయండి.",
        "goodbye": "MediAssist ఉపయోగించినందుకు ధన్యవాదాలు.",
    },
    "hi": {
        "app": "MEDIASSIST",
        "subtitle": "AI लक्षण मूल्यांकन और स्वास्थ्य सहायता प्रणाली",
        "symptoms": "अपने लक्षण दर्ज करें",
        "example": "उदाहरण: बुखार, खांसी और शरीर में दर्द",
        "yes_no": "(हाँ/नहीं)",
        "answer_yes_no": "कृपया हाँ या नहीं में उत्तर दें।",
        "invalid_yes_no": "कृपया हाँ या नहीं दर्ज करें।",
        "duration": "ये लक्षण कितने समय से हैं?",
        "severity": "लक्षण कितने गंभीर हैं?",
        "severity_options": "(mild/moderate/severe)",
        "invalid_severity": "कृपया mild, moderate या severe दर्ज करें।",
        "goodbye": "MediAssist का उपयोग करने के लिए धन्यवाद।",
    },
}

QUESTION_TRANSLATIONS = {
    "te": {
        "Do you have a runny nose?": "మీకు ముక్కు కారడం ఉందా?",
        "Is your nose blocked or congested?": "మీ ముక్కు బ్లాక్ అయిందా లేదా మూసుకుపోయిందా?",
        "Are you sneezing frequently?": "మీకు తరచుగా తుమ్ములు వస్తున్నాయా?",
        "Do you have a cough?": "మీకు దగ్గు ఉందా?",
        "Are you coughing up phlegm or sputum?": "మీ దగ్గుతో కఫం వస్తుందా?",
        "Do you have shortness of breath or difficulty breathing?": "మీకు శ్వాస తీసుకోవడంలో ఇబ్బంది ఉందా?",
        "Do you have a sore throat?": "మీకు గొంతు నొప్పి ఉందా?",
        "Do you have a fever?": "మీకు జ్వరం ఉందా?",
        "Do you feel unusually tired or fatigued?": "మీకు అసాధారణమైన అలసట ఉందా?",
        "Do you have a headache?": "మీకు తలనొప్పి ఉందా?",
        "Do you feel dizzy?": "మీకు తల తిరుగుతోందా?",
        "Do you feel that the room is spinning?": "చుట్టూ ఉన్న గది తిరుగుతున్నట్లు అనిపిస్తుందా?",
        "Do you feel nauseous?": "మీకు వికారం ఉందా?",
        "Have you been vomiting?": "మీకు వాంతులు అవుతున్నాయా?",
        "Do you have diarrhea or loose stools?": "మీకు విరేచనాలు లేదా పలుచని మలం ఉందా?",
        "Do you have abdominal or stomach pain?": "మీకు కడుపు నొప్పి ఉందా?",
        "Do you have a skin rash?": "మీ చర్మంపై దద్దుర్లు ఉన్నాయా?",
        "Do you have itching?": "మీకు దురద ఉందా?",
        "Do you have chest pain?": "మీకు ఛాతి నొప్పి ఉందా?",
        "Do you have joint pain?": "మీకు కీళ్ల నొప్పి ఉందా?",
        "Do you have eye pain?": "మీకు కళ్ల నొప్పి ఉందా?",
        "Are your eyes red?": "మీ కళ్లలో ఎర్రదనం ఉందా?",
        "Do you have burning or pain while urinating?": "మూత్రం పోయేటప్పుడు మంట లేదా నొప్పి ఉందా?",
        'Do you have nasal discharge?': 'మీ ముక్కు నుంచి నీరు లేదా స్రావం వస్తుందా?',
        'Do you have throat pain or irritation?': 'మీకు గొంతు నొప్పి లేదా మంటగా ఉందా?',
        'Do you have difficulty swallowing?': 'మీకు మింగడంలో ఇబ్బంది ఉందా?',
        'Do you have chills or shivering?': 'మీకు చలి వేస్తూ వణుకు వస్తుందా?',
        'Is the headache mainly in the front of your head?': 'మీ తల ముందు భాగంలో ఎక్కువగా నొప్పి ఉందా?',
        'Do you have hives or raised itchy patches?': 'మీ చర్మంపై దురదతో కూడిన ఉబ్బిన మచ్చలు ఉన్నాయా?',
        'Do you have back pain?': 'మీకు వెన్నునొప్పి ఉందా?',
        'Do you have blurred vision?': 'మీకు చూపు మసకగా కనిపిస్తున్నదా?',
        'Are you urinating more frequently than usual?': 'మీరు సాధారణం కంటే ఎక్కువసార్లు మూత్ర విసర్జన చేస్తున్నారా?',
        'Have you noticed blood in your urine?': 'మీ మూత్రంలో రక్తం కనిపించిందా?',
        'Do you have itchy or watery eyes?': 'మీ కళ్లలో దురద లేదా నీరు కారడం ఉందా?',
    },
    "hi": {
        "Do you have a runny nose?": "क्या आपकी नाक बह रही है?",
        "Is your nose blocked or congested?": "क्या आपकी नाक बंद है?",
        "Are you sneezing frequently?": "क्या आपको बार-बार छींक आ रही है?",
        "Do you have a cough?": "क्या आपको खांसी है?",
        "Are you coughing up phlegm or sputum?": "क्या खांसी के साथ बलगम आ रहा है?",
        "Do you have shortness of breath or difficulty breathing?": "क्या आपको सांस लेने में दिक्कत है?",
        "Do you have a sore throat?": "क्या आपके गले में दर्द है?",
        "Do you have a fever?": "क्या आपको बुखार है?",
        "Do you feel unusually tired or fatigued?": "क्या आपको बहुत ज्यादा थकान महसूस हो रही है?",
        "Do you have a headache?": "क्या आपको सिरदर्द है?",
        "Do you feel dizzy?": "क्या आपको चक्कर आ रहे हैं?",
        "Do you feel that the room is spinning?": "क्या आपको ऐसा लगता है कि कमरा घूम रहा है?",
        "Do you feel nauseous?": "क्या आपको मितली आ रही है?",
        "Have you been vomiting?": "क्या आपको उल्टी हो रही है?",
        "Do you have diarrhea or loose stools?": "क्या आपको दस्त हैं?",
        "Do you have abdominal or stomach pain?": "क्या आपके पेट में दर्द है?",
        "Do you have a skin rash?": "क्या आपकी त्वचा पर दाने हैं?",
        "Do you have itching?": "क्या आपको खुजली है?",
        "Do you have chest pain?": "क्या आपको सीने में दर्द है?",
        "Do you have joint pain?": "क्या आपको जोड़ों में दर्द है?",
        "Do you have eye pain?": "क्या आपकी आंखों में दर्द है?",
        "Are your eyes red?": "क्या आपकी आंखें लाल हैं?",
        "Do you have burning or pain while urinating?": "क्या पेशाब करते समय जलन या दर्द होता है?",
        'Do you have nasal discharge?': 'क्या आपकी नाक से पानी या स्राव निकलता है?',
        'Do you have throat pain or irritation?': 'क्या आपके गले में दर्द या जलन है?',
        'Do you have difficulty swallowing?': 'क्या आपको निगलने में कठिनाई होती है?',
        'Do you have chills or shivering?': 'क्या आपको ठंड लगती है या कंपकंपी होती है?',
        'Is the headache mainly in the front of your head?': 'क्या सिरदर्द मुख्य रूप से आपके माथे के हिस्से में है?',
        'Do you have hives or raised itchy patches?': 'क्या आपकी त्वचा पर खुजली वाले उभरे हुए चकत्ते हैं?',
        'Do you have back pain?': 'क्या आपकी पीठ में दर्द है?',
        'Do you have blurred vision?': 'क्या आपको धुंधला दिखाई देता है?',
        'Are you urinating more frequently than usual?': 'क्या आप सामान्य से अधिक बार पेशाब कर रहे हैं?',
        'Have you noticed blood in your urine?': 'क्या आपने अपने पेशाब में खून देखा है?',
        'Do you have itchy or watery eyes?': 'क्या आपकी आँखों में खुजली होती है या पानी आता है?',
    },
}


def t(key):
    return TEXT.get(LANGUAGE, TEXT["en"]).get(key, TEXT["en"].get(key, key))


def translate_text(text, target_language=None):
    """Translate text on demand. Failed translations are not cached."""
    value = str(text or "")
    target = target_language or LANGUAGE
    if target == "en" or not value.strip():
        return value
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="auto", target=target).translate(value)
        return result.strip() if result and result.strip() else value
    except Exception:
        return value


def translate_question(question):
    # Translate follow-up questions into the currently selected language.
    language = LANGUAGE
    if language == "en":
        return question
    known_translation = QUESTION_TRANSLATIONS.get(language, {}).get(question)
    if known_translation:
        return known_translation
    return translate_text(question, language)


def title(text):
    print("\n" + "=" * 76)
    print(text.center(76))
    print("=" * 76)


def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============================================================
# 3. DATASET DETECTION / LOADING
# ============================================================


def find_disease_column(df):
    preferred = [
        "diseases",
        "disease",
        "diagnosis",
        "prognosis",
        "label",
        "condition",
    ]

    for col in df.columns:
        name = normalize(col)
        if name in preferred:
            return col

    for col in df.columns:
        name = normalize(col)
        if any(key in name for key in preferred):
            return col

    return None


def find_symptom_columns(df, disease_col):
    symptom_cols = []

    for col in df.columns:
        if col == disease_col:
            continue

        series = df[col]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) == 0:
            continue

        unique_values = set(values.unique())
        if unique_values.issubset({0, 1}):
            symptom_cols.append(col)

    return symptom_cols


def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Dataset not found:\n" + DATA_FILE
        )

    title("LOADING 773-DISEASE DATASET")
    print(DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    print("Original shape:", df.shape)

    disease_col = find_disease_column(df)
    if disease_col is None:
        raise ValueError("Could not find the disease column.")

    symptom_cols = find_symptom_columns(df, disease_col)
    if not symptom_cols:
        raise ValueError("No numeric 0/1 symptom columns were found.")

    print("Disease column:", disease_col)
    print("Symptom columns:", len(symptom_cols))

    data = df[[disease_col] + symptom_cols].copy()

    data[disease_col] = data[disease_col].astype(str).str.strip()
    data = data[data[disease_col] != ""]

    for col in symptom_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data[col] = (data[col] > 0).astype(np.uint8)

    data = data[data[symptom_cols].sum(axis=1) > 0]
    data.drop_duplicates(inplace=True)

    print("Clean shape:", data.shape)
    print("Number of diseases:", data[disease_col].nunique())
    print("Number of samples:", len(data))

    return data, disease_col, symptom_cols

# ============================================================
# 4. SAFE TRAIN/TEST SPLIT
# ============================================================


def safe_split(y, test_fraction=0.20, random_state=42):
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    y_series = pd.Series(np.asarray(y), index=np.arange(len(y)))

    for _, group in y_series.groupby(y_series):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n = len(indices)

        if n == 1:
            train_indices.extend(indices.tolist())
            continue

        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)

        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices

# ============================================================
# 5. TRAIN / SAVE MODEL
# ============================================================


def build_profiles(X, y, symptom_cols):
    frame = pd.DataFrame(X, columns=symptom_cols)
    frame["__disease__"] = y

    grouped = frame.groupby("__disease__")
    profiles = {}

    for disease, group in grouped:
        profiles[disease] = group[symptom_cols].mean()

    return profiles


def train_and_save():
    data, disease_col, symptom_cols = load_dataset()

    title("TRAINING MEDIASSIST MODEL")

    X = data[symptom_cols].to_numpy(dtype=np.uint8)
    y = data[disease_col].to_numpy()

    print("Creating safe train/test split...")
    train_idx, test_idx = safe_split(y)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training diseases:", len(np.unique(y_train)))
    print("Testing diseases:", len(np.unique(y_test)))

    print("\nTraining classifier...")

    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=40,
        tol=1e-3,
        random_state=42,
        class_weight="balanced",
        average=True,
        n_jobs=2,
    )

    model.fit(X_train, y_train)
    print("Training completed!")

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    title("MODEL EVALUATION")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score : {metrics['f1'] * 100:.2f}%")
    print("Note: diseases with only one sample remain training-only.")

    profiles = build_profiles(X, y, symptom_cols)

    package = {
        "model": model,
        "symptom_columns": symptom_cols,
        "disease_column": disease_col,
        "profiles": profiles,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": "MediAssist-773-v2",
    }

    with open(MODEL_FILE, "wb") as file:
        pickle.dump(package, file)

    print("\nModel saved successfully:")
    print(MODEL_FILE)

    return package


def load_saved_model():
    with open(MODEL_FILE, "rb") as file:
        package = pickle.load(file)

    required = ["model", "symptom_columns", "profiles"]
    missing = [key for key in required if key not in package]

    if missing:
        raise ValueError(
            "Saved model is from an older version and is missing: "
            + ", ".join(missing)
        )

    return package

# ============================================================
# 6. TEXT -> SYMPTOM MATCHING
# ============================================================

SYNONYMS = {
    "temperature": "fever",
    "high temperature": "fever",
    "hot body": "fever",
    "tired": "fatigue",
    "very tired": "fatigue",
    "weak": "weakness",
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "vomit": "vomiting",
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "loose stool": "diarrhea",
    "loose stools": "diarrhea",
    "breathlessness": "shortness of breath",
    "breathing problem": "difficulty breathing",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "running nose": "runny nose",
    "body ache": "body pain",
    "muscle ache": "muscle pain",
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "phlegm": "sputum",
    "mucus": "sputum",
    "throat pain": "sore throat",
}


def normalize_user_text(text):
    text = normalize(text)

    for old, new in sorted(
        SYNONYMS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = text.replace(normalize(old), normalize(new))

    return text


def symptom_is_present(user_text, symptom):
    user = normalize_user_text(user_text)
    sym = normalize(symptom)

    if not sym:
        return False

    user_words = set(user.split())
    symptom_words = set(sym.split())

    # Single-word symptom: exact word only.
    if len(symptom_words) == 1:
        return sym in user_words

    # Multi-word symptom: require all meaningful words.
    return symptom_words.issubset(user_words)


def text_to_vector(text, symptom_cols):
    vector = np.zeros(len(symptom_cols), dtype=np.uint8)
    matched = []

    for i, symptom in enumerate(symptom_cols):
        if symptom_is_present(text, symptom):
            vector[i] = 1
            matched.append(symptom)

    return vector, matched

# ============================================================
# 7. PREDICTIONS
# ============================================================


def get_predictions(model, vector, top_n=5):
    probabilities = model.predict_proba([vector])[0]
    classes = model.classes_
    order = np.argsort(probabilities)[::-1]

    return [
        (classes[idx], float(probabilities[idx]))
        for idx in order[:top_n]
    ]


def show_predictions(results, heading="CURRENT PREDICTION"):
    print("\n" + "-" * 76)
    print(heading)
    print("-" * 76)

    for rank, (disease, probability) in enumerate(results, start=1):
        print(f"{rank}. {disease} - {probability * 100:.2f}%")

# ============================================================
# 8. SYMPTOM GROUPS
# ============================================================

SYMPTOM_GROUPS = {
    "nasal": {
        "keywords": [
            "runny nose", "nasal discharge", "nasal congestion",
            "blocked nose", "stuffy nose", "sneezing", "nasal", "nose",
        ],
        "templates": [
            ("runny nose", "Do you have a runny nose?"),
            ("nasal discharge", "Do you have nasal discharge?"),
            ("nasal congestion", "Is your nose blocked or congested?"),
            ("blocked nose", "Is your nose blocked or stuffy?"),
            ("sneezing", "Are you sneezing frequently?"),
        ],
    },
    "throat": {
        "keywords": [
            "sore throat", "throat pain", "throat irritation",
            "difficulty swallowing", "swallowing",
        ],
        "templates": [
            ("sore throat", "Do you have a sore throat?"),
            ("throat pain", "Do you have throat pain or irritation?"),
            ("difficulty swallowing", "Do you have difficulty swallowing?"),
        ],
    },
    "respiratory": {
        "keywords": [
            "cough", "coughing", "sputum", "phlegm",
            "shortness of breath", "difficulty breathing", "breathing", "wheezing",
        ],
        "templates": [
            ("cough", "Do you have a cough?"),
            ("coughing up sputum", "Are you coughing up phlegm or sputum?"),
            ("shortness of breath", "Do you have shortness of breath or difficulty breathing?"),
            ("wheezing", "Do you have wheezing while breathing?"),
        ],
    },
    "systemic": {
        "keywords": [
            "fever", "chills", "fatigue", "weakness",
            "body pain", "muscle pain", "malaise",
        ],
        "templates": [
            ("fever", "Do you have a fever?"),
            ("chills", "Do you have chills or shivering?"),
            ("fatigue", "Do you feel unusually tired or fatigued?"),
            ("weakness", "Do you feel unusually weak?"),
            ("body pain", "Do you have body pain or muscle aches?"),
        ],
    },
    "head_neuro": {
        "keywords": [
            "headache", "frontal headache", "dizziness", "vertigo",
            "migraine", "fainting", "confusion",
        ],
        "templates": [
            ("headache", "Do you have a headache?"),
            ("frontal headache", "Is the headache mainly in the front of your head?"),
            ("dizziness", "Do you feel dizzy?"),
            ("vertigo", "Do you feel that the room is spinning?"),
        ],
    },
    "gastrointestinal": {
        "keywords": [
            "nausea", "vomiting", "diarrhea", "abdominal pain",
            "stomach pain", "constipation", "loose stool",
        ],
        "templates": [
            ("nausea", "Do you feel nauseous?"),
            ("vomiting", "Have you been vomiting?"),
            ("diarrhea", "Do you have diarrhea or loose stools?"),
            ("abdominal pain", "Do you have abdominal or stomach pain?"),
            ("constipation", "Are you having constipation?"),
        ],
    },
    "skin": {
        "keywords": [
            "rash", "itching", "skin", "hives", "redness", "swelling",
        ],
        "templates": [
            ("rash", "Do you have a skin rash?"),
            ("itching", "Do you have itching?"),
            ("hives", "Do you have hives or raised itchy patches?"),
            ("skin redness", "Do you notice redness or swelling of the skin?"),
        ],
    },
    "pain": {
        "keywords": [
            "chest pain", "abdominal pain", "back pain", "joint pain",
            "muscle pain", "body pain", "pain",
        ],
        "templates": [
            ("chest pain", "Do you have chest pain?"),
            ("back pain", "Do you have back pain?"),
            ("joint pain", "Do you have joint pain?"),
            ("muscle pain", "Do you have muscle pain?"),
        ],
    },
    "eye": {
        "keywords": [
            "eye pain", "eye redness", "blurred vision", "vision",
            "itchy eyes", "watery eyes",
        ],
        "templates": [
            ("eye pain", "Do you have eye pain?"),
            ("eye redness", "Are your eyes red?"),
            ("blurred vision", "Do you have blurred vision?"),
            ("watery eyes", "Do your eyes water excessively?"),
        ],
    },
    "urinary": {
        "keywords": [
            "urination", "urine", "burning urination",
            "frequent urination", "blood in urine",
        ],
        "templates": [
            ("burning urination", "Do you have burning or pain while urinating?"),
            ("frequent urination", "Are you urinating more frequently than usual?"),
            ("blood in urine", "Have you noticed blood in your urine?"),
        ],
    },
    "allergy": {
        "keywords": [
            "allergy", "allergic", "itchy eyes", "sneezing", "hives",
        ],
        "templates": [
            ("itchy eyes", "Do you have itchy or watery eyes?"),
            ("hives", "Do you have hives or raised itchy patches?"),
        ],
    },
}


def get_symptom_group(symptom):
    name = normalize(symptom)

    for group_name, info in SYMPTOM_GROUPS.items():
        for keyword in info["keywords"]:
            if normalize(keyword) in name:
                return group_name

    return "other"


def get_initial_relevant_groups(initial_text):
    text = normalize_user_text(initial_text)
    scored = []

    for group_name, info in SYMPTOM_GROUPS.items():
        score = 0
        for keyword in info["keywords"]:
            key = normalize(keyword)
            if key and key in text:
                score += 1

        if score:
            scored.append((group_name, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [group for group, _ in scored]


def symptom_already_mentioned(symptom, initial_text, vector, symptom_cols):
    if symptom in symptom_cols:
        idx = symptom_cols.index(symptom)
        if vector[idx] == 1:
            return True

    symptom_name = normalize(symptom)
    user_text = normalize_user_text(initial_text)

    words = set(symptom_name.split())
    user_words = set(user_text.split())

    if len(words) == 1:
        return symptom_name in user_words

    return words.issubset(user_words)


def question_for_symptom(group_name, symptom):
    readable = normalize(symptom)
    templates = SYMPTOM_GROUPS[group_name]["templates"]

    for keyword, question in templates:
        if normalize(keyword) == readable:
            return question

    return f"Do you have {readable}?"


def choose_group_question(
    initial_text,
    vector,
    results,
    profiles,
    symptom_cols,
    asked_symptoms,
    asked_groups,
):
    candidate_diseases = [
        disease
        for disease, _ in results[:5]
        if disease in profiles
    ]

    if not candidate_diseases:
        return None

    initial_groups = get_initial_relevant_groups(initial_text)
    group_scores = {}

    # First score the groups.
    for group_name in SYMPTOM_GROUPS:
        if group_name in asked_groups:
            continue

        group_score = 0.0

        # Strongly prioritize groups suggested by the initial complaint.
        if group_name in initial_groups:
            position = initial_groups.index(group_name)
            group_score += max(12.0 - position * 2.5, 5.0)

        best_symptom_score = 0.0

        for symptom in symptom_cols:
            if get_symptom_group(symptom) != group_name:
                continue
            if symptom in asked_symptoms:
                continue
            if symptom_already_mentioned(
                symptom, initial_text, vector, symptom_cols
            ):
                continue

            values = [
                float(profiles[disease].get(symptom, 0.0))
                for disease in candidate_diseases
            ]

            if not values:
                continue

            spread = max(values) - min(values)
            relevance = max(values)
            symptom_score = 0.70 * spread + 0.30 * relevance
            best_symptom_score = max(best_symptom_score, symptom_score)

        group_score += best_symptom_score * 5.0
        group_scores[group_name] = group_score

    if not group_scores:
        return None

    selected_group = max(group_scores, key=group_scores.get)

    best_symptom = None
    best_score = -1.0

    for symptom in symptom_cols:
        if symptom in asked_symptoms:
            continue
        if get_symptom_group(symptom) != selected_group:
            continue
        if symptom_already_mentioned(
            symptom, initial_text, vector, symptom_cols
        ):
            continue

        values = [
            float(profiles[disease].get(symptom, 0.0))
            for disease in candidate_diseases
        ]

        if not values:
            continue

        spread = max(values) - min(values)
        relevance = max(values)
        score = 0.70 * spread + 0.30 * relevance

        if score > best_score:
            best_score = score
            best_symptom = symptom

    if best_symptom is None:
        return None

    return (
        selected_group,
        best_symptom,
        question_for_symptom(selected_group, best_symptom),
    )

# ============================================================
# 9. RED FLAGS / SAFETY
# ============================================================

RED_FLAGS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "cannot breathe",
    "fainting",
    "unconscious",
    "severe bleeding",
    "coughing blood",
    "vomiting blood",
    "seizure",
]


def check_red_flags(text):
    normalized = normalize_user_text(text)
    return [flag for flag in RED_FLAGS if normalize(flag) in normalized]


def show_red_flag_warning(flags):
    print("\n" + "!" * 76)
    print("POTENTIAL RED-FLAG SYMPTOM DETECTED")
    print("Detected:", ", ".join(flags))
    print("Seek appropriate urgent medical attention.")
    print("!" * 76)

# ============================================================
# 10. INPUT HELPERS
# ============================================================


def ask_yes_no(question):
    translated = translate_question(question)

    while True:
        answer = input(
            f"\n{translated} {t('yes_no')}: "
        ).strip().lower()

        if answer in {"yes", "y", "हाँ", "ha", "అవును"}:
            return True

        if answer in {"no", "n", "नहीं", "nahi", "కాదు"}:
            return False

        print(t("invalid_yes_no"))


def ask_severity():
    while True:
        severity = input(
            f"{t('severity')} {t('severity_options')}: "
        ).strip().lower()

        if severity in {"mild", "moderate", "severe"}:
            return severity

        print(t("invalid_severity"))


def choose_language():
    global LANGUAGE

    print("\nLanguage / భాష / भाषा")
    print("1. English")
    print("2. Telugu")
    print("3. Hindi")

    choice = input("Choose language [1]: ").strip()

    if choice == "2":
        LANGUAGE = "te"
    elif choice == "3":
        LANGUAGE = "hi"
    else:
        LANGUAGE = "en"

# ============================================================
# 11. SESSION STORAGE
# ============================================================


def load_json_list(path):
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_json_list(path, items):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(items, file, indent=2, ensure_ascii=False)


def save_session(session):
    history = load_json_list(HISTORY_FILE)
    history.append(session)
    save_json_list(HISTORY_FILE, history)


def show_history():
    history = load_json_list(HISTORY_FILE)

    title("MEDIASSIST SESSION HISTORY")

    if not history:
        print("No previous sessions found.")
        return

    for number, item in enumerate(history[-10:][::-1], start=1):
        print(f"\n{number}. {item.get('timestamp', 'Unknown time')}")
        print("   Symptoms:", item.get("initial_symptoms", ""))
        print("   Duration:", item.get("duration", ""))
        print("   Severity:", item.get("severity", ""))
        print("   Top classification:", item.get("top_classification", ""))

# ============================================================
# 12. APPOINTMENT / DOCTOR REQUEST DEMO
# ============================================================


def appointment_request():
    title("DOCTOR / APPOINTMENT REQUEST")

    print("This demo stores a request locally.")
    print("Connect your hospital/doctor API here for real booking.")

    name = input("Patient name: ").strip()
    phone = input("Phone number: ").strip()
    specialty = input(
        "Preferred specialty (example: General Physician): "
    ).strip()
    preferred_date = input("Preferred date: ").strip()

    request = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "phone": phone,
        "specialty": specialty,
        "preferred_date": preferred_date,
        "status": "REQUESTED",
    }

    requests = load_json_list(APPOINTMENT_FILE)
    requests.append(request)
    save_json_list(APPOINTMENT_FILE, requests)

    print("\nAppointment request saved successfully.")
    print("Status: REQUESTED")
    print("For production, connect this module to a verified provider API.")

# ============================================================
# 13. MODEL INFORMATION
# ============================================================


def show_model_info(package):
    title("MEDIASSIST MODEL INFORMATION")

    model = package["model"]
    symptom_cols = package["symptom_columns"]
    metrics = package.get("metrics", {})

    print("Model version:", package.get("model_version", "Unknown"))
    print("Trained at:", package.get("trained_at", "Unknown"))
    print("Disease classes:", len(model.classes_))
    print("Symptom features:", len(symptom_cols))

    if metrics:
        print(f"Accuracy : {metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"Precision: {metrics.get('precision', 0) * 100:.2f}%")
        print(f"Recall   : {metrics.get('recall', 0) * 100:.2f}%")
        print(f"F1 Score : {metrics.get('f1', 0) * 100:.2f}%")

# ============================================================
# 14. MEDIASSIST ASSESSMENT
# ============================================================


def run_assessment(package):
    model = package["model"]
    symptom_cols = package["symptom_columns"]
    profiles = package["profiles"]

    title("MEDIASSIST SYMPTOM ASSESSMENT")

    print(t("symptoms"))
    print(t("example"))

    initial = input("\n> ").strip()

    if not initial:
        print("No symptoms entered.")
        return

    if normalize(initial) == "exit":
        return

    initial_flags = check_red_flags(initial)

    if initial_flags:
        show_red_flag_warning(initial_flags)

        proceed = input(
            "Continue assessment for research/demo purposes? (yes/no): "
        ).strip().lower()

        if proceed not in {"yes", "y"}:
            print("Assessment stopped for safety.")
            return

    vector, matched = text_to_vector(
        initial,
        symptom_cols
    )

    print("\nSymptoms recognized by the model:")
    if matched:
        for symptom in matched[:25]:
            print(" +", symptom)
    else:
        print(" + No exact dataset symptom matched.")

    results = get_predictions(
        model,
        vector,
        top_n=5
    )

    show_predictions(
        results,
        "INITIAL MODEL CLASSIFICATION"
    )

    print("\nModel probability is not clinical certainty.")

    # --------------------------------------------------------
    # Adaptive grouped questions
    # --------------------------------------------------------

    title("ADAPTIVE FOLLOW-UP")
    print("One question is selected from each relevant symptom group.")
    print("The system prioritizes your initial complaint and avoids duplicates.")

    asked_symptoms = set()
    asked_groups = set()
    answers = []

    max_questions = 6

    for question_number in range(1, max_questions + 1):
        results = get_predictions(model, vector, top_n=5)

        selected = choose_group_question(
            initial,
            vector,
            results,
            profiles,
            symptom_cols,
            asked_symptoms,
            asked_groups,
        )

        if selected is None:
            print("\nNo more distinct relevant follow-up groups available.")
            break

        group_name, symptom, question = selected

        asked_symptoms.add(symptom)
        asked_groups.add(group_name)

        print("\n" + "-" * 76)
        print(f"Question {question_number} / {max_questions}")
        print("Symptom group:", group_name)
        print("Current leading classification:", results[0][0])
        print(f"Current model probability: {results[0][1] * 100:.2f}%")

        answer = ask_yes_no(question)
        answers.append({
            "group": group_name,
            "symptom": symptom,
            "question": question,
            "answer": "yes" if answer else "no",
        })

        idx = symptom_cols.index(symptom)
        vector[idx] = 1 if answer else 0

        updated = get_predictions(model, vector, top_n=5)
        show_predictions(updated, "UPDATED MODEL CLASSIFICATION")

    # --------------------------------------------------------
    # Duration / severity
    # --------------------------------------------------------

    print("\n" + "-" * 76)
    duration = input(
        f"{t('duration')} (example: 2 days / 1 week): "
    ).strip()

    severity = ask_severity()

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    final_results = get_predictions(model, vector, top_n=5)
    top_disease, top_probability = final_results[0]
    second_probability = final_results[1][1] if len(final_results) > 1 else 0.0
    margin = top_probability - second_probability

    title("FINAL MEDIASSIST RESULT")

    print("Duration:", duration or "Not provided")
    print("Severity:", severity)

    print("\nTop model classifications:")
    for rank, (disease, probability) in enumerate(final_results, start=1):
        print(f"{rank}. {disease} - {probability * 100:.2f}%")

    print("\n" + "-" * 76)

    if top_probability >= 0.60 and margin >= 0.15:
        print("Leading model classification:", top_disease)
        print(f"Model probability: {top_probability * 100:.2f}%")
        print("This is a model classification, not a confirmed diagnosis.")
    else:
        print("NO SUFFICIENTLY SEPARATED RESULT")
        print("The model sees overlap between possible conditions.")
        print("Do not treat any single disease label as confirmed.")

    # --------------------------------------------------------
    # Final red flags from user's original text + yes answers
    # --------------------------------------------------------

    yes_symptoms = [item["symptom"] for item in answers if item["answer"] == "yes"]
    final_context = initial + " " + " ".join(yes_symptoms)
    final_flags = check_red_flags(final_context)

    if final_flags:
        show_red_flag_warning(final_flags)

    # --------------------------------------------------------
    # Personalized next step: non-prescriptive
    # --------------------------------------------------------

    print("\nRecommended next step:")
    if final_flags or severity == "severe":
        print("Seek appropriate professional medical evaluation promptly.")
    elif top_probability < 0.30 or margin < 0.10:
        print("Consider discussing the symptoms and this summary with a healthcare professional.")
    else:
        print("Use the symptom summary below when discussing your condition with a healthcare professional.")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nSymptom summary:")
    print("Initial complaint:", initial)
    print("Recognized symptoms:", ", ".join(matched) if matched else "None")
    print("Duration:", duration or "Not provided")
    print("Severity:", severity)

    if answers:
        print("Follow-up answers:")
        for item in answers:
            print(
                f" - {item['question']} -> {item['answer'].upper()}"
            )

    session = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "initial_symptoms": initial,
        "recognized_symptoms": matched,
        "duration": duration,
        "severity": severity,
        "follow_up_answers": answers,
        "top_classification": top_disease,
        "top_probability": top_probability,
        "top_5": [
            {"disease": disease, "probability": probability}
            for disease, probability in final_results
        ],
        "red_flags": final_flags,
        "language": LANGUAGE,
    }

    save_session(session)
    print("\nSession saved to:")
    print(HISTORY_FILE)

# ============================================================
# 15. MAIN MENU
# ============================================================


def main():
    title("MEDIASSIST")
    print("Rural AI Symptom Assessment & Health Assistance System")
    print("\nMediAssist is an educational/research system, not a diagnosis service.")

    choose_language()

    # --------------------------------------------------------
    # Load or train once
    # --------------------------------------------------------

    if os.path.exists(MODEL_FILE):
        print("\nSaved model found.")
        print("Loading saved model...")

        try:
            package = load_saved_model()
        except Exception as exc:
            print("\nSaved model could not be loaded:")
            print(str(exc))
            print("\nYou need to train a compatible model once.")
            package = train_and_save()

        print("Model loaded successfully.")
        print("Training skipped.")

    else:
        print("\nNo saved model found.")
        print("Training MediAssist model for the first time...")
        package = train_and_save()

    while True:
        title("MEDIASSIST MENU")
        print("1. Start symptom assessment")
        print("2. View model information")
        print("3. View recent symptom history")
        print("4. Create doctor/appointment request")
        print("5. Change language")
        print("6. Exit")

        choice = input("\nChoose an option [1]: ").strip()

        if choice in {"", "1"}:
            run_assessment(package)

        elif choice == "2":
            show_model_info(package)

        elif choice == "3":
            show_history()

        elif choice == "4":
            appointment_request()

        elif choice == "5":
            choose_language()

        elif choice == "6":
            print("\n" + t("goodbye"))
            break

        else:
            print("Invalid option. Please choose 1-6.")


if __name__ == "__main__":
    main()
