# ============================================================
# MEDISATHI / MEDIASSIST
# 773-DISEASE SMART SYMPTOM CLASSIFIER
#
# IMPORTANT:
# Replace the COMPLETE contents of your old train_model.py
# with this file.
#
# This version:
# - Uses the 773-disease augmented CSV
# - Handles diseases with only 1 sample safely
# - Uses a RAM-friendlier SGD text-free classifier
# - Evaluates accuracy / precision / recall / F1
# - Saves the model
# - ALWAYS enters the follow-up-question stage after input
# - Uses adaptive questions when possible
# - Has generic fallback questions so it never silently skips
# ============================================================

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "archive (8)",
    "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_model"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "medisathi_773_model.pkl"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. DISPLAY HELPERS
# ============================================================

def title(text):
    print("\n" + "=" * 72)
    print(text.center(72))
    print("=" * 72)


def normalize(text):
    text = str(text).lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 3. COLUMN DETECTION
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

        if any(
            key in name
            for key in preferred
        ):
            return col

    return None


def find_symptom_columns(df, disease_col):
    symptom_cols = []

    for col in df.columns:

        if col == disease_col:
            continue

        series = df[col]

        # The 773-disease dataset should contain 0/1
        # symptom columns.
        if pd.api.types.is_numeric_dtype(series):

            values = pd.to_numeric(
                series,
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            unique_values = set(
                values.unique()
            )

            if unique_values.issubset({0, 1}):
                symptom_cols.append(col)

    return symptom_cols


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            "\nDataset not found:\n"
            + DATA_FILE
        )

    print(
        "\nLoading 773-disease dataset..."
    )

    print(
        DATA_FILE
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "\nOriginal shape:",
        df.shape
    )

    disease_col = find_disease_column(
        df
    )

    if disease_col is None:

        raise ValueError(
            "Could not find the disease column."
        )

    symptom_cols = find_symptom_columns(
        df,
        disease_col
    )

    if len(symptom_cols) == 0:

        raise ValueError(
            "No numeric 0/1 symptom columns were found."
        )

    print(
        "Disease column:",
        disease_col
    )

    print(
        "Symptom columns:",
        len(symptom_cols)
    )

    data = df[
        [disease_col] + symptom_cols
    ].copy()

    # Clean disease labels.
    data[disease_col] = (
        data[disease_col]
        .astype(str)
        .str.strip()
    )

    data = data[
        data[disease_col] != ""
    ]

    # Convert symptom matrix to compact integers.
    for col in symptom_cols:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).fillna(0)

        data[col] = (
            data[col] > 0
        ).astype(np.uint8)

    # Remove rows containing no symptoms.
    data = data[
        data[symptom_cols]
        .sum(axis=1)
        > 0
    ]

    # Remove exact duplicate rows.
    data.drop_duplicates(
        inplace=True
    )

    print(
        "\nClean shape:",
        data.shape
    )

    print(
        "Number of diseases:",
        data[disease_col].nunique()
    )

    print(
        "Number of samples:",
        len(data)
    )

    return (
        data,
        disease_col,
        symptom_cols
    )


# ============================================================
# 5. SAFE TRAIN / TEST SPLIT
# ============================================================
def safe_split(
    X,
    y,
    test_fraction=0.20,
    random_state=42
):
    """
    Safe train/test split.

    - Diseases with only 1 sample stay in TRAIN.
    - Diseases with >= 2 samples contribute to TEST.
    - Every disease keeps at least 1 training sample.
    """

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    y_series = pd.Series(
        np.asarray(y),
        index=np.arange(len(y))
    )

    for disease, group in y_series.groupby(y_series):

        # copy=True makes the NumPy array writable
        indices = group.index.to_numpy(copy=True)

        # Now shuffle works correctly
        rng.shuffle(indices)

        n = len(indices)

        # Singleton disease -> TRAIN only
        if n == 1:
            train_indices.extend(
                indices.tolist()
            )
            continue

        # Calculate test samples
        n_test = max(
            1,
            int(round(n * test_fraction))
        )

        # Keep at least 1 sample in training
        n_test = min(
            n_test,
            n - 1
        )

        test_part = indices[:n_test]
        train_part = indices[n_test:]

        train_indices.extend(
            train_part.tolist()
        )

        test_indices.extend(
            test_part.tolist()
        )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64
    )

    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        train_indices,
        test_indices
    )

# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(
    data,
    disease_col,
    symptom_cols
):

    X = data[
        symptom_cols
    ].to_numpy(
        dtype=np.uint8
    )

    y = data[
        disease_col
    ].to_numpy()

    print(
        "\nCreating safe train/test split..."
    )

    train_idx, test_idx = safe_split(
        X,
        y
    )

    X_train = X[
        train_idx
    ]

    X_test = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )

    print(
        "Training diseases:",
        len(np.unique(y_train))
    )

    print(
        "Testing diseases:",
        len(np.unique(y_test))
    )

    print(
        "\nTraining classifier..."
    )

    # SGD logistic regression is much lighter on RAM than
    # hundreds of large decision trees for this matrix.
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=40,
        tol=1e-3,
        random_state=42,
        class_weight="balanced",
        average=True,
        n_jobs=2
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed!"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print(
        "\nEvaluating model..."
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    title(
        "MODEL EVALUATION"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print(
        "\nNote: singleton diseases are kept in training, "
        "so they do not contribute test examples."
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return (
        model,
        X,
        y
    )


# ============================================================
# 7. TEXT -> SYMPTOM VECTOR
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
    "breathing problem": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "running nose": "runny nose",
    "body ache": "body pain",
    "muscle ache": "muscle pain",
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
}


def normalize_user_text(
    text
):

    text = normalize(
        text
    )

    for old, new in sorted(
        SYNONYMS.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        text = text.replace(
            normalize(old),
            normalize(new)
        )

    return text


def symptom_is_present(
    user_text,
    symptom
):

    user = normalize_user_text(
        user_text
    )

    sym = normalize(
        symptom
    )

    if not sym:
        return False

    # Exact phrase match first.
    if sym in user:
        return True

    # For single-word symptoms, require an exact word match.
    # This prevents vague words such as "cold" from matching
    # a dataset feature such as "feeling cold".
    symptom_words = sym.split()

    if len(symptom_words) == 1:
        return sym in set(user.split())

    # Do not use loose partial matching for multi-word symptoms.
    # Synonyms are normalized before reaching this function, so
    # exact phrase matching is safer and avoids misleading matches.
    return False

def text_to_vector(
    text,
    symptom_cols
):

    vector = np.zeros(
        len(symptom_cols),
        dtype=np.uint8
    )

    matched = []

    for i, symptom in enumerate(
        symptom_cols
    ):

        if symptom_is_present(
            text,
            symptom
        ):

            vector[i] = 1

            matched.append(
                symptom
            )

    return (
        vector,
        matched
    )


# ============================================================
# 8. PREDICTIONS
# ============================================================

def get_predictions(
    model,
    vector,
    top_n=5
):

    probabilities = model.predict_proba(
        [vector]
    )[0]

    classes = model.classes_

    order = np.argsort(
        probabilities
    )[::-1]

    results = []

    for idx in order[:top_n]:

        results.append(
            (
                classes[idx],
                float(
                    probabilities[idx]
                )
            )
        )

    return results


def rerank_predictions(
    model,
    vector,
    profiles,
    symptom_cols,
    answered,
    top_n=5
):
    """
    Combine the ML probability with the yes/no answers.

    The original binary classifier treats an unanswered symptom
    and an explicit "no" as the same 0 value. This small profile-
    based reranking layer lets follow-up answers affect the
    candidate ordering without retraining the model.
    """

    base_prob = model.predict_proba([vector])[0]
    classes = model.classes_

    scores = np.asarray(
        base_prob,
        dtype=float
    ).copy()

    if answered and profiles:

        for idx, disease in enumerate(classes):

            profile = profiles.get(
                disease
            )

            if profile is None:
                continue

            score = scores[idx] + 1e-12

            for symptom, answer in answered.items():

                prevalence = float(
                    profile.get(
                        symptom,
                        0.0
                    )
                )

                # Gentle evidence weighting.
                # YES favors diseases where the symptom is common.
                # NO favors diseases where it is uncommon.
                if answer:
                    factor = 0.20 + 0.80 * prevalence
                else:
                    factor = 0.20 + 0.80 * (1.0 - prevalence)

                score *= max(
                    factor,
                    0.05
                )

            scores[idx] = score

    total = scores.sum()

    if total > 0:
        scores = scores / total

    order = np.argsort(
        scores
    )[::-1]

    return [
        (
            classes[idx],
            float(scores[idx])
        )
        for idx in order[:top_n]
    ]


def show_predictions(
    results,
    heading="CURRENT PREDICTION"
):

    print(
        "\n" + "-" * 72
    )

    print(
        heading
    )

    print(
        "-" * 72
    )

    for rank, (
        disease,
        probability
    ) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. {disease}"
            f" - {probability * 100:.2f}%"
        )


# ============================================================
# 9. DISEASE/SYMPTOM PROFILES
# ============================================================

def build_profiles(
    X,
    y,
    symptom_cols
):

    profiles = {}

    frame = pd.DataFrame(
        X,
        columns=symptom_cols
    )

    frame["__disease__"] = y

    grouped = frame.groupby(
        "__disease__"
    )

    for disease, group in grouped:

        profiles[disease] = (
            group[symptom_cols]
            .mean()
        )

    return profiles


# ============================================================
# 10. QUESTION HELPERS
# ============================================================


def symptom_family(symptom):
    """Return a family so closely related features are not asked twice."""

    s = normalize(symptom)

    family_rules = [
        ("headache", ["headache", "migraine"]),
        ("cough", ["cough", "sputum", "phlegm"]),
        ("nose", ["runny nose", "nasal congestion", "stuffy nose", "blocked nose", "sneezing"]),
        ("fever", ["fever", "temperature", "chills", "feeling cold"]),
        ("breathing", ["shortness of breath", "difficulty breathing", "breathlessness"]),
        ("chest", ["chest pain", "chest tightness"]),
        ("vomiting", ["vomiting", "vomit", "nausea"]),
        ("abdominal", ["abdominal pain", "stomach pain", "belly pain"]),
        ("diarrhea", ["diarrhea", "loose stool", "loose motion"]),
        ("throat", ["sore throat", "throat pain"]),
        ("fatigue", ["fatigue", "weakness", "tired"]),
        ("dizziness", ["dizziness", "vertigo"]),
        ("rash", ["rash", "skin eruption"]),
        ("itching", ["itching", "itch"]),
        ("joint", ["joint pain", "joint swelling"]),
        ("urinary", ["urination", "urinary", "burning urination", "painful urination"]),
    ]

    for family, keys in family_rules:
        if any(key in s for key in keys):
            return family

    return s


def find_feature(
    symptom_cols,
    aliases
):
    """Find the best matching dataset feature for a human symptom name."""

    normalized = {
        normalize(col): col
        for col in symptom_cols
    }

    # Exact aliases first.
    for alias in aliases:
        key = normalize(alias)
        if key in normalized:
            return normalized[key]

    # Then safe substring matching.
    for alias in aliases:
        key = normalize(alias)
        for name, original in normalized.items():
            if key == name or key in name:
                return original

    return None


def human_question(symptom):
    """Turn dataset feature names into clearer yes/no questions."""

    s = normalize(symptom)

    special = {
        "feeling cold": "Do you feel unusually cold or have chills?",
        "frontal headache": "Is your headache mainly in the front of your head?",
        "occipital headache": "Is your headache mainly at the back of your head?",
        "coughing up sputum": "Are you coughing up phlegm or sputum?",
        "nasal congestion": "Is your nose blocked or stuffy?",
        "runny nose": "Do you have a runny nose?",
        "shortness of breath": "Are you having shortness of breath or difficulty breathing?",
        "difficulty breathing": "Are you having difficulty breathing?",
        "body pain": "Do you have body aches or general body pain?",
        "muscle pain": "Do you have muscle pain or body aches?",
        "sore throat": "Do you have a sore throat?",
        "abdominal pain": "Do you have stomach or abdominal pain?",
        "burning urination": "Do you have burning or pain when urinating?",
        "blurred vision": "Do you have blurred or unclear vision?",
    }

    if s in special:
        return special[s]

    readable = s.replace(
        "_",
        " "
    )

    return f"Do you have {readable}?"


QUESTION_BANK = [
    (
        "runny nose",
        ["runny nose"]
    ),
    (
        "nasal congestion",
        ["nasal congestion", "stuffy nose", "blocked nose"]
    ),
    (
        "sneezing",
        ["sneezing"]
    ),
    (
        "fever",
        ["fever"]
    ),
    (
        "sore throat",
        ["sore throat"]
    ),
    (
        "cough",
        ["cough"]
    ),
    (
        "coughing up sputum",
        ["coughing up sputum", "sputum", "phlegm"]
    ),
    (
        "body pain",
        ["body pain", "body aches"]
    ),
    (
        "headache",
        ["headache"]
    ),
    (
        "nausea",
        ["nausea"]
    ),
    (
        "vomiting",
        ["vomiting"]
    ),
    (
        "dizziness",
        ["dizziness"]
    ),
    (
        "shortness of breath",
        ["shortness of breath", "difficulty breathing"]
    ),
    (
        "chest pain",
        ["chest pain"]
    ),
    (
        "diarrhea",
        ["diarrhea", "loose stools"]
    ),
    (
        "abdominal pain",
        ["abdominal pain", "stomach pain", "belly pain"]
    ),
    (
        "rash",
        ["rash"]
    ),
    (
        "itching",
        ["itching"]
    ),
    (
        "joint pain",
        ["joint pain"]
    ),
]


def context_queue(
    initial,
    symptom_cols
):
    """Return focused questions based on the user's words."""

    text = normalize_user_text(initial)

    queue = []

    def add(*names):
        for name in names:
            feature = find_feature(
                symptom_cols,
                [name]
            )
            if feature is not None and feature not in queue:
                queue.append(feature)

    # Vague "cold" is treated as an ambiguous complaint, NOT
    # as the dataset feature "feeling cold".
    if any(term in text for term in [
        "cold",
        "common cold",
        "runny nose",
        "stuffy nose",
        "blocked nose",
        "sneezing",
    ]):
        add(
            "runny nose",
            "nasal congestion",
            "sneezing",
            "sore throat",
            "cough",
            "fever",
            "body pain",
            "headache",
            "shortness of breath",
            "chest pain",
        )

    elif any(term in text for term in [
        "cough",
        "phlegm",
        "sputum",
    ]):
        add(
            "fever",
            "sore throat",
            "coughing up sputum",
            "shortness of breath",
            "chest pain",
            "runny nose",
            "headache",
        )

    elif any(term in text for term in [
        "headache",
        "migraine",
        "head pain",
    ]):
        add(
            "fever",
            "nausea",
            "vomiting",
            "dizziness",
            "blurred vision",
            "sore throat",
        )

    elif any(term in text for term in [
        "stomach",
        "abdominal",
        "belly",
        "diarrhea",
        "vomiting",
        "nausea",
    ]):
        add(
            "nausea",
            "vomiting",
            "diarrhea",
            "abdominal pain",
            "fever",
            "body pain",
        )

    else:
        # Generic but useful order when the complaint is unclear.
        add(
            "fever",
            "cough",
            "sore throat",
            "headache",
            "nausea",
            "shortness of breath",
            "chest pain",
            "rash",
        )

    return queue


def choose_adaptive_question(
    vector,
    results,
    profiles,
    symptom_cols,
    asked_families,
    answered,
):
    """Choose an informative symptom that also reads naturally."""

    if not profiles or len(results) < 2:
        return None

    candidate_diseases = [
        disease
        for disease, probability in results[:10]
        if disease in profiles
    ]

    if len(candidate_diseases) < 2:
        return None

    weights = np.array([
        max(
            float(
                next(
                    (p for d, p in results if d == disease),
                    0.0
                )
            ),
            1e-6
        )
        for disease in candidate_diseases
    ], dtype=float)

    weights /= weights.sum()

    best_symptom = None
    best_score = -1.0

    for i, symptom in enumerate(
        symptom_cols
    ):
        if vector[i] == 1:
            continue

        if symptom in answered:
            continue

        family = symptom_family(symptom)
        if family in asked_families:
            continue

        prevalence = np.array([
            float(
                profiles[disease].get(
                    symptom,
                    0.0
                )
            )
            for disease in candidate_diseases
        ])

        weighted_yes = float(
            np.dot(
                weights,
                prevalence
            )
        )

        # Symptoms that are almost always yes or almost always no
        # are weak questions.
        if weighted_yes < 0.05 or weighted_yes > 0.95:
            continue

        spread = float(
            prevalence.max() - prevalence.min()
        )

        relevance = float(
            prevalence.max()
        )

        # Expected binary entropy, weighted by candidate probability.
        entropy = 0.0
        for p in prevalence:
            if p > 0 and p < 1:
                entropy += (-p * np.log2(p) - (1-p) * np.log2(1-p))
        entropy /= len(prevalence)

        score = (
            0.45 * spread
            + 0.30 * entropy
            + 0.25 * relevance
        )

        if score > best_score:
            best_score = score
            best_symptom = symptom

    return best_symptom

# ============================================================
# 12. YES / NO
# ============================================================

def ask_yes_no(
    question
):

    while True:

        answer = input(
            f"\n{question} (yes/no): "
        ).strip().lower()

        if answer in [
            "yes",
            "y"
        ]:
            return True

        if answer in [
            "no",
            "n"
        ]:
            return False

        print(
            "Please enter yes or no."
        )


# ============================================================
# 13. RED FLAGS
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


def check_red_flags(
    text
):

    normalized = normalize(
        text
    )

    found = []

    for flag in RED_FLAGS:

        if normalize(
            flag
        ) in normalized:

            found.append(
                flag
            )

    return found


# ============================================================
# 14. USER SESSION
# ============================================================

def run_session(
    model,
    symptom_cols,
    profiles,
):
    title(
        "MEDISATHI SYMPTOM SESSION"
    )

    print(
        "Enter the symptoms in your own words."
    )

    print(
        "Example: fever, cough and body pain"
    )

    print(
        "Type 'exit' to close the program."
    )

    initial = input(
        "\nEnter symptoms: "
    ).strip()

    if initial.lower() in (
        "exit",
        "quit",
    ):
        print(
            "\nMediSathi closed."
        )
        return

    if not initial:
        print(
            "\nNo symptoms entered."
        )
        return

    # --------------------------------------------------------
    # RED FLAGS
    # --------------------------------------------------------

    red_flags = check_red_flags(
        initial
    )

    if red_flags:
        print(
            "\n" + "!" * 72
        )
        print(
            "POTENTIAL RED-FLAG SYMPTOM DETECTED"
        )
        print(
            "Detected:",
            ", ".join(red_flags)
        )
        print(
            "Please seek appropriate medical attention."
        )
        print(
            "!" * 72
        )

    # --------------------------------------------------------
    # VECTORIZE
    # --------------------------------------------------------

    vector, matched = text_to_vector(
        initial,
        symptom_cols
    )

    print(
        "\nSymptoms recognized by the model:"
    )

    if matched:
        for symptom in matched[:25]:
            print(
                " +",
                symptom
            )
    else:
        print(
            " + No clear dataset symptom matched yet."
        )

        print(
            "   I will ask a few clarifying questions first."
        )

    # --------------------------------------------------------
    # QUESTION SESSION
    # --------------------------------------------------------

    title(
        "FOLLOW-UP QUESTIONS"
    )

    print(
        "Questions are selected from your complaint first, "
        "then refined using the model candidates."
    )

    print(
        "The system will ask up to 6 focused questions."
    )

    answered = {}
    asked_families = set()
    context_questions = context_queue(
        initial,
        symptom_cols
    )

    # Only show a model prediction immediately when there is at
    # least one clear recognized symptom. For vague input such as
    # "cold", asking clarification first is more useful.
    if matched:
        results = get_predictions(
            model,
            vector,
            top_n=5
        )

        show_predictions(
            results,
            "INITIAL MODEL PREDICTION"
        )
    else:
        results = get_predictions(
            model,
            vector,
            top_n=5
        )

    MAX_QUESTIONS = 6

    for question_number in range(
        1,
        MAX_QUESTIONS + 1
    ):

        # First prefer complaint-specific questions.
        question_symptom = None

        while context_questions:
            candidate = context_questions.pop(0)

            if candidate in answered:
                continue

            if vector[symptom_cols.index(candidate)] == 1:
                continue

            family = symptom_family(candidate)

            if family in asked_families:
                continue

            question_symptom = candidate
            break

        # Once the focused queue is exhausted, use the learned
        # disease profiles to select a discriminative question.
        if question_symptom is None:
            results = rerank_predictions(
                model,
                vector,
                profiles,
                symptom_cols,
                answered,
                top_n=5
            )

            question_symptom = choose_adaptive_question(
                vector,
                results,
                profiles,
                symptom_cols,
                asked_families,
                answered,
            )

        if question_symptom is None:
            print(
                "\nNo additional focused question is available."
            )
            break

        family = symptom_family(
            question_symptom
        )

        asked_families.add(
            family
        )

        question_text = human_question(
            question_symptom
        )

        # Current ranking, using the answers collected so far.
        results = rerank_predictions(
            model,
            vector,
            profiles,
            symptom_cols,
            answered,
            top_n=5
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"Question {question_number}"
            f" / {MAX_QUESTIONS}"
        )

        if results:
            print(
                "Current leading possibility:",
                results[0][0]
            )

            print(
                f"Current model score: "
                f"{results[0][1] * 100:.2f}%"
            )

        answer = ask_yes_no(
            question_text
        )

        index = symptom_cols.index(
            question_symptom
        )

        # Store the explicit answer for profile-based reranking.
        answered[question_symptom] = answer

        if answer:
            vector[index] = 1
            print(
                f"Recorded YES for: "
                f"{normalize(question_symptom)}"
            )
        else:
            vector[index] = 0
            print(
                f"Recorded NO for: "
                f"{normalize(question_symptom)}"
            )

        updated_results = rerank_predictions(
            model,
            vector,
            profiles,
            symptom_cols,
            answered,
            top_n=5
        )

        show_predictions(
            updated_results,
            "UPDATED POSSIBILITY LIST"
        )

        # Stop once the top two are sufficiently separated after
        # at least three clarifying answers.
        if len(answered) >= 3 and len(updated_results) >= 2:
            margin = (
                updated_results[0][1]
                - updated_results[1][1]
            )

            if (
                updated_results[0][1] >= 0.60
                and margin >= 0.15
            ):
                print(
                    "\nThe current possibilities are sufficiently separated, "
                    "so no more routine questions are needed."
                )
                break

    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    print(
        "\n" + "-" * 72
    )

    duration = input(
        "How long have you had these symptoms? "
        "(example: 2 days / 1 week): "
    ).strip()

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    while True:
        severity = input(
            "How severe are your symptoms? "
            "(mild/moderate/severe): "
        ).strip().lower()

        if severity in (
            "mild",
            "moderate",
            "severe"
        ):
            break

        print(
            "Please enter mild, moderate, or severe."
        )

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    final_results = rerank_predictions(
        model,
        vector,
        profiles,
        symptom_cols,
        answered,
        top_n=5
    )

    top_disease = final_results[0][0]
    top_probability = final_results[0][1]

    if len(final_results) > 1:
        second_probability = final_results[1][1]
    else:
        second_probability = 0.0

    margin = (
        top_probability
        - second_probability
    )

    title(
        "FINAL MEDISATHI RESULT"
    )

    print(
        "\nDuration:",
        duration if duration else "Not provided"
    )

    print(
        "Severity:",
        severity
    )

    print(
        "\nFinal top possibilities:"
    )

    for rank, (
        disease,
        probability
    ) in enumerate(
        final_results,
        start=1
    ):
        print(
            f"{rank}. {disease}"
            f" - {probability * 100:.2f}%"
        )

    print(
        "\n" + "-" * 72
    )

    if (
        top_probability >= 0.60
        and margin >= 0.15
    ):
        print(
            "Leading model classification:",
            top_disease
        )

        print(
            f"Model score: "
            f"{top_probability * 100:.2f}%"
        )

        print(
            "\nThis is a model classification, "
            "not a confirmed diagnosis."
        )

    else:
        print(
            "NO SUFFICIENTLY SEPARATED RESULT"
        )

        print(
            "The model still sees substantial overlap "
            "between possible conditions."
        )

        print(
            "\nDo not treat one disease label as confirmed."
        )

    # --------------------------------------------------------
    # FINAL RED FLAG CHECK
    # --------------------------------------------------------

    all_answered_yes = [
        symptom
        for symptom, answer in answered.items()
        if answer
    ]

    combined_context = (
        initial
        + " "
        + " ".join(matched)
        + " "
        + " ".join(all_answered_yes)
    )

    final_flags = check_red_flags(
        combined_context
    )

    if final_flags:
        print(
            "\n" + "!" * 72
        )

        print(
            "IMPORTANT MEDICAL WARNING"
        )

        print(
            "Potential red-flag symptoms:",
            ", ".join(final_flags)
        )

        print(
            "Please seek appropriate medical attention."
        )

        print(
            "!" * 72
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    print(
        "\n" + "-" * 72
    )

    print(
        "DISCLAIMER"
    )

    print(
        "MediSathi is an educational/research "
        "symptom classification system."
    )

    print(
        "It is NOT a medical diagnosis."
    )

    print(
        "For severe, sudden, or worsening symptoms, "
        "seek professional medical care."
    )

    print(
        "-" * 72
    )

    print(
        "\nMediSathi session completed."
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

def save_model_package(
    model,
    symptom_cols,
    disease_col,
    profiles,
):
    package = {
        "model": model,
        "symptom_columns": symptom_cols,
        "disease_column": disease_col,
        "profiles": profiles,
    }

    with open(
        MODEL_FILE,
        "wb"
    ) as file:
        pickle.dump(
            package,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        "\nModel saved successfully:"
    )

    print(
        MODEL_FILE
    )


# ============================================================
# 16. TRAIN ONLY WHEN MODEL DOES NOT EXIST
# ============================================================

def train_and_save():
    title(
        "LOADING 773-DISEASE DATASET"
    )

    (
        data,
        disease_col,
        symptom_cols
    ) = load_dataset()

    title(
        "TRAINING MODEL"
    )

    (
        model,
        X,
        y
    ) = train_model(
        data,
        disease_col,
        symptom_cols
    )

    title(
        "BUILDING FOLLOW-UP PROFILES"
    )

    profiles = build_profiles(
        X,
        y,
        symptom_cols
    )

    save_model_package(
        model,
        symptom_cols,
        disease_col,
        profiles
    )

    return (
        model,
        symptom_cols,
        disease_col,
        profiles
    )


# ============================================================
# 17. LOAD EXISTING SAVED MODEL
# ============================================================

def load_saved_package():
    print(
        "\nSaved model found."
    )

    print(
        "Loading saved model..."
    )

    try:
        with open(
            MODEL_FILE,
            "rb"
        ) as file:
            package = pickle.load(file)

        if not isinstance(
            package,
            dict
        ):
            raise ValueError(
                "Saved model has an invalid format."
            )

        required = [
            "model",
            "symptom_columns"
        ]

        for key in required:
            if key not in package:
                raise ValueError(
                    f"Saved model is missing: {key}"
                )

        model = package["model"]

        symptom_cols = package[
            "symptom_columns"
        ]

        disease_col = package.get(
            "disease_column",
            "diseases"
        )

        profiles = package.get(
            "profiles",
            {}
        )

        print(
            "Model loaded successfully."
        )

        return (
            model,
            symptom_cols,
            disease_col,
            profiles
        )

    except Exception as exc:
        print(
            "\nCould not load saved model."
        )

        print(
            "Reason:",
            exc
        )

        return None


# ============================================================
# 18. MAIN
# ============================================================

def main():
    title(
        "MEDISATHI / MEDIASSIST"
    )

    print(
        "773-disease symptom classification "
        "with adaptive follow-up questions."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # If model exists, DO NOT train again.
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_FILE
    ):
        package = load_saved_package()

        if package is not None:
            (
                model,
                symptom_cols,
                _disease_col,
                profiles
            ) = package

            print(
                "\nUsing saved model."
            )

            print(
                "Training skipped."
            )

            run_session(
                model,
                symptom_cols,
                profiles
            )

            return

        print(
            "\nThe existing saved model is invalid."
        )

        print(
            "A new model will be trained."
        )

    # --------------------------------------------------------
    # FIRST RUN ONLY
    # --------------------------------------------------------

    (
        model,
        symptom_cols,
        _disease_col,
        profiles
    ) = train_and_save()

    run_session(
        model,
        symptom_cols,
        profiles
    )


# ============================================================
# 19. PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()
