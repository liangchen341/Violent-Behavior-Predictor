from __future__ import annotations

# Static feature configuration (English)
STATIC_FEATURES = {
    "continuous": [
        "age",
        "disease_duration_years",
    ],
    "binary": [
        "gender",
        "employment",
        "marital_status",
        "personality",
        "substance_abuse",
        "history_of_violence_or_suicide",
        "high_risk_command_hallucinations",
        "persecutory_delusions",
        "thought_activity",
        "sensation_and_perception",
        "intelligence",
        "attention",
        "memory",
        "hopelessness_or_depression",
        "mania",
    ],
    "ordinal": [
        "education_level",
    ],
}


# Behavioral (dynamic) features - English names, 39 items
BEHAVIOR_FEATURES = [
    "Rule Compliance",
    "Personal Affairs Management",
    "Bed Making",
    "Cleaning",
    "Clothing Adjustment",
    "Physical Discomfort Description",
    "Work Therapy Participation",
    "Attitude Towards Others",
    "Interest in Surroundings",
    "Conversation with Others",
    "Family Concern",
    "Discussing Personal Interests",
    "Laughs at Jokes",
    "Recreational Activities",
    "Exercise Participation",
    "Neat Appearance",
    "Face Washing",
    "Teeth Brushing",
    "Personal Hygiene",
    "Foot Washing",
    "Hand Washing Before Meals",
    "Eating",
    "Hair Grooming",
    "Anger Expression",
    "Rapid Speech",
    "Agitation",
    "Cooperation with Staff",
    "Talking to Self",
    "Inappropriate Laughing",
    "Auditory Hallucinations",
    "Immobility",
    "Lying Down",
    "Psychomotor Retardation",
    "Insomnia",
    "Crying",
    "Self-reported Depression",
    "Negative Self-Evaluation",
    "Illness Awareness",
    "Discharge Request",
]


TARGET_COLUMN = "high_risk_group"


# Binary variable mappings for encoding English categorical values to 0/1
BINARY_MAPPINGS = {
    "gender": {"male": 1, "female": 0},
    "employment": {"employed": 1, "unemployed": 0},
    "marital_status": {"married": 1, "unmarried": 0},
    "personality": {"extroverted": 1, "introverted": 0},
    "substance_abuse": {"yes": 1, "no": 0},
    "history_of_violence_or_suicide": {"yes": 1, "no": 0},
    "high_risk_command_hallucinations": {"yes": 1, "no": 0},
    "persecutory_delusions": {"yes": 1, "no": 0},
    "thought_activity": {"abnormal": 1, "normal": 0},
    "sensation_and_perception": {"abnormal": 1, "normal": 0},
    "intelligence": {"abnormal": 1, "normal": 0},
    "attention": {"abnormal": 1, "normal": 0},
    "memory": {"abnormal": 1, "normal": 0},
    "hopelessness_or_depression": {"yes": 1, "no": 0},
    "mania": {"yes": 1, "no": 0},
}


# Ordered mapping for education level
EDUCATION_MAPPING = {
    "no_schooling_or_elementary": 0,
    "junior_high": 1,
    "high_school_or_vocational": 2,
    "college_or_higher": 3,
}


# English labels mapping (identity in the English version)
ENGLISH_LABELS = {
    # Behavioral (dynamic) features
    "Rule Compliance": "Rule Compliance",
    "Personal Affairs Management": "Personal Affairs Management",
    "Bed Making": "Bed Making",
    "Cleaning": "Cleaning",
    "Clothing Adjustment": "Clothing Adjustment",
    "Physical Discomfort Description": "Physical Discomfort Description",
    "Work Therapy Participation": "Work Therapy Participation",
    "Attitude Towards Others": "Attitude Towards Others",
    "Interest in Surroundings": "Interest in Surroundings",
    "Conversation with Others": "Conversation with Others",
    "Family Concern": "Family Concern",
    "Discussing Personal Interests": "Discussing Personal Interests",
    "Laughs at Jokes": "Laughs at Jokes",
    "Recreational Activities": "Recreational Activities",
    "Exercise Participation": "Exercise Participation",
    "Neat Appearance": "Neat Appearance",
    "Face Washing": "Face Washing",
    "Teeth Brushing": "Teeth Brushing",
    "Personal Hygiene": "Personal Hygiene",
    "Foot Washing": "Foot Washing",
    "Hand Washing Before Meals": "Hand Washing Before Meals",
    "Eating": "Eating",
    "Hair Grooming": "Hair Grooming",
    "Anger Expression": "Anger Expression",
    "Rapid Speech": "Rapid Speech",
    "Agitation": "Agitation",
    "Cooperation with Staff": "Cooperation with Staff",
    "Talking to Self": "Talking to Self",
    "Inappropriate Laughing": "Inappropriate Laughing",
    "Auditory Hallucinations": "Auditory Hallucinations",
    "Immobility": "Immobility",
    "Lying Down": "Lying Down",
    "Psychomotor Retardation": "Psychomotor Retardation",
    "Insomnia": "Insomnia",
    "Crying": "Crying",
    "Self-reported Depression": "Self-reported Depression",
    "Negative Self-Evaluation": "Negative Self-Evaluation",
    "Illness Awareness": "Illness Awareness",
    "Discharge Request": "Discharge Request",
    # Static features
    "age": "Age",
    "disease_duration_years": "Disease Duration (years)",
    "gender": "Gender",
    "employment": "Employment",
    "marital_status": "Marital Status",
    "education_level": "Education Level",
    "personality": "Personality",
    "substance_abuse": "Substance Abuse",
    "history_of_violence_or_suicide": "History of Violence/Suicide",
    "high_risk_command_hallucinations": "High-risk Command Hallucinations",
    "persecutory_delusions": "Persecutory Delusions",
    "thought_activity": "Thought Activity",
    "sensation_and_perception": "Sensation and Perception",
    "intelligence": "Intelligence",
    "attention": "Attention",
    "memory": "Memory",
    "hopelessness_or_depression": "Hopelessness/Depression",
    "mania": "Mania",
    "high_risk_group": "High-risk Group",
}


