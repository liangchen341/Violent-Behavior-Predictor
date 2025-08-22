from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_dataset(n_patients: int = 346, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # patient ids
    patient_id = np.arange(1, n_patients + 1)

    # static features (English)
    age = rng.integers(18, 66, size=n_patients)
    disease_duration_years = np.clip(np.round(rng.normal(14, 6, size=n_patients), 1), 0, 40)
    gender = rng.choice(["male", "female"], size=n_patients, p=[0.55, 0.45])
    employment = rng.choice(["employed", "unemployed"], size=n_patients, p=[0.8, 0.2])
    marital_status = rng.choice(["married", "unmarried"], size=n_patients, p=[0.25, 0.75])
    education_level = rng.choice(
        ["no_schooling_or_elementary", "junior_high", "high_school_or_vocational", "college_or_higher"],
        size=n_patients, p=[0.2, 0.35, 0.3, 0.15]
    )
    personality = rng.choice(["introverted", "extroverted"], size=n_patients, p=[0.6, 0.4])
    substance_abuse = rng.choice(["yes", "no"], size=n_patients, p=[0.35, 0.65])
    history_of_violence_or_suicide = rng.choice(["yes", "no"], size=n_patients, p=[0.75, 0.25])
    high_risk_command_hallucinations = rng.choice(["yes", "no"], size=n_patients, p=[0.08, 0.92])
    persecutory_delusions = rng.choice(["yes", "no"], size=n_patients, p=[0.55, 0.45])
    thought_activity = rng.choice(["abnormal", "normal"], size=n_patients, p=[0.85, 0.15])
    sensation_and_perception = rng.choice(["abnormal", "normal"], size=n_patients, p=[0.6, 0.4])
    intelligence = rng.choice(["abnormal", "normal"], size=n_patients, p=[0.08, 0.92])
    attention = rng.choice(["abnormal", "normal"], size=n_patients, p=[0.18, 0.82])
    memory = rng.choice(["abnormal", "normal"], size=n_patients, p=[0.08, 0.92])
    hopelessness_or_depression = rng.choice(["yes", "no"], size=n_patients, p=[0.12, 0.88])
    mania = rng.choice(["yes", "no"], size=n_patients, p=[0.03, 0.97])

    # behavioral features (0-3 ordinal, 39 items)
    behavior_names = [
        "Rule Compliance", "Personal Affairs Management", "Bed Making", "Cleaning", "Clothing Adjustment",
        "Physical Discomfort Description", "Work Therapy Participation", "Attitude Towards Others",
        "Interest in Surroundings", "Conversation with Others", "Family Concern", "Discussing Personal Interests",
        "Laughs at Jokes", "Recreational Activities", "Exercise Participation", "Neat Appearance", "Face Washing",
        "Teeth Brushing", "Personal Hygiene", "Foot Washing", "Hand Washing Before Meals", "Eating", "Hair Grooming",
        "Anger Expression", "Rapid Speech", "Agitation", "Cooperation with Staff", "Talking to Self",
        "Inappropriate Laughing", "Auditory Hallucinations", "Immobility", "Lying Down", "Psychomotor Retardation",
        "Insomnia", "Crying", "Self-reported Depression", "Negative Self-Evaluation", "Illness Awareness",
        "Discharge Request",
    ]

    # base ordinal draws
    behavior_mat = rng.integers(0, 4, size=(n_patients, len(behavior_names)))

    # Construct a synthetic latent risk score with some known contributors
    # Static contributors: younger age, history of violence, mania, high-risk command hallucinations
    risk = (
        (60 - age) * 0.02
        + (history_of_violence_or_suicide == "yes") * 0.9
        + (mania == "yes") * 1.1
        + (high_risk_command_hallucinations == "yes") * 0.6
    )
    # Dynamic contributors: anger expression, insomnia, auditory hallucinations
    def get_col(name: str) -> np.ndarray:
        idx = behavior_names.index(name)
        return behavior_mat[:, idx]

    risk += get_col("Anger Expression") * 0.35
    risk += get_col("Insomnia") * 0.35
    risk += get_col("Auditory Hallucinations") * 0.25
    risk -= get_col("Psychomotor Retardation") * 0.15
    risk -= get_col("Illness Awareness") * 0.1

    # Normalize risk to probability via logistic transform
    prob = 1 / (1 + np.exp(-(risk - np.mean(risk))))
    high_risk_group = (rng.random(n_patients) < prob).astype(int)

    data = {
        "patient_id": patient_id,
        "age": age,
        "disease_duration_years": disease_duration_years,
        "gender": gender,
        "employment": employment,
        "marital_status": marital_status,
        "education_level": education_level,
        "personality": personality,
        "substance_abuse": substance_abuse,
        "history_of_violence_or_suicide": history_of_violence_or_suicide,
        "high_risk_command_hallucinations": high_risk_command_hallucinations,
        "persecutory_delusions": persecutory_delusions,
        "thought_activity": thought_activity,
        "sensation_and_perception": sensation_and_perception,
        "intelligence": intelligence,
        "attention": attention,
        "memory": memory,
        "hopelessness_or_depression": hopelessness_or_depression,
        "mania": mania,
    }

    for j, name in enumerate(behavior_names):
        data[name] = behavior_mat[:, j]

    data["high_risk_group"] = high_risk_group

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = make_synthetic_dataset()
    df.to_csv("merged_patient_data.csv", index=False)
    print("Synthetic dataset written to merged_patient_data.csv with shape:", df.shape)


