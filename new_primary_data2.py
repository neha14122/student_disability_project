import pandas as pd
import numpy as np

np.random.seed(42)

# Disability distribution (approximate realistic proportions from AISHE/UDISE patterns)
disability_distribution = {
    "Blindness": 0.05,
    "Low Vision": 0.15,
    "Hearing impairment": 0.10,
    "Speech and Language": 0.08,
    "Locomotor Disability": 0.20,
    "Mental illness": 0.07,
    "Specific Learning Disabilities": 0.10,
    "Cerebral palsy": 0.05,
    "Autism Spectrum Disorder": 0.05,
    "Intellectual Disability": 0.15,
}

genders = ["Boys", "Girls"]
regions = ["Urban", "Rural"]
ses_levels = ["Low", "Medium", "High"]
assistive_tech = ["Yes", "No"]
training = ["None", "Workshop", "Formal Training"]
parent_involve = ["Low", "Medium", "High"]
curriculum = ["None", "Some", "Significant"]

# --- Balanced Dataset ---
n_rows = 400
disabilities = list(disability_distribution.keys())
rows_per_disability = n_rows // len(disabilities)

balanced = {
    "Disability_Type": np.repeat(disabilities, rows_per_disability),
    "Gender": np.tile(np.random.choice(genders, rows_per_disability), len(disabilities)),
    "Region": np.random.choice(regions, n_rows),
    "Socioeconomic_Status": np.random.choice(ses_levels, n_rows),
    "Assistive_Tech_Use": np.random.choice(assistive_tech, n_rows),
    "Specialized_Training_Access": np.random.choice(training, n_rows),
    "Parental_Involvement": np.random.choice(parent_involve, n_rows),
    "Curriculum_Adaptation": np.random.choice(curriculum, n_rows),
}
df_balanced = pd.DataFrame(balanced)

# --- Realistic Dataset ---
realistic_rows = 400
realistic = []

for dis, proportion in disability_distribution.items():
    n_dis = int(realistic_rows * proportion)
    for _ in range(n_dis):
        realistic.append({
            "Disability_Type": dis,
            "Gender": np.random.choice(genders),
            "Region": np.random.choice(regions),
            "Socioeconomic_Status": np.random.choice(ses_levels),
            "Assistive_Tech_Use": np.random.choice(assistive_tech),
            "Specialized_Training_Access": np.random.choice(training),
            "Parental_Involvement": np.random.choice(parent_involve),
            "Curriculum_Adaptation": np.random.choice(curriculum),
        })

df_realistic = pd.DataFrame(realistic)

# --- Add Survival/Transition/Employment Logic ---
def add_outcomes(df):
    high_dropout = ["Intellectual Disability", "Mental illness", "Cerebral palsy"]
    medium_dropout = ["Hearing impairment", "Autism Spectrum Disorder", "Speech and Language"]

    df["Survival_to_Class10"] = np.where(
        df["Disability_Type"].isin(high_dropout),
        np.random.choice([0, 1], size=len(df), p=[0.8, 0.2]),
        np.where(
            df["Disability_Type"].isin(medium_dropout),
            np.random.choice([0, 1], size=len(df), p=[0.6, 0.4]),
            np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
        )
    )

    df["Transition_to_Class12"] = df["Survival_to_Class10"] * np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])

    df["Employment_Prospect"] = np.where(
        (df["Parental_Involvement"] == "High") | (df["Socioeconomic_Status"] == "High"),
        np.random.choice([0, 1], size=len(df), p=[0.3, 0.7]),
        np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
    )
    return df

df_balanced = add_outcomes(df_balanced)
df_realistic = add_outcomes(df_realistic)

# --- Add some nulls ---
for col in ["Region", "Assistive_Tech_Use", "Parental_Involvement", "Curriculum_Adaptation"]:
    df_balanced.loc[df_balanced.sample(frac=0.08).index, col] = np.nan
    df_realistic.loc[df_realistic.sample(frac=0.08).index, col] = np.nan

# --- Save ---
df_balanced.to_csv("Maharashtra_Disability_Synthetic_Balanced1.csv", index=False)
df_realistic.to_csv("Maharashtra_Disability_Synthetic_Realistic.csv", index=False)

print("✅ Datasets saved:")
print("Balanced shape:", df_balanced.shape)
print("Realistic shape:", df_realistic.shape)
print("\nBalanced distribution:\n", df_balanced["Disability_Type"].value_counts())
print("\nRealistic distribution:\n", df_realistic["Disability_Type"].value_counts())
