import pandas as pd
import numpy as np

np.random.seed(42)

# Categories
disabilities = [
    "Blindness", "Low Vision", "Hearing impairment", "Speech and Language",
    "Locomotor Disability", "Mental illness", "Specific Learning Disabilities",
    "Cerebral palsy", "Autism Spectrum Disorder", "Intellectual Disability"
]
genders = ["Boys", "Girls"]
regions = ["Urban", "Rural"]
ses_levels = ["Low", "Medium", "High"]
assistive_tech = ["Yes", "No"]
training = ["None", "Workshop", "Formal Training"]
parent_involve = ["Low", "Medium", "High"]
curriculum = ["None", "Some", "Significant"]

# --- Balanced Row Count ---
n_rows = 400
rows_per_disability = n_rows // len(disabilities)

data = {
    "Disability_Type": np.repeat(disabilities, rows_per_disability),
    "Gender": np.tile(np.random.choice(genders, rows_per_disability), len(disabilities)),
    "Region": np.random.choice(regions, n_rows),
    "Socioeconomic_Status": np.random.choice(ses_levels, n_rows),
    "Assistive_Tech_Use": np.random.choice(assistive_tech, n_rows),
    "Specialized_Training_Access": np.random.choice(training, n_rows),
    "Parental_Involvement": np.random.choice(parent_involve, n_rows),
    "Curriculum_Adaptation": np.random.choice(curriculum, n_rows),
}

df = pd.DataFrame(data)

# --- Survival / Transition Logic ---
# Approx. Maharashtra UDISE→AISHE gap ~90% dropout (10% survival)
# We'll apply lower survival for cognitive/mental disabilities

high_dropout = ["Intellectual Disability", "Mental illness", "Cerebral palsy"]
medium_dropout = ["Hearing impairment", "Autism Spectrum Disorder", "Speech and Language"]

df["Survival_to_Class10"] = np.where(
    df["Disability_Type"].isin(high_dropout),
    np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
    np.where(
        df["Disability_Type"].isin(medium_dropout),
        np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
        np.random.choice([0, 1], size=n_rows, p=[0.4, 0.6])
    )
)

df["Transition_to_Class12"] = df["Survival_to_Class10"] * np.random.choice([0, 1], size=n_rows, p=[0.5, 0.5])

df["Employment_Prospect"] = np.where(
    (df["Parental_Involvement"] == "High") | (df["Socioeconomic_Status"] == "High"),
    np.random.choice([0, 1], size=n_rows, p=[0.3, 0.7]),
    np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4])
)

# --- Add Null Values (5–10% in random non-critical cols) ---
for col in ["Region", "Assistive_Tech_Use", "Parental_Involvement", "Curriculum_Adaptation"]:
    df.loc[df.sample(frac=0.08).index, col] = np.nan

# --- Save ---
df.to_csv("Maharashtra_Disability_Synthetic_Balanced.csv", index=False)

print("✅ Synthetic balanced dataset created with ~400 rows and nulls added")
print(df.head(15))
print("\nDistribution check:\n", df["Disability_Type"].value_counts())
