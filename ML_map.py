#C:/Msc/Semester_3_2025_2026/Research_project/220925/Statewise_Enrolment_Drop_Cleaned1.csv
#C:/Msc/Semester_3_2025_2026/Research_project/220925/Disability_progression_analysis.csv
#C:/Msc/Semester_3_2025_2026/Research_project/220925/Primary_dataset.csv

import pandas as pd
import numpy as np

# -----------------------
# 1. Load datasets
# -----------------------
udise_aishe = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Statewise_Enrolment_Drop_Cleaned1.csv")
secondary = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Disability_progression_analysis.csv")
primary = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Primary_dataset.csv")

# -----------------------
# 2. Clean column names
# -----------------------
primary.columns = primary.columns.str.strip()

# -----------------------
# 3. Preprocess UDISE + AISHE (macro dropout trends)
# -----------------------
dropout_dict = {}
for _, row in udise_aishe.iterrows():
    dropout_dict[(row['State'], 'Male')] = row['Male_Drop_Percent']
    dropout_dict[(row['State'], 'Female')] = row['Female_Drop_Percent']
    dropout_dict[(row['State'], 'Total')] = row['Total_Drop_Percent']

# -----------------------
# 4. Preprocess Secondary Disability data (risk scores)
# -----------------------
risk_before_dict = {}
risk_after_dict = {}
for _, row in secondary.iterrows():
    risk_before_dict[(row['Disability'], row['Gender'])] = row['Risk_Before_Class10']
    risk_after_dict[(row['Disability'], row['Gender'])] = row['Risk_After_Class10']

# -----------------------
# 5. Prepare Primary Institution dataset
# -----------------------
df = primary.copy()

# Dropout percent mapping
def get_dropout_percent(row):
    gender = row.get('Gender', 'Total')  # add Gender column if not present
    state = row.get('State', 'Maharashtra')  # default to Maharashtra
    return dropout_dict.get((state, gender), np.nan)

df['Dropout_Percent'] = df.apply(get_dropout_percent, axis=1)

# Risk score mapping with safe missing value handling
def get_risk_before(row):
    value = row.get('What type of disabilities do your students have? (Select all that apply)', np.nan)
    if pd.isna(value):
        return 0
    disability_list = [d.strip() for d in value.split(',')]
    risks = [risk_before_dict.get((d, row.get('Gender','Total')), 0) for d in disability_list]
    return max(risks) if risks else 0

def get_risk_after(row):
    value = row.get('What type of disabilities do your students have? (Select all that apply)', np.nan)
    if pd.isna(value):
        return 0
    disability_list = [d.strip() for d in value.split(',')]
    risks = [risk_after_dict.get((d, row.get('Gender','Total')), 0) for d in disability_list]
    return max(risks) if risks else 0

df['Risk_Before_Class10'] = df.apply(get_risk_before, axis=1)
df['Risk_After_Class10'] = df.apply(get_risk_after, axis=1)

# -----------------------
# 6. One-hot encode disabilities safely
# -----------------------
disabilities = secondary['Disability'].unique().tolist()
for d in disabilities:
    df[f'Disability_{d}'] = df.get('What type of disabilities do your students have? (Select all that apply)', '').apply(
        lambda x: 1 if pd.notna(x) and d in x else 0
    )

# -----------------------
# 7. Encode Yes/No columns safely
# -----------------------
yes_no_columns = [
    "Do parents provide additional learning support at home?",
    "Do students receive emotional or psychological support (counseling, therapy, etc.)?"
]

for col in yes_no_columns:
    if col in df.columns:
        df[col] = df[col].map({'Yes':1, 'No':0})
    else:
        print(f"Warning: Column '{col}' not found in dataset. Skipping.")

# -----------------------
# 8. Select features for ML
# -----------------------
feature_cols = ['Dropout_Percent', 'Risk_Before_Class10', 'Risk_After_Class10'] + \
               [f'Disability_{d}' for d in disabilities] + \
               [col for col in yes_no_columns if col in df.columns]

ml_df = df[feature_cols + ['sentiment_label']]  # sentiment_label as target

# -----------------------
# 9. Check the final dataset
# -----------------------
print("ML-ready dataset shape:", ml_df.shape)
print(ml_df.head())

# Copy previous ML-ready df
ml_df2 = df.copy()

# Function to map survival and transition scores from secondary dataset
def map_secondary_feature(row, feature):
    value = row.get('What type of disabilities do your students have? (Select all that apply)', np.nan)
    if pd.isna(value):
        return 0
    disability_list = [d.strip() for d in value.split(',')]
    # Take max value across multiple disabilities
    feature_values = [secondary.loc[
        (secondary['Disability'] == d) & 
        (secondary['Gender'] == row.get('Gender','Total')), feature
    ].values[0] if len(secondary.loc[
        (secondary['Disability'] == d) & 
        (secondary['Gender'] == row.get('Gender','Total')), feature
    ].values) > 0 else 0 for d in disability_list]
    return max(feature_values) if feature_values else 0

# Add new features
new_features = ['Survival_to_Class10', 'Transition_to_Class12', 'Overall_Survival']
for f in new_features:
    ml_df2[f] = ml_df2.apply(lambda row: map_secondary_feature(row, f), axis=1)

# Check the updated dataset
print("Enhanced ML-ready dataset shape:", ml_df2.shape)
print(ml_df2.head())
