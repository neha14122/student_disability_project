#graphs showing disability
#"C:/Msc/Semester_3_2025_2026/Research_project/220925/Disability_progression_analysis.csv"
#"C:/Msc/Semester_3_2025_2026/Research_project/220925/Statewise_Enrolment_Drop_Cleaned1.csv"
#"C:/Msc/Semester_3_2025_2026/Research_project/220925/Maharashtra_Disability_Synthetic_Balanced1.csv"

# results_pipeline_final_colorful.py
# results_pipeline_threshold_colorful.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1️⃣ Load Datasets
# -----------------------------
udise_aishe = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Statewise_Enrolment_Drop_Cleaned1.csv")
progression = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Disability_progression_analysis.csv")
primary = pd.read_csv("C:/Msc/Semester_3_2025_2026/Research_project/220925/Maharashtra_Disability_Synthetic_Balanced1.csv")

# Clean column names
udise_aishe.columns = udise_aishe.columns.str.strip()
progression.columns = progression.columns.str.strip()
primary.columns = primary.columns.str.strip()

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
udise_col = next((c for c in udise_aishe.columns if "udise" in c.lower() and "total" in c.lower()), None)
aishe_col = next((c for c in udise_aishe.columns if "aishe" in c.lower() and "total" in c.lower()), None)

if udise_col and aishe_col:
    udise_aishe[udise_col] = udise_aishe[udise_col].replace(0, np.nan)
    udise_aishe["Gap_Risk_Index"] = (udise_aishe[udise_col] - udise_aishe[aishe_col]) / udise_aishe[udise_col]
else:
    raise ValueError(f"Could not find UDISE/AISHE total columns. Found: {udise_aishe.columns.tolist()}")

maha_gap = udise_aishe[udise_aishe["State"].str.contains("Maharashtra", case=False)]
gap_index = maha_gap["Gap_Risk_Index"].values[0] if not maha_gap.empty else np.nan

merge_cols = ["Disability", "Survival_to_Class10", "Transition_to_Class12", "Overall_Survival"]
available_cols = [c for c in merge_cols if c in progression.columns]
primary = primary.merge(
    progression[available_cols],
    how="left",
    left_on="Disability_Type",
    right_on="Disability"
)

# Keep proper columns
for c in ["Survival_to_Class10_y", "Transition_to_Class12_y"]:
    if c in primary.columns:
        primary[c.replace("_y","")] = primary[c]

primary = primary.drop(columns=[col for col in primary.columns if col.endswith("_x") or col.endswith("_y") or col=="Disability"])
primary["Gap_Risk_Index"] = gap_index

# -----------------------------
# 3️⃣ Null Handling
# -----------------------------
for col in primary.select_dtypes(include=["float64", "int64"]).columns:
    primary[col] = primary[col].fillna(primary[col].median())

for col in primary.select_dtypes(include=["object"]).columns:
    primary[col] = primary[col].fillna("Unknown")

numeric_targets = ["Survival_to_Class10", "Transition_to_Class12", "Employment_Prospect", "Overall_Survival"]
for col in numeric_targets:
    if col in primary.columns:
        primary[col] = pd.to_numeric(primary[col], errors='coerce').fillna(primary[col].median())

# -----------------------------
# 4️⃣ Encode categorical
# -----------------------------
categorical_cols = primary.select_dtypes(include=["object"]).columns.tolist()
primary_encoded = pd.get_dummies(primary, columns=categorical_cols, drop_first=True)

# -----------------------------
# 5️⃣ Early Intervention: create binary target
# -----------------------------
threshold = primary["Survival_to_Class10"].median()
primary["Survival_Class"] = (primary["Survival_to_Class10"] > threshold).astype(int)
y1 = primary["Survival_Class"]
X1 = primary_encoded.drop(columns=["Survival_to_Class10", "Employment_Prospect"])

# Employment Prediction target
y2 = primary_encoded["Employment_Prospect"]
X2 = primary_encoded.drop(columns=["Employment_Prospect"])

# -----------------------------
# 6️⃣ Training Helper with SMOTE
# -----------------------------
def train_model_balanced(X, y, task_name):
    if y.nunique() < 2:
        print(f"⚠️ Cannot train {task_name}: only 1 class present.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Balance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train_bal, y_train_bal)

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print(f"\n📌 {task_name} Results")
    print(classification_report(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
    except:
        pass

    return rf, report, importances

# -----------------------------
# 7️⃣ Train Models
# -----------------------------
rf1, report1, imp1 = train_model_balanced(X1, y1, "Early Intervention (Dropout Risk)")
rf2, report2, imp2 = train_model_balanced(X2, y2, "Employment Prediction")

# -----------------------------
# 8️⃣ Analysis Tables
# -----------------------------
survival_summary = primary.groupby("Disability_Type")[["Survival_to_Class10", "Overall_Survival"]].mean().sort_values("Overall_Survival")
employment_summary = primary.groupby("Disability_Type")["Employment_Prospect"].mean().sort_values()

# -----------------------------
# 9️⃣ Colorful Visualizations
# -----------------------------
palette_survival = sns.color_palette("tab10", n_colors=len(survival_summary))
palette_employment = sns.color_palette("tab10", n_colors=len(employment_summary))

plt.figure(figsize=(10,6))
sns.barplot(x=survival_summary.index, y=survival_summary["Overall_Survival"], palette=palette_survival)
plt.xticks(rotation=75)
plt.title("Average Survival to Class 12 by Disability")
plt.ylabel("Survival Rate")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=employment_summary.index, y=employment_summary, palette=palette_employment)
plt.xticks(rotation=75)
plt.title("Employment Prospects by Disability")
plt.ylabel("Employment Probability")
plt.show()

# -----------------------------
# 10️⃣ Color-coded Feature Importance
# -----------------------------
demographic_features = [c for c in X2.columns if "Gender" in c or "Region" in c or "Socioeconomic_Status" in c]
intervention_features = [c for c in X2.columns if "Assistive" in c or "Specialized" in c or "Parental" in c or "Curriculum" in c]
gap_features = [c for c in X2.columns if "Gap_Risk_Index" in c]

colors = []
for f in imp2.index[:15]:
    if f in demographic_features:
        colors.append("skyblue")
    elif f in intervention_features:
        colors.append("lightgreen")
    elif f in gap_features:
        colors.append("salmon")
    else:
        colors.append("lightgrey")

plt.figure(figsize=(10,6))
imp2.head(15).plot(kind="barh", color=colors)
plt.title("Top 15 Features Driving Employment Prediction (Color-coded)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()
