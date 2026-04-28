# ML_small_data_improved.py
# ML_small_data_pipeline_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1️⃣ Load datasets
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
# Gap Risk Index
udise_col = next((c for c in udise_aishe.columns if "udise" in c.lower() and "total" in c.lower()), None)
aishe_col = next((c for c in udise_aishe.columns if "aishe" in c.lower() and "total" in c.lower()), None)
udise_aishe["Gap_Risk_Index"] = (udise_aishe[udise_col] - udise_aishe[aishe_col]) / udise_aishe[udise_col]
maha_gap = udise_aishe[udise_aishe["State"].str.contains("Maharashtra", case=False)]
gap_index = maha_gap["Gap_Risk_Index"].values[0] if not maha_gap.empty else np.nan

# Merge progression data
merge_cols = ["Disability", "Survival_to_Class10", "Transition_to_Class12", "Overall_Survival"]
available_cols = [c for c in merge_cols if c in progression.columns]
primary = primary.merge(
    progression[available_cols],
    how="left",
    left_on="Disability_Type",
    right_on="Disability"
)

# Fix column names after merge
primary["Survival_to_Class10"] = primary["Survival_to_Class10_y"]
primary["Transition_to_Class12"] = primary["Transition_to_Class12_y"]

primary = primary.drop(columns=[
    "Survival_to_Class10_x", "Transition_to_Class12_x",
    "Survival_to_Class10_y", "Transition_to_Class12_y", "Disability"
])

# Add Gap_Risk_Index
primary["Gap_Risk_Index"] = gap_index

# Fill nulls
for col in primary.select_dtypes(include=["float64","int64"]).columns:
    primary[col] = primary[col].fillna(primary[col].median())
for col in primary.select_dtypes(include=["object"]).columns:
    primary[col] = primary[col].fillna("Unknown")

# -----------------------------
# 3️⃣ Create binary target for Early Intervention
# -----------------------------
threshold = primary["Survival_to_Class10"].median()
primary["Survival_Class"] = (primary["Survival_to_Class10"] > threshold).astype(int)

# -----------------------------
# 4️⃣ Interaction features
# -----------------------------
primary["Gap_ParentalInteraction"] = primary["Gap_Risk_Index"] * primary["Parental_Involvement"].map({"Low":1,"Medium":2,"High":3,"Unknown":0})
primary["Assistive_Training_Interaction"] = primary["Assistive_Tech_Use"].map({"Yes":1,"No":0,"Unknown":0}) * primary["Specialized_Training_Access"].map({"Workshop":1,"None":0,"Unknown":0})
primary["Gender_Region_Interaction"] = primary["Gender"].map({"Boys":1,"Girls":0}) * primary["Region"].map({"Urban":1,"Rural":0,"Unknown":0})

# -----------------------------
# 5️⃣ Encode categorical variables
# -----------------------------
categorical_cols = primary.select_dtypes(include=["object"]).columns.tolist()
primary_encoded = pd.get_dummies(primary, columns=categorical_cols, drop_first=True)

# -----------------------------
# 6️⃣ Define features and targets
# -----------------------------
y1 = primary["Survival_Class"]
X1 = primary_encoded.drop(columns=["Survival_to_Class10","Employment_Prospect","Survival_Class"])

y2 = primary_encoded["Employment_Prospect"]
X2 = primary_encoded.drop(columns=["Employment_Prospect"])

# -----------------------------
# 7️⃣ Training function with SMOTE + Gaussian noise + 5-fold CV
# -----------------------------
def train_model_small_data(X, y, task_name):
    if y.nunique() < 2:
        print(f"⚠️ Cannot train {task_name}: only 1 class present.")
        return None, None

    auc_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx].fillna(0), X.iloc[test_idx].fillna(0)
        y_train, y_test = y.iloc[train_idx].fillna(0), y.iloc[test_idx].fillna(0)

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        # Gaussian noise augmentation
        numeric_cols = X_train_bal.select_dtypes(include=["float64","int64"]).columns
        noise = np.random.normal(0, 0.05*X_train_bal[numeric_cols].std(), X_train_bal[numeric_cols].shape)
        X_train_bal[numeric_cols] += noise

        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                    class_weight="balanced", random_state=42)
        rf.fit(X_train_bal, y_train_bal)

        y_proba = rf.predict_proba(X_test)[:,1]
        auc_scores.append(roc_auc_score(y_test, y_proba))

    print(f"\n📌 {task_name} Mean ROC AUC (5-fold CV): {np.mean(auc_scores):.3f}")
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return rf, importances

# -----------------------------
# 8️⃣ Train models
# -----------------------------
rf1, imp1 = train_model_small_data(X1, y1, "Early Intervention (Dropout Risk)")
rf2, imp2 = train_model_small_data(X2, y2, "Employment Prediction")

# -----------------------------
# 9️⃣ Colorful feature importance plot
# -----------------------------
colors = ["skyblue" if "Gender" in f or "Region" in f else
          "lightgreen" if "Assistive" in f or "Specialized" in f or "Parental" in f or "Curriculum" in f else
          "salmon" if "Gap_Risk_Index" in f else "lightgrey"
          for f in imp2.index[:15]]

plt.figure(figsize=(10,6))
imp2.head(15).plot(kind="barh", color=colors)
plt.title("Top 15 Features Driving Employment Prediction (Color-coded)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

# -----------------------------
# 10️⃣ Colorful per-disability plots
# -----------------------------
# Early Intervention
survival_summary = primary.groupby("Disability_Type")[["Survival_to_Class10","Overall_Survival"]].mean()
plt.figure(figsize=(10,6))
sns.barplot(x=survival_summary.index, y=survival_summary["Overall_Survival"], palette="tab10")
plt.xticks(rotation=75)
plt.title("Average Survival to Class 12 by Disability")
plt.ylabel("Survival Rate")
plt.show()

# Employment Prospects
employment_summary = primary.groupby("Disability_Type")["Employment_Prospect"].mean()
plt.figure(figsize=(10,6))
sns.barplot(x=employment_summary.index, y=employment_summary, palette="tab10")
plt.xticks(rotation=75)
plt.title("Employment Prospects by Disability")
plt.ylabel("Employment Probability")
plt.show()

#Different Model test

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# ----------------------------
# Assume X1, y1 and X2, y2 are ready
# ----------------------------

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM": SVC(probability=True, kernel='rbf', class_weight='balanced'),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, random_state=42)
}

def evaluate_models(X, y, models_dict):
    # ----------------------------
    # Handle NaNs globally with SimpleImputer
    # ----------------------------
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns)
    
    auc_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models_dict.items():
        auc_scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # SMOTE for imbalanced small data
            smote = SMOTE(random_state=42)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

            model.fit(X_train_bal, y_train_bal)
            y_proba = model.predict_proba(X_test)[:,1]
            auc_scores.append(roc_auc_score(y_test, y_proba))

        auc_results[name] = np.mean(auc_scores)
        print(f"{name} Mean ROC AUC: {auc_results[name]:.3f}")
    
    return auc_results

# ----------------------------
# Evaluate Models
# ----------------------------
print("\n--- Early Intervention (Dropout Risk) ---")
auc_ei = evaluate_models(X1, y1, models)

print("\n--- Employment Prediction ---")
auc_emp = evaluate_models(X2, y2, models)

# ----------------------------
# Plot Comparison
# ----------------------------
plt.figure(figsize=(10,6))
bar_width = 0.35
x = np.arange(len(models))
plt.bar(x - bar_width/2, list(auc_ei.values()), width=bar_width, label="Early Intervention", color="skyblue")
plt.bar(x + bar_width/2, list(auc_emp.values()), width=bar_width, label="Employment Prediction", color="salmon")
plt.xticks(x, list(models.keys()), rotation=20)
plt.ylabel("Mean ROC AUC (5-fold CV)")
plt.title("ML Model Comparison on Small Dataset")
plt.ylim(0.5, 1.05)
plt.legend()
plt.show()
