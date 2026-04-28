import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained models
# ----------------------------
rf_dropout = joblib.load("rf_dropout_model.pkl")
hist_employment = joblib.load("hist_employment_model.pkl")

# ----------------------------
# Custom option sets for key categories
# ----------------------------
custom_categories = {
    "Gender": ["Boys", "Girls"],
    "Region": ["Urban", "Rural", "Unknown"],
    "Socioeconomic_Status": ["Low", "Medium", "High"],
    "Assistive_Tech_Use": ["Yes", "No", "Unknown"],
    "Transition_to_Class12": ["Likely", "Unlikely"],
}

# ----------------------------
# Helper function to detect categorical prefixes
# ----------------------------
def get_categorical_features(feature_names):
    cat_features = {}
    for col in feature_names:
        if "_" in col and not any(x in col for x in ["Survival", "Overall"]):
            prefix, value = col.rsplit("_", 1)

            if prefix in custom_categories:
                cat_features[prefix] = custom_categories[prefix]
            else:
                if prefix not in cat_features:
                    cat_features[prefix] = []
                if value not in cat_features[prefix]:
                    cat_features[prefix].append(value)
    return cat_features


# ----------------------------
# Get feature info for both models
# ----------------------------
dropout_features = rf_dropout.feature_names_in_
employment_features = hist_employment.feature_names_in_

dropout_cat = get_categorical_features(dropout_features)
employment_cat = get_categorical_features(employment_features)

all_cat_features = {**dropout_cat, **employment_cat}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Disability Student Prediction", layout="centered")

st.title("🎓 Disability Student Prediction & Recommendation System")
st.markdown("---")
st.header("📝 Student Information")

user_input = {}

# Dynamically create input widgets
for prefix, options in all_cat_features.items():
    label = prefix.replace("_", " ")
    user_input[prefix] = st.selectbox(f"**{label}:**", options)

# ----------------------------
# Function to create input DataFrame
# ----------------------------
def create_input_df(user_input, feature_names):
    input_dict = {col: 0 for col in feature_names}

    for prefix, value in user_input.items():
        col_name = f"{prefix}_{value}"
        if col_name in input_dict:
            input_dict[col_name] = 1

    return pd.DataFrame([input_dict], columns=feature_names)


# ----------------------------
# Prepare inputs for both models
# ----------------------------
input_dropout_df = create_input_df(user_input, dropout_features)
input_employment_df = create_input_df(user_input, employment_features)

# ----------------------------
# Predictions
# ----------------------------
dropout_prob = rf_dropout.predict_proba(input_dropout_df)[:, 1][0]
employment_prob = hist_employment.predict_proba(input_employment_df)[:, 1][0]

# ----------------------------
# Display results
# ----------------------------
st.markdown("---")
st.header("📊 Predictions")

st.subheader("Early Intervention (Dropout Risk)")
st.progress(int(dropout_prob * 100))
st.write(f"**Probability:** {dropout_prob:.2f}")

if dropout_prob > 0.5:
    st.warning("⚠️ High risk of dropout! Early intervention recommended.")
else:
    st.success("✅ Low risk of dropout.")

st.subheader("Employment Prospects")
st.progress(int(employment_prob * 100))
st.write(f"**Probability:** {employment_prob:.2f}")

if employment_prob > 0.5:
    st.success("💼 Good employment prospects after education.")
else:
    st.warning("⚠️ Employment support may be needed.")

# ----------------------------
# Recommendations
# ----------------------------
st.markdown("---")
st.header("💡 Recommendations")

if dropout_prob > 0.5:
    st.markdown(
        """
        - Provide **additional mentoring and counseling**.  
        - Encourage **parental involvement**.  
        - Offer **specialized training and assistive technology support**.  
        - Adapt **curriculum** to student’s needs.  
        """
    )
else:
    st.markdown(" Maintain current support system and monitor progress.")

if employment_prob < 0.5:
    st.markdown(
        """
        - Provide **career guidance and vocational training**.  
        - Connect with **employment support organizations**.  
        - Encourage **skill-building workshops and internships**.  
        """
    )
