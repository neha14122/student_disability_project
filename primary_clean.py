import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load your dataset
file_path = "C:/Msc/Semester_3_2025_2026/Research_project/220925/Primary_dataset.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 1. Rename columns for simplicity
# -----------------------------
column_mapping = {
    "What is your age group? ": "age_group",
    "How many years of experience do you have teaching disabled children? ": "experience_years",
    "What type of disabilities do your students have? (Select all that apply) ": "disability_types",
    "How many students with disabilities do you currently teach? ": "num_disabled_students",
    "What age group do you primarily teach? ": "primary_age_group",
    "Do you receive any specialized training to support disabled students? ": "specialized_training",
    "What percentage of your disabled students can keep up with the standard curriculum? ": "curriculum_percentage",
    "What additional learning support do these students require? (Select all that apply)": "learning_support",
    "Which skills are most difficult for disabled students to develop? (Select up to 3)": "difficult_skills",
    "How often do students engage in hands-on learning activities (projects, experiments, etc.)? ": "hands_on_learning",
    "Do students struggle with social interactions and teamwork? ": "social_difficulties",
    "What social challenges do they face the most? (Select all that apply)": "social_challenges",
    "Do students receive emotional or psychological support (counseling, therapy, etc.)? ": "psychological_support",
    "Are students introduced to basic work-related skills (teamwork, time management, responsibility, etc.)? ": "work_skills",
    "Do students get opportunities to explore careers (career days, field trips, guest speakers, etc.)? ": "career_exploration",
    "How many students show an interest in future jobs or careers? ": "career_interest",
    "Do students participate in practical life skills training (e.g., financial literacy, independent living skills, basic job tasks)? ": "life_skills_training",
    "Are students introduced to career-related topics (e.g., \"What do you want to be when you grow up?\")? ": "career_topics",
    "Do students show curiosity about future careers? ": "career_curiosity",
    "What career-related activities do students participate in? (Select all that apply) ": "career_activities",
    "Do students have opportunities to develop basic life skills (e.g., managing money, independent living, transportation)? ": "basic_life_skills",
    "Do students use assistive technology in learning? ": "assistive_tech_use",
    "Which types of assistive technologies are used in your classroom? (Select all that apply) ": "assistive_tech_types",
    "Do students feel comfortable using technology to learn? ": "tech_comfort",
    "How involved are parents in their child's education? ": "parental_involvement",
    "Do parents provide additional learning support at home? ": "home_learning_support",
    "Do students from more involved families perform better academically? ": "family_involvement_academic",
    "What are the biggest challenges these students might face in their future careers? (Select up to 3) ": "career_challenges",
    "What additional support do you believe would help disabled students prepare for future work opportunities? (Select up to 3) ": "work_support_needed",
    "What are the biggest challenges these students might face in the future? (Select up to 3) ": "future_challenges",
    "What additional support would help disabled students transition into adulthood successfully? (Select up to 3) ": "adulthood_support",
    "Is there any existing help/policy provided by the government?": "government_support",
    "Additional comment": "additional_comment",
    "sentiment_label": "sentiment_label"
}

df.rename(columns=column_mapping, inplace=True)

# -----------------------------
# 2. Drop unnecessary columns
# -----------------------------
drop_cols = ['Timestamp', 'What is your name']  # keep 'Additional comment'
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# -----------------------------
# 3. Sentiment Analysis on Additional Comment
# -----------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['additional_comment'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)

# -----------------------------
# 4. Save clean dataset
# -----------------------------
clean_file_path = "C:/Msc/Semester_3_2025_2026/Research_project/220925/Primary_cleaned.csv"
df.to_csv(clean_file_path, index=False)

print(f"Cleaned dataset saved at: {clean_file_path}")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
