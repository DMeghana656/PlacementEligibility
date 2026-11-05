
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. Custom dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF2;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #163A70, #2866A8);
        padding: 30px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #E0ECFF;
        font-size: 1rem;
        margin: 0;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 15px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 3px 12px rgba(30, 55, 90, 0.04);
    }

    .metric-label {
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .metric-value {
        color: #172B4D;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .metric-note {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    .section-title {
        color: #172B4D;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 12px 0 4px 0;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 0.92rem;
        margin-bottom: 18px;
    }

    .result-card {
        padding: 22px;
        border-radius: 15px;
        margin-top: 16px;
        margin-bottom: 12px;
    }

    .result-eligible {
        background: #E8F7EE;
        border: 1px solid #B8E5C8;
        color: #17663A;
    }

    .result-not-eligible {
        background: #FFF0ED;
        border: 1px solid #F5C7BE;
        color: #A33729;
    }

    .result-card h2 {
        margin: 0 0 8px 0;
        font-size: 1.5rem;
    }

    .result-card p {
        margin: 0;
        font-size: 0.92rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 9px;
        min-height: 44px;
        font-weight: 600;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 15px;
        padding: 22px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        padding: 14px;
        border-radius: 12px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Load the student dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("students.csv")

    required_columns = [
        "RollNumber",
        "StudentName",
        "CGPA",
        "Attendance",
        "SkillScore",
        "Eligible",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in students.csv: {missing}"
        )

    df = df.dropna(subset=required_columns).copy()

    df["RollNumber"] = df["RollNumber"].astype(str).str.strip()
    df["StudentName"] = df["StudentName"].astype(str).str.strip()

    for col in ["CGPA", "Attendance", "SkillScore"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["Eligible"] = (
        df["Eligible"].astype(str).str.strip().str.title()
    )

    if not df["Eligible"].isin(["Yes", "No"]).all():
        raise ValueError(
            "Eligible column must contain only Yes or No."
        )

    if df.empty:
        raise ValueError("The dataset has no valid student records.")

    return df


try:
    data = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info(
        "Make sure students.csv is in the same folder as app.py "
        "and contains the required columns."
    )
    st.stop()

# --------------------------------------------------
# Placement eligibility rules requested for this project
# Eligible when CGPA >= 6.5 AND Attendance >= 45 AND SkillScore >= 60.
# --------------------------------------------------
data["Eligible"] = (
    (data["CGPA"] >= 6.5)
    & (data["Attendance"] >= 45)
    & (data["SkillScore"] >= 60)
).map({True: "Yes", False: "No"})

# --------------------------------------------------
# 4. Train the decision tree
# --------------------------------------------------
FEATURES = ["CGPA", "Attendance", "SkillScore"]
TARGET = "Eligible"

# Train on boundary examples so the tree represents all three eligibility thresholds.
rule_rows = []
for cgpa_value in [0.0, 6.4, 6.5, 6.6, 10.0]:
    for attendance_value in [0, 44, 45, 100]:
        for skill_value in [0, 59, 60, 100]:
            eligible = cgpa_value >= 6.5 and attendance_value >= 45 and skill_value >= 60
            rule_rows.append(
                [cgpa_value, attendance_value, skill_value,
                 "Yes" if eligible else "No"]
            )

rule_training_data = pd.DataFrame(
    rule_rows,
    columns=FEATURES + [TARGET],
)
X = rule_training_data[FEATURES]
y = rule_training_data[TARGET]

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42,
)

model.fit(X, y)

eligible_count = int((data[TARGET] == "Yes").sum())
not_eligible_count = int((data[TARGET] == "No").sum())
total_students = len(data)
# --------------------------------------------------
# Prediction history stored in the current session
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Placement AI")
    st.caption("Student eligibility dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Student Dataset", "Decision Tree"],
    )

    st.divider()

    st.markdown("### Model information")
    st.write("**Algorithm:** Decision Tree")
    st.write("**Criterion:** Entropy")
    st.write("**Maximum depth:** 3")
    st.write("**Input features:** 3")

    st.info(
        "This project uses a small demonstration dataset. "
        "Predictions are educational estimates, not official "
        "placement decisions."
    )

# --------------------------------------------------
# 6. Dashboard header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Placement Predictor</h1>
        <p>
            Predict eligibility using CGPA >= 6.5, Attendance >= 45, and
            Skill Score >= 60. Enter student details and explore the result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 7. Dashboard page
# --------------------------------------------------
if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Placement Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'A quick summary of the students in your dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)

    metrics = [
        (
            "👥 Total Students",
            total_students,
            "Records in dataset",
        ),
        (
            "✅ Eligible",
            eligible_count,
            f"{eligible_count / total_students:.1%} of records",
        ),
        (
            "📋 Not Eligible",
            not_eligible_count,
            f"{not_eligible_count / total_students:.1%} of records",
        ),
    ]

    for col, (label, value, note) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Keep the prediction form full-width; dataset snapshot and graph removed.
    left_col = st.container()

    # ----------------------------------------------
    # Prediction form
    # ----------------------------------------------
    with left_col:
        st.markdown(
            '<div class="section-title">Predict Eligibility</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">'
            'Enter the student details below.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Eligibility requires all three: CGPA at least 6.5, "
            "Attendance at least 45%, and Skill Score at least 60."
        )

        with st.form("prediction_form"):
            roll_number = st.text_input(
                "Roll Number",
                placeholder="e.g. 25CS101",
            )

            student_name = st.text_input(
                "Student Name",
                placeholder="Enter student name",
            )

            cgpa = st.number_input(
                "CGPA (out of 10)",
                min_value=0.0,
                max_value=10.0,
                value=8.0,
                step=0.1,
                format="%.2f",
            )

            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=85,
            )

            skill_score = st.slider(
                "Skill Score",
                min_value=0,
                max_value=100,
                value=75,
            )

            submitted = st.form_submit_button(
                "✨ Predict Eligibility",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            roll_number = roll_number.strip()
            student_name = student_name.strip()

            if not roll_number or not student_name:
                st.warning(
                    "Please enter both the roll number and student name."
                )

            elif (
                data["RollNumber"].str.casefold()
                == roll_number.casefold()
            ).any():
                st.warning(
                    "This roll number already exists in the dataset. "
                    "Please enter a new roll number."
                )

            else:
                student_input = pd.DataFrame(
                    [[cgpa, attendance, skill_score]],
                    columns=FEATURES,
                )

                prediction = model.predict(student_input)[0]
                probabilities = model.predict_proba(student_input)[0]

                probability_by_class = dict(
                    zip(model.classes_, probabilities)
                )

                predicted_probability = float(
                    probability_by_class[prediction]
                )
                # Save this prediction to session history
                history_record = {
                    "Prediction Time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "RollNumber": roll_number,
                    "StudentName": student_name,
                    "CGPA": cgpa,
                    "Attendance": attendance,
                    "SkillScore": skill_score,
                    "Predicted Eligibility": (
                        "Eligible" if prediction == "Yes"
                        else "Not Eligible"
                    ),
                    "Eligible Probability (%)": round(
                        float(probability_by_class.get("Yes", 0.0)) * 100, 2
                    ),
                    "Not Eligible Probability (%)": round(
                        float(probability_by_class.get("No", 0.0)) * 100, 2
                    ),
                    "Predicted Class Probability (%)": round(
                        predicted_probability * 100, 2
                    ),
                }

                st.session_state.prediction_history.append(history_record)
                if prediction == "Yes":
                    st.markdown(
                        f"""
                        <div class="result-card result-eligible">
                            <h2>✅ Eligible for Placement</h2>
                            <p>
                                {student_name} ({roll_number}) is
                                predicted to be eligible based on
                                the decision tree.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card result-not-eligible">
                            <h2>⚠️ Not Eligible for Placement</h2>
                            <p>
                                {student_name} ({roll_number}) is
                                predicted not to be eligible based
                                on the decision tree.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Prediction details")

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.metric(
                        "Predicted class",
                        "Eligible" if prediction == "Yes"
                        else "Not Eligible",
                    )

                with prob_col2:
                    st.metric(
                        "Model probability for predicted class",
                        f"{predicted_probability:.1%}",
                    )

                st.caption(
                    "This probability reflects the fitted decision "
                    "tree's estimated class probabilities. It is not "
                    "a guarantee of actual placement."
                )

                st.markdown("#### Class probability breakdown")

                st.progress(
                    float(probability_by_class.get("Yes", 0.0)),
                    text=(
                        "Eligible: "
                        f"{probability_by_class.get('Yes', 0.0):.1%}"
                    ),
                )

                st.progress(
                    float(probability_by_class.get("No", 0.0)),
                    text=(
                        "Not eligible: "
                        f"{probability_by_class.get('No', 0.0):.1%}"
                    ),
                )

# --------------------------------------------------
# 8. Student dataset: view, add, edit, delete, save
# --------------------------------------------------
elif page == "Student Dataset":
    st.markdown(
        '<div class="section-title">Student Dataset</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Add new students, edit existing details, or delete rows. '
        'Save your changes to update the dataset and retrain the decision tree.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Edit student details directly in the table. Use the table controls to add "
        "or remove rows, then click Save Dataset Changes. Roll numbers must be unique. "
        "Eligibility is calculated automatically: CGPA >= 6.5 AND Attendance >= 45 AND Skill Score >= 60."
    )
    st.warning(
        "Saving updates students.csv permanently. Keep a backup before making major changes."
    )

    edited_data = st.data_editor(
        data.copy(),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        disabled=["Eligible"],
        key="editable_student_dataset",
        column_config={
            "RollNumber": st.column_config.TextColumn("Roll Number", required=True),
            "StudentName": st.column_config.TextColumn("Student Name", required=True),
            "CGPA": st.column_config.NumberColumn(
                "CGPA", min_value=0.0, max_value=10.0, step=0.1, format="%.2f", required=True
            ),
            "Attendance": st.column_config.NumberColumn(
                "Attendance", min_value=0, max_value=100, step=1, required=True
            ),
            "SkillScore": st.column_config.NumberColumn(
                "Skill Score", min_value=0, max_value=100, step=1, required=True
            ),
            "Eligible": st.column_config.SelectboxColumn(
                "Eligible", options=["Yes", "No"], required=True
            ),
        },
    )

    if st.button("💾 Save Dataset Changes", type="primary", use_container_width=True):
        required_columns = [
            "RollNumber", "StudentName", "CGPA", "Attendance", "SkillScore", "Eligible"
        ]
        candidate = edited_data.copy()

        if list(candidate.columns) != required_columns:
            st.error("Please keep the original columns unchanged.")
        else:
            candidate["RollNumber"] = candidate["RollNumber"].astype(str).str.strip()
            candidate["StudentName"] = candidate["StudentName"].astype(str).str.strip()
            for numeric_column in ["CGPA", "Attendance", "SkillScore"]:
                candidate[numeric_column] = pd.to_numeric(candidate[numeric_column], errors="coerce")
            # Calculate eligibility from all three requested rules; do not rely on manually entered labels.
            candidate["Eligible"] = (
                (candidate["CGPA"] >= 6.5) &
                (candidate["Attendance"] >= 45) &
                (candidate["SkillScore"] >= 60)
            ).map({True: "Yes", False: "No"})

            if candidate.empty:
                st.error("The dataset must contain at least one student record.")
            elif candidate[required_columns].isna().any().any():
                st.error("Fill in every field and ensure numeric values are valid.")
            elif (candidate["RollNumber"] == "").any() or (candidate["StudentName"] == "").any():
                st.error("Roll number and student name cannot be empty.")
            elif candidate["RollNumber"].duplicated().any():
                st.error("Each roll number must be unique.")
            elif not candidate["CGPA"].between(0, 10).all():
                st.error("CGPA must be between 0 and 10.")
            elif not candidate["Attendance"].between(0, 100).all():
                st.error("Attendance must be between 0 and 100.")
            elif not candidate["SkillScore"].between(0, 100).all():
                st.error("Skill score must be between 0 and 100.")
            elif not candidate["Eligible"].isin(["Yes", "No"]).all():
                st.error("Eligible must contain only Yes or No.")
            else:
                candidate.to_csv("students.csv", index=False)
                load_data.clear()
                st.session_state.pop("editable_student_dataset", None)
                st.success("Dataset saved. The model and decision tree will be refreshed.")
                st.rerun()

# --------------------------------------------------
# 9. Decision tree page
# --------------------------------------------------
elif page == "Decision Tree":

    st.markdown(
        '<div class="section-title">How the AI Makes Decisions</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Inspect the rules learned by the decision tree model.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This tree is trained on examples around your specified thresholds, so it "
        "shows the rules CGPA >= 6.5, Attendance >= 45, and SkillScore >= 60. "
        "Student records are still saved in students.csv."
    )

    fig, ax = plt.subplots(figsize=(18, 9))

    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=9,
        ax=ax,
    )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("View model configuration"):
        st.write(
            {
                "Algorithm": "DecisionTreeClassifier",
                "Criterion": model.criterion,
                "Maximum depth": model.max_depth,
                "Actual tree depth": model.get_depth(),
                "Number of leaves": model.get_n_leaves(),
                "Features": FEATURES,
                "Rule training examples": len(rule_training_data),
            }
        )


# --------------------------------------------------
# 12. Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Student Placement Eligibility Predictor · "
    "Built with Python, Streamlit, pandas, matplotlib, and scikit-learn"
)
