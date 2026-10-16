
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Predictive Bank Customer Churn")
st.write(
    "Predict customer churn risk, explore probabilities, "
    "review feature importance, and simulate what-if scenarios."
)

# =========================================================
# 2. FEATURE SCHEMA FROM THE TRAINING NOTEBOOK
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent / "churn_model.pkl"
DATASET_YEAR = 2025

FEATURES = [
    "Year",
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
]


# =========================================================
# 3. LOAD MODEL BEFORE ANY REFERENCE TO model
# =========================================================
@st.cache_resource
def load_model(path):
    loaded_model = joblib.load(path)

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba()."
        )

    return loaded_model


if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Save the fitted Gradient Boosting model as "
        "'churn_model.pkl' in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Unable to load churn_model.pkl.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.warning(
        "Resolve the model-file or Python/scikit-learn compatibility "
        "problem before making predictions. The application has "
        "stopped safely because no model was loaded."
    )
    st.stop()

# model is now loaded. The following checks cannot run before loading.
if not hasattr(model, "n_features_in_"):
    st.error("The loaded model does not expose its expected feature count.")
    st.stop()

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = FEATURES.copy()

if len(model_features) != model.n_features_in_:
    st.error("The model's feature metadata is inconsistent.")
    st.stop()

if model_features != FEATURES:
    st.error(
        "The model's feature names or order do not match this app.\n\n"
        f"Expected by app: {FEATURES}\n\n"
        f"Found in model: {model_features}"
    )
    st.stop()

if 1 not in list(model.classes_):
    st.error("The model does not contain churn class 1 (Exited = 1).")
    st.stop()

st.sidebar.success("Gradient Boosting model loaded successfully.")

# =========================================================
# 4. CUSTOMER INPUTS
# =========================================================
st.sidebar.header("Customer Details")

credit_score = st.sidebar.slider(
    "Credit Score", 350, 850, 650
)

age = st.sidebar.slider(
    "Age", 18, 92, 40
)

tenure = st.sidebar.slider(
    "Tenure (years)", 0, 10, 5
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
)

num_products = st.sidebar.selectbox(
    "Number of Products", [1, 2, 3, 4], index=1
)

has_card = st.sidebar.selectbox(
    "Has Credit Card?", ["Yes", "No"]
)

is_active = st.sidebar.selectbox(
    "Active Member?", ["Yes", "No"]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=60000.0,
    step=1000.0,
)

geography = st.sidebar.selectbox(
    "Geography", ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender", ["Female", "Male"]
)

# =========================================================
# 5. FEATURE ENGINEERING
# Matches the formulas in the uploaded notebook.
# =========================================================
def build_customer_features(
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    salary,
    geography,
    gender,
):
    card_value = int(has_card == "Yes")
    active_value = int(is_active == "Yes")

    row = {
        "Year": DATASET_YEAR,
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": card_value,
        "IsActiveMember": active_value,
        "EstimatedSalary": salary,
        "Geography_Germany": int(geography == "Germany"),
        "Geography_Spain": int(geography == "Spain"),
        "Gender_Male": int(gender == "Male"),

        # Engineered features
        "BalanceSalaryRatio": balance / (salary + 1),
        "ProductDensity": num_products / (tenure + 1),
        "EngagementProductInteraction": active_value * num_products,
        "AgeTenureInteraction": age * tenure,
    }

    # Exact feature names and order expected by the trained model.
    return pd.DataFrame([row], columns=model_features)


# =========================================================
# 6. PREDICTION FUNCTION
# =========================================================
def predict_churn_probability(input_df):
    if list(input_df.columns) != model_features:
        raise ValueError("Prediction input columns do not match the model.")

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    return float(probabilities[classes.index(1)])


def get_risk_category(probability):
    # Illustrative thresholds; not calibrated bank policy.
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


# =========================================================
# 7. CURRENT CUSTOMER PREDICTION
# =========================================================
customer_df = build_customer_features(
    credit_score=credit_score,
    age=age,
    tenure=tenure,
    balance=balance,
    num_products=num_products,
    has_card=has_card,
    is_active=is_active,
    salary=salary,
    geography=geography,
    gender=gender,
)

try:
    current_probability = predict_churn_probability(customer_df)
except Exception as exc:
    st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
    st.stop()

retention_probability = 1.0 - current_probability
risk = get_risk_category(current_probability)

# =========================================================
# 8. CUSTOMER CHURN RISK CALCULATOR
# =========================================================
st.header("1. Customer Churn Risk Calculator")

col1, col2, col3 = st.columns(3)

col1.metric("Churn Probability", f"{current_probability:.1%}")
col2.metric("Retention Probability", f"{retention_probability:.1%}")
col3.metric("Churn Risk Category", risk)

st.progress(current_probability)

if risk == "High Risk":
    st.error("High predicted churn risk according to this model.")
elif risk == "Medium Risk":
    st.warning("Medium predicted churn risk according to this model.")
else:
    st.success("Low predicted churn risk according to this model.")

st.caption(
    "Risk categories use illustrative thresholds. "
    "They are not calibrated business policy."
)

# =========================================================
# 9. PROBABILITY DISTRIBUTION
# =========================================================
st.header("2. Probability Distribution")

probability_df = pd.DataFrame(
    {
        "Outcome": ["Churn (Exited = 1)", "Retained (Exited = 0)"],
        "Probability": [current_probability, retention_probability],
    }
)

st.bar_chart(probability_df.set_index("Outcome"))

with st.expander("View engineered features used for this customer"):
    st.dataframe(
        customer_df.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

# =========================================================
# 10. FEATURE IMPORTANCE DASHBOARD
# =========================================================
st.header("3. Feature Importance Dashboard")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame(
        {
            "Feature": model_features,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance_df.head(15).set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "This model does not expose feature_importances_. "
        "Permutation importance requires a representative evaluation dataset."
    )



# =========================================================
# 4. WHAT-IF SCENARIO SIMULATOR
# =========================================================

st.header("4. What-If Scenario Simulator")

st.write(
    "Change customer engagement and product count to compare "
    "the predicted churn probability with the original customer."
)

whatif_col1, whatif_col2 = st.columns(2)

with whatif_col1:
    scenario_active = st.selectbox(
        "Scenario: Is Active Member?",
        options=["Yes", "No"],
        index=0,  # Default: Yes
        key="scenario_active"
    )

with whatif_col2:
    scenario_products = st.selectbox(
        "Scenario: Number of Products",
        options=[1, 2, 3, 4],
        index=3,  # Default: 4 products
        key="scenario_products"
    )

if st.button("Run What-If Simulation", type="primary"):

    # Build scenario features using the same feature-engineering
    # function used by the churn calculator.
    scenario_df = build_customer_features(
        credit_score=credit_score,
        age=age,
        tenure=tenure,
        balance=balance,
        num_products=scenario_products,
        has_card=has_card,
        is_active=scenario_active,
        salary=salary,
        geography=geography,
        gender=gender
    )

    try:
        scenario_probability = predict_churn_probability(scenario_df)

        difference = scenario_probability - current_probability

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Original Churn Probability",
            f"{current_probability:.2%}"
        )

        result2.metric(
            "Scenario Churn Probability",
            f"{scenario_probability:.2%}"
        )

        result3.metric(
            "Change in Probability",
            f"{difference * 100:+.2f} percentage points"
        )

        # Compare the two predictions
        comparison_df = pd.DataFrame({
            "Scenario": [
                "Original Customer",
                "What-If Scenario"
            ],
            "Churn Probability (%)": [
                current_probability * 100,
                scenario_probability * 100
            ]
        })

        st.subheader("Original vs Scenario Comparison")

        st.bar_chart(
            comparison_df.set_index("Scenario")
        )

        st.caption(
            "The scenario is a model prediction, not a guarantee "
            "that changing these attributes will cause churn."
        )

    except Exception as exc:
        st.error(
            f"Scenario prediction failed: "
            f"{type(exc).__name__}: {exc}"
        )


# =========================================================
# 12. FOOTER
# =========================================================
st.divider()
st.caption(
    "Predictive decision support only. Validate model performance, "
    "probability calibration, feature definitions, and business "
    "thresholds before using predictions for customer actions."
)
