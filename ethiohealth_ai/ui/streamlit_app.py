import streamlit as st

from ethiohealth_ai.ml import explain_prediction
from ethiohealth_ai.services import (
    build_ai_reason,
    build_input_dataframe,
    get_confidence,
    load_artifacts,
)


YES_NO_OPTIONS = ["No", "Yes"]
GENDER_OPTIONS = ["Male", "Female", "Other"]
MALARIA_OPTIONS = ["Negative", "Positive", "Unknown"]
SEASON_OPTIONS = ["Summer", "Autumn", "Winter", "Spring"]
REGION_OPTIONS = [
    "Addis Ababa",
    "Oromia",
    "Amhara",
    "Tigray",
    "Southern",
    "Afar",
    "Somali",
    "Benishangul",
    "Gambela",
    "Harari",
    "Dire Dawa",
]


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f6f8;
            --panel: #ffffff;
            --border: #d8e0e7;
            --text: #17212b;
            --muted: #5e6b78;
            --primary: #0f5f77;
            --primary-dark: #0a4658;
            --danger: #a33a3a;
            --warning: #95620f;
            --success: #2e6f4e;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div {
            color: var(--text);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            color: var(--text);
            letter-spacing: 0;
        }

        .dashboard-header {
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 6px solid var(--primary);
            border-radius: 6px;
            padding: 18px 22px;
            margin-bottom: 18px;
        }

        .dashboard-title {
            margin: 0;
            color: var(--text);
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .dashboard-subtitle {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .section-label {
            color: var(--primary-dark);
            font-weight: 700;
            font-size: 1rem;
            margin: 10px 0 8px;
        }

        .result-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px 18px;
            min-height: 142px;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .result-value {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            word-break: break-word;
        }

        .result-detail {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 10px;
        }

        .risk-low {
            border-left: 5px solid var(--success);
        }

        .risk-medium {
            border-left: 5px solid var(--warning);
        }

        .risk-high {
            border-left: 5px solid var(--danger);
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px 14px;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 44px;
            border-radius: 4px;
            border: 2px solid #083b4b !important;
            background-color: #0f5f77 !important;
            color: #ffffff !important;
            font-weight: 700;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: #062f3c !important;
            background-color: #0a4658 !important;
            color: #ffffff !important;
        }

        .stButton > button:focus,
        .stButton > button:active,
        .stFormSubmitButton > button:focus,
        .stFormSubmitButton > button:active {
            border-color: #031f29 !important;
            background-color: #083b4b !important;
            color: #ffffff !important;
            box-shadow: 0 0 0 3px rgba(15, 95, 119, 0.25) !important;
        }

        .stButton > button:disabled,
        .stFormSubmitButton > button:disabled {
            background-color: #b7c4cc !important;
            border-color: #94a3ad !important;
            color: #ffffff !important;
        }

        .stButton > button *,
        .stFormSubmitButton > button * {
            color: #ffffff !important;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: var(--text) !important;
        }

        [data-testid="stTabs"] button,
        [data-testid="stTabs"] button p {
            color: var(--text) !important;
            font-weight: 700;
        }

        [data-baseweb="tab-highlight"] {
            background-color: var(--primary) !important;
        }

        input,
        textarea,
        [data-baseweb="select"] *,
        [data-baseweb="input"] * {
            color: var(--text) !important;
        }

        input,
        textarea,
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            background-color: #ffffff !important;
            border-color: #9fb0bf !important;
            color: var(--text) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #7a8794 !important;
            opacity: 1 !important;
        }

        [data-baseweb="select"] svg,
        [data-baseweb="input"] svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }

        [role="listbox"],
        [role="option"],
        [data-baseweb="popover"] div {
            background-color: #ffffff !important;
            color: var(--text) !important;
        }

        [role="option"]:hover,
        [aria-selected="true"] {
            background-color: #e7f0f4 !important;
            color: var(--text) !important;
        }

        .stNumberInput button {
            background-color: #eef3f6 !important;
            border-color: #9fb0bf !important;
            color: var(--text) !important;
        }

        .stNumberInput button svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        [data-testid="stExpander"] {
            background-color: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] button {
            background-color: var(--panel) !important;
            padding: 10px 15px !important;
            border-bottom: 1px solid var(--border) !important;
            transition: background-color 0.2s ease;
        }

        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] button:hover {
            background-color: #f8f9fa !important;
        }

        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] button p {
            font-weight: 700 !important;
            color: var(--text) !important;
        }

        [data-testid="stExpander"] svg {
            fill: var(--text) !important;
            color: var(--text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="dashboard-title">Hospital Clinical Support Dashboard</h1>
            <p class="dashboard-subtitle">
                Patient intake, provisional disease prediction, risk assessment, and length-of-stay estimate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_inputs():
    with st.form("patient_assessment_form", clear_on_submit=False):
        st.markdown('<div class="section-label">Patient Demographics</div>', unsafe_allow_html=True)
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
        with d_col2:
            season = st.selectbox("Season", SEASON_OPTIONS)
        with d_col3:
            comorbidity = st.selectbox("Comorbidity", YES_NO_OPTIONS)

        st.markdown('<div class="section-label">Vital Signs & Labs</div>', unsafe_allow_html=True)
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, format="%.1f")
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
        with v_col2:
            wbc_count = st.number_input("WBC Count (/µL)", min_value=1000, max_value=50000, value=8000)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=22.0, value=13.0, format="%.1f")
        with v_col3:
            malaria_test = st.selectbox("Malaria Test", MALARIA_OPTIONS)

        st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            fever = st.selectbox("Fever", YES_NO_OPTIONS)
            cough = st.selectbox("Cough", YES_NO_OPTIONS)
            headache = st.selectbox("Headache", YES_NO_OPTIONS)
        with s_col2:
            fatigue = st.selectbox("Fatigue", YES_NO_OPTIONS)
            vomiting = st.selectbox("Vomiting", YES_NO_OPTIONS)
            diarrhea = st.selectbox("Diarrhea", YES_NO_OPTIONS)
        with s_col3:
            chest_pain = st.selectbox("Chest Pain", YES_NO_OPTIONS)
            shortness_of_breath = st.selectbox("Shortness of Breath", YES_NO_OPTIONS)
            dizziness = st.selectbox("Dizziness", YES_NO_OPTIONS)

        submitted = st.form_submit_button("Run Clinical Assessment")

    inputs = {
        "Age": age,
        "Gender": gender,
        "Fever": fever,
        "Cough": cough,
        "Headache": headache,
        "Fatigue": fatigue,
        "Vomiting": vomiting,
        "Diarrhea": diarrhea,
        "Chest Pain": chest_pain,
        "Shortness of Breath": shortness_of_breath,
        "Dizziness": dizziness,
        "Temperature": temperature,
        "Heart Rate": heart_rate,
        "WBC Count": wbc_count,
        "Hemoglobin": hemoglobin,
        "Malaria Test": malaria_test,
        "Comorbidity": comorbidity,
        "Season": season,
    }
    return inputs, submitted



def get_prediction(model, encoder, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
    return encoder.inverse_transform(prediction)[0], get_confidence(probability)


def render_result(label: str, value: str, detail: str, css_class: str = ""):
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="result-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_class(risk_label: str) -> str:
    risk = str(risk_label).strip().lower()
    if risk == "high":
        return "risk-high"
    if risk == "medium":
        return "risk-medium"
    return "risk-low"


def render_explanation(model, processed_input, feature_names):
    try:
        for line in explain_prediction(model, processed_input, feature_names, top_n=5):
            st.write(f"- {line}")
    except Exception as ex:
        st.warning(f"Unable to explain prediction: {ex}")


def render_patient_summary(input_df):
    inputs = input_df.iloc[0].to_dict()

    st.markdown('<div class="section-label">Patient Demographics</div>', unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.write(f"**Age:** {inputs.get('Age')}")
        st.write(f"**Gender:** {inputs.get('Gender')}")
    with d_col2:
        st.write(f"**Season:** {inputs.get('Season')}")
    with d_col3:
        st.write(f"**Comorbidity:** {inputs.get('Comorbidity')}")

    st.markdown('<div class="section-label">Vital Signs & Labs</div>', unsafe_allow_html=True)
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.write(f"**Temperature:** {inputs.get('Temperature')} °C")
        st.write(f"**Heart Rate:** {inputs.get('Heart Rate')} bpm")
    with v_col2:
        st.write(f"**WBC Count:** {inputs.get('WBC Count')} /µL")
        st.write(f"**Hemoglobin:** {inputs.get('Hemoglobin')} g/dL")
    with v_col3:
        st.write(f"**Malaria Test:** {inputs.get('Malaria Test')}")

    st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.write(f"**Fever:** {inputs.get('Fever')}")
        st.write(f"**Cough:** {inputs.get('Cough')}")
        st.write(f"**Headache:** {inputs.get('Headache')}")
    with s_col2:
        st.write(f"**Fatigue:** {inputs.get('Fatigue')}")
        st.write(f"**Vomiting:** {inputs.get('Vomiting')}")
        st.write(f"**Diarrhea:** {inputs.get('Diarrhea')}")
    with s_col3:
        st.write(f"**Chest Pain:** {inputs.get('Chest Pain')}")
        st.write(f"**Shortness of Breath:** {inputs.get('Shortness of Breath')}")
        st.write(f"**Dizziness:** {inputs.get('Dizziness')}")


def main():
    st.set_page_config(
        page_title="Hospital Clinical Support Dashboard",
        page_icon="H",
        layout="wide",
    )
    render_styles()
    render_header()

    try:
        artifacts = load_artifacts()
    except Exception as ex:
        st.error(f"Error loading model artifacts: {ex}")
        st.info("Retrain the models with the active Python environment if the saved artifacts are incompatible.")
        return

    disease_model = artifacts["disease_model.joblib"]
    risk_model = artifacts["risk_model.joblib"]
    disease_encoder = artifacts["disease_label_encoder.joblib"]
    risk_encoder = artifacts["risk_label_encoder.joblib"]
    feature_names = artifacts["feature_names.joblib"]
    preprocessor = artifacts["preprocessor.joblib"]

    intake_tab, results_tab, summary_tab = st.tabs(["Patient Intake", "Assessment Results", "Patient Summary"])

    with intake_tab:
        inputs, submitted = collect_inputs()
        input_df = build_input_dataframe(inputs)

        if submitted:
            try:
                processed_input = preprocessor.transform(input_df)
                disease_label, disease_confidence = get_prediction(disease_model, disease_encoder, input_df)
                risk_label, risk_confidence = get_prediction(risk_model, risk_encoder, input_df)
                st.session_state.assessment = {
                    "input_df": input_df,
                    "processed_input": processed_input,
                    "disease_label": disease_label,
                    "disease_confidence": disease_confidence,
                    "risk_label": risk_label,
                    "risk_confidence": risk_confidence,
                    "clinical_reason": build_ai_reason(inputs, disease_label),
                }
                st.success("Clinical assessment completed. Open Assessment Results to review the output.")
            except Exception as ex:
                st.error(f"Assessment failed: {ex}")

    assessment = st.session_state.get("assessment")

    with results_tab:
        if not assessment:
            st.info("Run a clinical assessment from Patient Intake to generate results.")
        else:
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                render_result(
                    "Disease Prediction",
                    assessment["disease_label"],
                    f"Confidence: {assessment['disease_confidence'] * 100:.1f}%",
                )
            with result_col2:
                render_result(
                    "Risk Level",
                    assessment["risk_label"],
                    f"Confidence: {assessment['risk_confidence'] * 100:.1f}%",
                    risk_class(assessment["risk_label"]),
                )

            st.markdown("### Clinical Signal Summary")
            st.write(assessment["clinical_reason"])

            explanation_col1, explanation_col2 = st.columns(2)
            with explanation_col1:
                with st.expander("Disease model feature importance", expanded=True):
                    render_explanation(disease_model, assessment["processed_input"], feature_names)
            with explanation_col2:
                with st.expander("Risk model feature importance", expanded=True):
                    render_explanation(risk_model, assessment["processed_input"], feature_names)

    with summary_tab:
        if not assessment:
            st.info("Run a clinical assessment from Patient Intake to view the patient summary.")
        else:
            render_patient_summary(assessment["input_df"])


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        import traceback

        st.error(f"A fatal error occurred during app execution: {ex}")
        st.code(traceback.format_exc())
