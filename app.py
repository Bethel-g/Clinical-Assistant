import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import validate_numeric_value, explain_prediction


MODEL_DIR = 'models'

LANGUAGE_LABELS = {
    'app_title': {'English': 'EthioHealth-AI Clinical Assistant', 'Amharic': 'EthioHealth-AI ለክሊኒክ እርዳታ'},
    'app_subtitle': {'English': 'Ethiopian Clinical Decision Support with explainable AI', 'Amharic': 'የኢትዮጵያ የሕክምና ዳይግኖስቲክ ድጋፍ ከተገለፀ የAI እይታ'},
    'language_label': {'English': 'Language', 'Amharic': 'ቋንቋ'},
    'sidebar_title': {'English': 'EthioHealth-AI Language Switch', 'Amharic': 'EthioHealth-AI ቋንቋ ቀይር'},
    'page_selector': {'English': 'Navigation', 'Amharic': 'አሰሳ'},
    'page_home': {'English': 'Home', 'Amharic': 'መነሻ'},
    'page_support': {'English': 'Support', 'Amharic': 'ድጋፍ'},
    'patient_profile': {'English': 'Patient Profile', 'Amharic': 'የታካሚ መረጃ'},
    'age': {'English': 'Age', 'Amharic': 'ዕድሜ'},
    'gender': {'English': 'Gender', 'Amharic': 'ጾታ'},
    'region': {'English': 'Region', 'Amharic': 'ክልል'},
    'fever': {'English': 'Fever', 'Amharic': 'የሙቀት ምልክት'},
    'cough': {'English': 'Cough', 'Amharic': 'ሕክምና ይዘት'},
    'headache': {'English': 'Headache', 'Amharic': 'ርእስ ህመም'},
    'fatigue': {'English': 'Fatigue', 'Amharic': 'ድካም'},
    'vomiting': {'English': 'Vomiting', 'Amharic': 'ማር'},
    'diarrhea': {'English': 'Diarrhea', 'Amharic': 'ዳይአሪያ'},
    'temperature': {'English': 'Temperature (°C)', 'Amharic': 'ሙቀት (°C)'},
    'heart_rate': {'English': 'Heart Rate', 'Amharic': 'የልብ ግፊት'},
    'comorbidity': {'English': 'Comorbidity', 'Amharic': 'የተጨማሪ ታማሚነት'},
    'season': {'English': 'Season', 'Amharic': 'የወቅት ጊዜ'},
    'predict_disease': {'English': 'Predict Disease', 'Amharic': 'የበሽታ እትንት'},
    'predict_risk': {'English': 'Predict Risk', 'Amharic': 'የአደጋ ደረጃ እትንት'},
    'predict_stay': {'English': 'Predict Stay', 'Amharic': 'የማረፊያ ጊዜ እትንት'},
    'explanation': {'English': 'Explanation', 'Amharic': 'ምክንያት'},
    'patient_summary': {'English': 'Patient Summary', 'Amharic': 'የታካሚ ማጠቃለያ'},
    'confidence': {'English': 'Confidence', 'Amharic': 'እምነት'},
    'no_data': {'English': 'No', 'Amharic': 'አይ'},
    'yes_data': {'English': 'Yes', 'Amharic': 'አዎ'},
    'this_estimate': {
        'English': 'This estimate is based on the patient profile and clinical risk factors.',
        'Amharic': 'ይህ ግምገማ በታካሚው መገለጫ እና የሕክምና አደጋ አይነቶች የተመሠረተ ነው።'
    },
    'overview_header': {
        'English': 'Accelerating Decisions for Ethiopian Healthcare',
        'Amharic': 'ለኢትዮጵያ ጤና እርምጃ ፈጣን ውሳኔ'
    },
    'overview_text': {
        'English': 'Use EthioHealth-AI to generate fast disease, risk, and stay predictions with clear explanations. Adjust the language and patient inputs, then evaluate results in a clean clinical interface.',
        'Amharic': 'EthioHealth-AIን ለፈጣን የበሽታ ፣ የአደጋ ደረጃ እና የማረፊያ ጊዜ ግምገማ ለማጀመር ይጠቀሙ። ቋንቋን ይቀይሩ እና መረጃን ያስገቡ።'
    },
    'feature_disease': {
        'English': 'Disease Prediction',
        'Amharic': 'የበሽታ እትንት'
    },
    'feature_risk': {
        'English': 'Risk Level Assessment',
        'Amharic': 'የአደጋ ደረጃ ግምገማ'
    },
    'feature_stay': {
        'English': 'Stay Estimate',
        'Amharic': 'የማረፊያ ጊዜ ግምገማ'
    },
    'feature_explain': {
        'English': 'Explainable AI',
        'Amharic': 'አስረዳዊ AI'
    },
    'feature_clustering': {
        'English': 'Patient Clustering',
        'Amharic': 'የታካሚ ክለስተሪንግ'
    },
    'feature_disease_desc': {
        'English': 'Predict the most likely disease using clinical and symptom data.',
        'Amharic': 'በክሊኒክ መረጃ እና ምልክቶች ላይ የተመሰረተ የበሽታ ግምገማ ያድርጉ።'
    },
    'feature_risk_desc': {
        'English': 'Estimate Low, Medium, or High risk levels for each patient.',
        'Amharic': 'ለእያንዳንዱ ታካሚ የዝቅ ፣ መካከለኛ ወይም ከፍተኛ አደጋ ደረጃ ይገምጹ።'
    },
    'feature_stay_desc': {
        'English': 'Predict how many days patients are likely to stay in the hospital.',
        'Amharic': 'ታካሚዎች በሆስፒታል ምን ያህል ቀን እንደሚቀመጡ ይገምጹ።'
    },
    'feature_explain_desc': {
        'English': 'View the top features that influence each prediction.',
        'Amharic': 'የእያንዱን ግምገማ የሚቀጥሉ እንደሆኑ ምንጮች ይታዩ።'
    },
    'feature_clustering_desc': {
        'English': 'Group patients into clusters for better clinical insights.',
        'Amharic': 'ለሕክምና ምክንያቶች ታካሚዎችን ይደርጉ።'
    },
    'model_missing': {
        'English': 'Run `python train.py --data your_dataset.csv` to build models first.',
        'Amharic': 'የሞዴሉን ማከናወን ከመጀመሩ በፊት `python train.py --data your_dataset.csv` ይሂዱ።'
    },
    'home_description_title': {
        'English': 'About This Clinical Decision Support System',
        'Amharic': 'ስለዚህ የክሊኒክ ውሳኔ ድጋፍ ስርዓት'
    },
    'home_description': {
        'English': 'This application separates the introduction from the prediction workspace. Use Home to understand the purpose of EthioHealth-AI, then open Support to enter patient information and generate disease, risk, and hospital stay predictions.',
        'Amharic': 'ይህ መተግበሪያ መግቢያውን ከግምት መስሪያ ቦታው ይለያል። በመነሻ ገጽ EthioHealth-AI ምን እንደሚረዳ ይመልከቱ፣ ከዚያ በድጋፍ ገጽ የታካሚ መረጃ አስገብተው የበሽታ፣ የአደጋ ደረጃ እና የሆስፒታል ቆይታ ግምት ያግኙ።'
    },
    'support_title': {
        'English': 'Prediction Support',
        'Amharic': 'የግምት ድጋፍ'
    },
    'support_intro': {
        'English': 'Enter the patient profile in the sidebar, then choose the prediction you want to run.',
        'Amharic': 'በጎን አሞሌው የታካሚውን መረጃ ያስገቡ፣ ከዚያ ማስኬድ የሚፈልጉትን ግምት ይምረጡ።'
    },
    'description': {
        'English': 'EthioHealth-AI is designed to help clinicians by generating provisional predictions and explanations based on historical Ethiopian-style healthcare data.',
        'Amharic': 'EthioHealth-AI በኢትዮጵያዊ የጤና ውሂብ መሠረት እና በቅድሚያ የግምት ምስጢርነት ለክሊኒክ ባለሞያዎች ለማገዝ ተዘጋጅቷል።'
    }
}

OPTION_TRANSLATIONS = {
    'YesNo': {
        'English': {'No': 'No', 'Yes': 'Yes'},
        'Amharic': {'No': 'አይ', 'Yes': 'አዎ'}
    },
    'Gender': {
        'English': {'Male': 'Male', 'Female': 'Female', 'Other': 'Other'},
        'Amharic': {'Male': 'ወንድ', 'Female': 'ሴት', 'Other': 'ሌላ'}
    },
    'Season': {
        'English': {'Summer': 'Summer', 'Autumn': 'Autumn', 'Winter': 'Winter', 'Spring': 'Spring'},
        'Amharic': {'Summer': 'ክረምት', 'Autumn': 'መጋቢት', 'Winter': 'ጥር', 'Spring': 'ጥርስ'}
    }
}


def get_label(key: str, lang: str) -> str:
    return LANGUAGE_LABELS.get(key, {}).get(lang, key)


def translate_option(option_type: str, selected_value: str, lang: str) -> str:
    options = OPTION_TRANSLATIONS.get(option_type, {}).get(lang, {})
    reverse_map = {display: internal for internal, display in options.items()}
    return reverse_map.get(selected_value, selected_value)


def get_option_choices(option_type: str, lang: str):
    return list(OPTION_TRANSLATIONS.get(option_type, {}).get(lang, {}).values())


def load_artifacts():
    required_files = [
        'preprocessor.joblib',
        'disease_model.joblib',
        'risk_model.joblib',
        'stay_model.joblib',
        'disease_label_encoder.joblib',
        'risk_label_encoder.joblib',
        'feature_names.joblib'
    ]
    artifacts = {}
    for filename in required_files:
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model artifact not found: {path}")
        artifacts[filename] = joblib.load(path)
    return artifacts


def build_input_dataframe(inputs):
    return pd.DataFrame([inputs])


def get_confidence(prediction_proba):
    if prediction_proba is None:
        return 0.0
    return float(np.max(prediction_proba))


def display_explanation(model, input_df, feature_names):
    try:
        explanation = explain_prediction(model, input_df, feature_names, top_n=5)
        for line in explanation:
            st.write(f"- {line}")
    except Exception as ex:
        st.warning(f"Unable to explain prediction: {ex}")


def collect_sidebar_inputs(lang: str):
    st.sidebar.markdown('---')
    st.sidebar.header(get_label('patient_profile', lang))

    age = st.sidebar.number_input(get_label('age', lang), min_value=0, max_value=120, value=35)
    gender_display = st.sidebar.selectbox(get_label('gender', lang), get_option_choices('Gender', lang))
    gender = translate_option('Gender', gender_display, lang)
    region = st.sidebar.selectbox(get_label('region', lang), ['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'Southern', 'Afar', 'Somali', 'Benishangul', 'Gambela', 'Harari', 'Dire Dawa'])
    fever_display = st.sidebar.selectbox(get_label('fever', lang), get_option_choices('YesNo', lang))
    fever = translate_option('YesNo', fever_display, lang)
    cough_display = st.sidebar.selectbox(get_label('cough', lang), get_option_choices('YesNo', lang))
    cough = translate_option('YesNo', cough_display, lang)
    headache_display = st.sidebar.selectbox(get_label('headache', lang), get_option_choices('YesNo', lang))
    headache = translate_option('YesNo', headache_display, lang)
    fatigue_display = st.sidebar.selectbox(get_label('fatigue', lang), get_option_choices('YesNo', lang))
    fatigue = translate_option('YesNo', fatigue_display, lang)
    vomiting_display = st.sidebar.selectbox(get_label('vomiting', lang), get_option_choices('YesNo', lang))
    vomiting = translate_option('YesNo', vomiting_display, lang)
    diarrhea_display = st.sidebar.selectbox(get_label('diarrhea', lang), get_option_choices('YesNo', lang))
    diarrhea = translate_option('YesNo', diarrhea_display, lang)
    temperature = st.sidebar.number_input(get_label('temperature', lang), min_value=30.0, max_value=45.0, value=37.0, format='%.1f')
    heart_rate = st.sidebar.number_input(get_label('heart_rate', lang), min_value=30, max_value=200, value=80)
    comorbidity_display = st.sidebar.selectbox(get_label('comorbidity', lang), get_option_choices('YesNo', lang))
    comorbidity = translate_option('YesNo', comorbidity_display, lang)
    season_display = st.sidebar.selectbox(get_label('season', lang), get_option_choices('Season', lang))
    season = translate_option('Season', season_display, lang)

    inputs = {
        'Age': age,
        'Gender': gender,
        'Region': region,
        'Fever': fever,
        'Cough': cough,
        'Headache': headache,
        'Fatigue': fatigue,
        'Vomiting': vomiting,
        'Diarrhea': diarrhea,
        'Temperature': temperature,
        'Heart Rate': heart_rate,
        'Comorbidity': comorbidity,
        'Season': season
    }
    return inputs


def render_styles():
    st.markdown(
        """
        <style>
            :root {
                --ink: #17211d;
                --muted: #64736d;
                --line: #dfe8e3;
                --panel: #ffffff;
                --soft: #f4f8f6;
                --teal: #0f766e;
                --green: #2f855a;
                --amber: #b7791f;
                --blue: #2b6cb0;
            }

            .stApp {
                background:
                    radial-gradient(circle at 18% 6%, rgba(15, 118, 110, 0.08), transparent 28%),
                    linear-gradient(180deg, #f8fbfa 0%, #eef5f1 100%);
                color: var(--ink);
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 1.6rem;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            .app-shell {
                border: 1px solid rgba(15, 118, 110, 0.14);
                border-radius: 8px;
                padding: 28px 30px;
                background: linear-gradient(135deg, #ffffff 0%, #eef8f4 100%);
                box-shadow: 0 18px 45px rgba(20, 83, 70, 0.10);
                margin-bottom: 22px;
            }

            .eyebrow {
                color: var(--teal);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0;
                text-transform: uppercase;
                margin-bottom: 10px;
            }

            .hero-title {
                color: #12372f;
                font-size: 2.45rem;
                font-weight: 800;
                line-height: 1.12;
                margin: 0 0 10px;
            }

            .hero-subtitle {
                color: #3e4f49;
                font-size: 1.04rem;
                line-height: 1.65;
                max-width: 760px;
                margin: 0;
            }

            .status-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 22px;
            }

            .status-tile,
            .feature-card,
            .result-card,
            .summary-card {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: var(--panel);
                box-shadow: 0 10px 24px rgba(23, 33, 29, 0.06);
            }

            .status-tile {
                padding: 14px 16px;
            }

            .status-value {
                color: #12372f;
                font-size: 1.35rem;
                font-weight: 800;
                line-height: 1.1;
            }

            .status-label {
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 5px;
            }

            .section-title {
                color: #12372f;
                font-size: 1.35rem;
                font-weight: 800;
                margin: 8px 0 12px;
            }

            .section-copy {
                color: #4a5b55;
                line-height: 1.65;
                margin-bottom: 18px;
            }

            .feature-card {
                min-height: 158px;
                padding: 18px;
                border-top: 4px solid var(--teal);
            }

            .feature-code {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 38px;
                height: 38px;
                border-radius: 8px;
                color: #ffffff;
                background: var(--teal);
                font-weight: 800;
                margin-bottom: 13px;
            }

            .feature-title {
                color: #17211d;
                font-weight: 800;
                font-size: 1rem;
                margin-bottom: 8px;
            }

            .feature-copy {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.55;
            }

            .result-card {
                padding: 18px;
                min-height: 132px;
                border-top: 4px solid var(--blue);
                margin-bottom: 14px;
            }

            .result-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
            }

            .result-value {
                color: #102a24;
                font-size: 1.35rem;
                font-weight: 800;
                margin-top: 6px;
            }

            .summary-card {
                padding: 18px;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 10px;
            }

            .summary-item {
                border: 1px solid #e4ece8;
                border-radius: 8px;
                padding: 10px 12px;
                background: #fbfdfc;
            }

            .summary-key {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
            }

            .summary-value {
                color: #17211d;
                font-weight: 800;
                margin-top: 2px;
                word-break: break-word;
            }

            .stButton > button {
                width: 100%;
                border-radius: 8px;
                border: 1px solid rgba(15, 118, 110, 0.22);
                background: #0f766e;
                color: #ffffff;
                font-weight: 800;
                min-height: 44px;
                box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18);
            }

            .stButton > button:hover {
                border-color: #115e59;
                background: #115e59;
                color: #ffffff;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 12px;
            }

            hr {
                border-color: var(--line);
                margin: 1.4rem 0;
            }

            @media (max-width: 760px) {
                .hero-title {
                    font-size: 1.8rem;
                }

                .app-shell {
                    padding: 22px 18px;
                }

                .status-row,
                .summary-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(lang: str):
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="eyebrow">Clinical Decision Support</div>
            <div class="hero-title">{get_label('app_title', lang)}</div>
            <p class="hero-subtitle">{get_label('app_subtitle', lang)}</p>
            <div class="status-row">
                <div class="status-tile">
                    <div class="status-value">3</div>
                    <div class="status-label">Prediction tasks</div>
                </div>
                <div class="status-tile">
                    <div class="status-value">10+</div>
                    <div class="status-label">ML models trained</div>
                </div>
                <div class="status-tile">
                    <div class="status-value">2</div>
                    <div class="status-label">Interface languages</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_card(code: str, title: str, description: str, accent: str = '#0f766e'):
    st.markdown(
        f"""
        <div class="feature-card" style="border-top-color: {accent};">
            <div class="feature-code" style="background: {accent};">{code}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_result(label: str, value: str, detail: str, accent: str):
    st.markdown(
        f"""
        <div class="result-card" style="border-top-color: {accent};">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="feature-copy">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_patient_summary(inputs):
    summary_items = ''.join(
        f"""
        <div class="summary-item">
            <div class="summary-key">{key}</div>
            <div class="summary-value">{value}</div>
        </div>
        """
        for key, value in inputs.items()
    )
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-grid">{summary_items}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title='EthioHealth-AI Clinical Assistant',
        page_icon='🩺',
        layout='wide'
    )

    render_styles()

    if 'lang' not in st.session_state:
        st.session_state.lang = 'English'
    if 'page_key' not in st.session_state:
        st.session_state.page_key = 'home'

    st.sidebar.markdown('### EthioHealth-AI')
    st.sidebar.caption(get_label('description', st.session_state.lang))
    lang = st.sidebar.radio(get_label('language_label', 'English'), ['English', 'Amharic'], index=0 if st.session_state.lang == 'English' else 1)
    st.session_state.lang = lang
    page_options = {
        get_label('page_home', lang): 'home',
        get_label('page_support', lang): 'support'
    }
    current_page_label = get_label(f"page_{st.session_state.page_key}", lang)
    page_label = st.sidebar.radio(
        get_label('page_selector', lang),
        list(page_options.keys()),
        index=list(page_options.keys()).index(current_page_label)
    )
    st.session_state.page_key = page_options[page_label]

    render_header(lang)

    if st.session_state.page_key == 'home':
        hero_col1, hero_col2 = st.columns([1.4, 1])
        with hero_col1:
            st.markdown(f"<div class='section-title'>{get_label('overview_header', lang)}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='section-copy'>{get_label('overview_text', lang)}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-title'>{get_label('home_description_title', lang)}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='section-copy'>{get_label('home_description', lang)}</p>", unsafe_allow_html=True)
        with hero_col2:
            render_feature_card('CL', get_label('feature_clustering', lang), get_label('feature_clustering_desc', lang), '#2b6cb0')
            render_feature_card('AI', get_label('feature_explain', lang), get_label('feature_explain_desc', lang), '#805ad5')

        st.markdown('<hr />', unsafe_allow_html=True)
        card_1, card_2, card_3, card_4 = st.columns(4)
        with card_1:
            render_feature_card('DX', get_label('feature_disease', lang), get_label('feature_disease_desc', lang), '#0f766e')
        with card_2:
            render_feature_card('RK', get_label('feature_risk', lang), get_label('feature_risk_desc', lang), '#b7791f')
        with card_3:
            render_feature_card('LOS', get_label('feature_stay', lang), get_label('feature_stay_desc', lang), '#2b6cb0')
        with card_4:
            render_feature_card('XAI', get_label('feature_explain', lang), get_label('feature_explain_desc', lang), '#4a5568')
        return

    st.markdown(f"<div class='section-title'>{get_label('support_title', lang)}</div>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-copy'>{get_label('support_intro', lang)}</p>", unsafe_allow_html=True)
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(get_label('model_missing', lang))
        return

    inputs = collect_sidebar_inputs(lang)
    input_df = build_input_dataframe(inputs)
    preprocessor = artifacts['preprocessor.joblib']

    try:
        processed_input = preprocessor.transform(input_df)
    except Exception as ex:
        st.error(f"Input preprocessing failed: {ex}")
        return

    disease_model = artifacts['disease_model.joblib']
    risk_model = artifacts['risk_model.joblib']
    stay_model = artifacts['stay_model.joblib']
    disease_encoder = artifacts['disease_label_encoder.joblib']
    risk_encoder = artifacts['risk_label_encoder.joblib']
    feature_names = artifacts['feature_names.joblib']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="result-label">Disease model</div>', unsafe_allow_html=True)
        if st.button(get_label('predict_disease', lang), type='primary', use_container_width=True):
            disease_pred = disease_model.predict(processed_input)
            disease_proba = disease_model.predict_proba(processed_input) if hasattr(disease_model, 'predict_proba') else None
            disease_label = disease_encoder.inverse_transform(disease_pred)[0]
            confidence = get_confidence(disease_proba)
            render_result(get_label('predict_disease', lang), disease_label, f"{get_label('confidence', lang)}: {confidence:.2f}", '#0f766e')
            st.markdown(f"**{get_label('explanation', lang)}**")
            display_explanation(disease_model, processed_input, feature_names)

    with col2:
        st.markdown('<div class="result-label">Risk model</div>', unsafe_allow_html=True)
        if st.button(get_label('predict_risk', lang), type='primary', use_container_width=True):
            risk_pred = risk_model.predict(processed_input)
            risk_proba = risk_model.predict_proba(processed_input) if hasattr(risk_model, 'predict_proba') else None
            risk_label = risk_encoder.inverse_transform(risk_pred)[0]
            confidence = get_confidence(risk_proba)
            render_result(get_label('predict_risk', lang), risk_label, f"{get_label('confidence', lang)}: {confidence:.2f}", '#b7791f')
            st.markdown(f"**{get_label('explanation', lang)}**")
            display_explanation(risk_model, processed_input, feature_names)

    with col3:
        st.markdown('<div class="result-label">Stay model</div>', unsafe_allow_html=True)
        if st.button(get_label('predict_stay', lang), type='primary', use_container_width=True):
            stay_pred = stay_model.predict(processed_input)
            render_result(get_label('predict_stay', lang), f"{float(stay_pred[0]):.1f} days", get_label('this_estimate', lang), '#2b6cb0')

    st.markdown('<hr />', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{get_label('patient_summary', lang)}</div>", unsafe_allow_html=True)
    render_patient_summary(inputs)


if __name__ == '__main__':
    main()
