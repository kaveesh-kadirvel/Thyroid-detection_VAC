import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


model = pickle.load(open('model.pkl', 'rb'))


training_df = pd.read_csv('Thyroid_Diff.csv')
training_df = training_df.drop(columns=['Recurred'])

categorical_cols = training_df.select_dtypes(include=['object']).columns.tolist()
feature_options = {col: sorted(training_df[col].astype(str).unique()) for col in categorical_cols}
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    training_df[col] = le.fit_transform(training_df[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
scaler.fit(training_df)


st.set_page_config(page_title='Thyroid Disease Prediction', layout='centered')


st.markdown('''
<style>
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
</style>
''', unsafe_allow_html=True)


st.title('🩺 Thyroid Disease Prediction System')
st.markdown('Enter patient details to predict thyroid disease risk.')

st.divider()

st.subheader('👤 Patient Information')
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
with col2:
    gender = st.selectbox('Gender', feature_options['Gender'])

st.subheader('🌡️ Clinical / Risk Factors')
col1, col2 = st.columns(2)
with col1:
    smoking = st.selectbox('Smoking', feature_options['Smoking'])
    hx_smoking = st.selectbox('Hx Smoking', feature_options['Hx Smoking'])
    hx_radiothreapy = st.selectbox('Hx Radiothreapy', feature_options['Hx Radiothreapy'])
    thyroid_function = st.selectbox('Thyroid Function', feature_options['Thyroid Function'])
    physical_exam = st.selectbox('Physical Examination', feature_options['Physical Examination'])
with col2:
    adenopathy = st.selectbox('Adenopathy', feature_options['Adenopathy'])
    pathology = st.selectbox('Pathology', feature_options['Pathology'])
    focality = st.selectbox('Focality', feature_options['Focality'])
    risk_level = st.selectbox('Risk', feature_options['Risk'])
    t_stage = st.selectbox('T', feature_options['T'])

col1, col2 = st.columns(2)
with col1:
    n_stage = st.selectbox('N', feature_options['N'])
    m_stage = st.selectbox('M', feature_options['M'])
with col2:
    stage = st.selectbox('Stage', feature_options['Stage'])
    response = st.selectbox('Response', feature_options['Response'])


user_row = {
    'Age': age,
    'Gender': gender,
    'Smoking': smoking,
    'Hx Smoking': hx_smoking,
    'Hx Radiothreapy': hx_radiothreapy,
    'Thyroid Function': thyroid_function,
    'Physical Examination': physical_exam,
    'Adenopathy': adenopathy,
    'Pathology': pathology,
    'Focality': focality,
    'Risk': risk_level,
    'T': t_stage,
    'N': n_stage,
    'M': m_stage,
    'Stage': stage,
    'Response': response
}

input_df = pd.DataFrame([user_row])
for col in categorical_cols:
    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

input_scaled = scaler.transform(input_df)


if st.button('🔍 Predict'):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader('📊 Result')
    if pred == 1:
        st.error(f'⚠️ High risk of recurrence\n\nConfidence: {prob:.2f}')
    else:
        st.success(f'✅ Low risk of recurrence\n\nConfidence: {1 - prob:.2f}')

    st.caption('📝 This is a predictive model and not a medical diagnosis.')
