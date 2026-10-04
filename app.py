import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Pharmacy Triage AI", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("final_clean_master.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgboost_triage_model.pkl")

df = load_data()
model = load_model()

features = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'avg_adherence', 'total_meds', 'abnormal_labs', 'total_tests'
]

# 1. Run the AI on the entire dataset instantly in the background
X_all = df[features].fillna(0)
df['Predicted_Tier'] = model.predict(X_all)

# 2. Map the predictions to visual labels
def get_tier_label(tier):
    if tier == 0: return "🔴 Tier 1"
    elif tier == 1: return "🟡 Tier 2"
    else: return "🟢 Tier 3"

df['Tier_Label'] = df['Predicted_Tier'].apply(get_tier_label)

# 3. Fuse the ID and the Tier for the UI
df['Dropdown_Display'] = df['patient_id'] + " (" + df['Tier_Label'] + ")"

st.title("Patient Refill Triage Profile")
st.write("Evaluating risk levels 5 days prior to medication refill.")

st.sidebar.header("Patient Lookup")

# Grabbing the first 100 patients to act as our daily refill queue for the demo
demo_df = df.head(100)
display_queue = demo_df['Dropdown_Display'].tolist()

selected_display = st.sidebar.selectbox("Select Patient from Refill Queue:", display_queue)

if selected_display:
    # Extract just the patient_id from the string so our logic doesn't break
    selected_patient = selected_display.split(" (")[0]
    patient_data = df[df['patient_id'] == selected_patient].iloc[0]
    
    st.subheader(f"Clinical Snapshot: {selected_patient}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Age", int(patient_data['age']))
    
    adherence = round(patient_data['avg_adherence'], 1) if pd.notna(patient_data['avg_adherence']) else 0
    col2.metric("Adherence %", f"{adherence}%")
    
    col3.metric("Abnormal Lab Count", int(patient_data['abnormal_labs']))
    col4.metric("ICU Admissions", int(patient_data['icu_admissions']))
    
    pred = patient_data['Predicted_Tier']
    
    st.markdown("---")
    st.subheader("AI Triage Recommendation")
    
    if pred == 0:
        st.error("🚨 TIER 1: CRITICAL RISK. High lab abnormalities or critical non-adherence. Direct Pharmacist Call Required.")
    elif pred == 1:
        st.warning("⚠️ TIER 2: MODERATE RISK. Sliding adherence or moderate lab flags. Automated SMS Alert Triggered.")
    else:
        st.success("✅ TIER 3: LOW RISK. Patient is stable. Proceed with standard automated refill.")