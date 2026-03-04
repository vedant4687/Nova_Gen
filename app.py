import streamlit as st
import joblib
import pandas as pd

# 1. Load objects
# It is best to wrap these in a function with st.cache_resource so they don't reload on every click
@st.cache_resource
def load_assets():
    model = joblib.load('NovaGen.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. EXACT FEATURE ORDER (Matches your model's training memory)
FINAL_FEATURE_ORDER = [
    'Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level', 
    'Heart_Rate', 'Sleep_Hours', 'Exercise_Hours', 'Water_Intake', 
    'Stress_Level', 'Smoking', 'Alcohol', 'Diet', 'MentalHealth', 
    'PhysicalActivity', 'MedicalHistory', 'Allergies',
    'Blood_Group_AB', 'Blood_Group_B', 'Blood_Group_O', 
    'Diet_Type__Vegan', 'Diet_Type__Vegetarian'
]

st.title("🏥 NovaGen Health Risk Predictor")

# 3. Define UI Inputs
col1, col2 = st.columns(2)

with col1:
    # We assign the widget output to 'age_input', 'bmi_input', etc.
    age_input = st.number_input("Age", 0, 100, 30)
    bmi_input = st.number_input("BMI", 10.0, 50.0, 25.0)
    bp_input = st.number_input("Blood Pressure", 50, 230, 120)
    chol_input = st.number_input("Cholesterol", 150, 250, 200)
    gl_input = st.number_input("Glucose Level", 90, 110, 100)
    hr_input = st.number_input("Heart Rate", 60, 100, 75)
    sleep_input = st.slider("Sleep Hours", 0, 14, 7)

with col2:
    ex_input = st.slider("Exercise Hours", 0, 10, 2)
    water_input = st.slider("Water Intake (L)", 0, 10, 3)
    stress_input = st.slider("Stress Level", 0, 12, 5)
    smoke_input = st.selectbox("Smoking Status", [0, 1, 2])
    alc_input = st.selectbox("Alcohol Status", [0, 1, 2])
    diet_input = st.selectbox("Diet Quality", [0, 1, 2])
    mental_input = st.selectbox("Mental Health", [0, 1, 2])
    phys_input = st.selectbox("Physical Activity", [0, 1, 2])
    med_input = st.selectbox("Medical History", [0, 1, 2])
    all_input = st.selectbox("Allergies", [0, 1, 2])

# 4. Prediction Logic
if st.button("Predict Risk", type="primary"):
    # Create dictionary using the variables defined above
    raw_data = {
        'Age': age_input, 
        'BMI': bmi_input, 
        'Blood_Pressure': bp_input, 
        'Cholesterol': chol_input, 
        'Glucose_Level': gl_input, 
        'Heart_Rate': hr_input, 
        'Sleep_Hours': sleep_input, 
        'Exercise_Hours': ex_input, 
        'Water_Intake': water_input, 
        'Stress_Level': stress_input, 
        'Smoking': smoke_input, 
        'Alcohol': alc_input, 
        'Diet': diet_input, 
        'MentalHealth': mental_input, 
        'PhysicalActivity': phys_input, 
        'MedicalHistory': med_input, 
        'Allergies': all_input,
        # Ghost features to satisfy the model's 22-feature requirement
        'Blood_Group_AB': 0, 
        'Blood_Group_B': 0, 
        'Blood_Group_O': 0, 
        'Diet_Type__Vegan': 0, 
        'Diet_Type__Vegetarian': 0
    }

    # Convert to DataFrame and Force Order
    input_df = pd.DataFrame([raw_data])
    input_df = input_df[FINAL_FEATURE_ORDER]

    # Transform and Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("High Health Risk Detected")
    else:
        st.success("Low Health Risk Detected")