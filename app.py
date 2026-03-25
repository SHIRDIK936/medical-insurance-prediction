import streamlit as st
import numpy as np
import pickle

# Load model and scaler
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# Page settings
st.set_page_config(page_title="Insurance Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Medical Insurance Price Prediction</h1>", unsafe_allow_html=True)
st.divider()

# 🧾 PERSONAL DETAILS
st.subheader("Personal Details")
name = st.text_input("Full Name")
phone = st.text_input("Phone Number")
email = st.text_input("Email (optional)")
st.divider()

# 🩺 BASIC HEALTH INFO
st.subheader("Basic Health Info")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65)
    bmi = st.number_input("BMI", 10.0, 50.0, value=25.0)
with col2:
    children = st.slider("Children", 0, 5)
    sex = st.selectbox("Sex", ["female", "male"])
st.divider()

# 🌿 LIFESTYLE
st.subheader("Lifestyle Details")
smoker = st.selectbox("Smoker", ["no", "yes"])
activity = st.selectbox("Physical Activity", ["low", "moderate", "high"])
stress = st.selectbox("Stress Level", ["low", "medium", "high"])
st.divider()

# 🏥 MEDICAL + FINANCIAL
st.subheader("Medical & Financial Details")
medical_history = st.text_area("Medical History (e.g., diabetes, BP, none)")
income = st.selectbox("Income Level", ["low", "middle", "high"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
st.divider()

# INR format function
def format_inr(number):
    number = round(number, 2)
    s = str(f"{number:.2f}")
    integer, decimal = s.split('.')
    last3 = integer[-3:]
    rest = integer[:-3]
    if rest != '':
        rest = rest[::-1]
        rest = ','.join([rest[i:i+2] for i in range(0, len(rest), 2)])
        rest = rest[::-1]
        formatted = rest + ',' + last3
    else:
        formatted = last3
    return f"₹{formatted}.{decimal}"

# Prediction Logic
if st.button("Predict Price"):
    phone_clean = phone.strip()
    medical_history_clean = medical_history.lower()

    if name.strip() == "":
        st.warning("⚠️ Please enter your name")
    elif not phone_clean.isdigit() or len(phone_clean) != 10:
        st.warning("⚠️ Please enter a valid 10-digit phone number")
    else:
        # --- CRITICAL FIX: FEATURE ENCODING & ORDER ---
        # Binary encoding
        sex_encoded = 1.0 if sex == "male" else 0.0
        smoker_encoded = 1.0 if smoker == "yes" else 0.0

        # Region One-Hot Encoding (Matches drop_first=True in training)
        r_nw = 1.0 if region == "northwest" else 0.0
        r_se = 1.0 if region == "southeast" else 0.0
        r_sw = 1.0 if region == "southwest" else 0.0

        # EXACT ORDER REQUIRED BY SCALER:
        # age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest
        input_data = np.array([[
            float(age), 
            sex_encoded, 
            float(bmi), 
            float(children), 
            smoker_encoded,
            r_nw, 
            r_se, 
            r_sw
        ]])

        try:
            # 1. Scale Input
            input_scaled = scaler.transform(input_data)

            # 2. Predict
            result = model.predict(input_scaled)
            prediction = float(np.ravel(result)[0])

            # 3. Handle Negative/Zero Outputs
            prediction = abs(prediction)

            # 4. Conversion (USD -> INR)
            inr = prediction * 83

            # 5. Logic Adjustments
            if any(word in medical_history_clean for word in ["diabetes", "bp", "heart", "asthma"]):
                inr *= 1.2
            
            if activity == "high": inr *= 0.9
            elif activity == "low": inr *= 1.1
            
            if stress == "high": inr *= 1.1
            if income == "high": inr *= 1.05

            # Display Result
            st.success(f"### Estimated Insurance Cost: {format_inr(inr)}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("<hr>", unsafe_allow_html=True)