import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'diabetes_model.joblib'
svc_model = joblib.load(model_filename)

# Set Streamlit app configuration (theme and layout)
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Display an image related to diabetes at the top with reduced size
st.image("diabetes_image.jpg", caption="Diabetes Awareness", use_column_width=False, width=300)

# Streamlit App
st.markdown("<h1 style='color: red;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

st.write("""
This app uses a machine learning model to predict whether an individual is likely to have diabetes.
Simply enter the required health information, and the model will provide a prediction.
""")

# Collect user input
st.markdown("### Enter Health Information:")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Make a prediction
if st.button("Predict"):
    prediction = svc_model.predict(input_data)
    if prediction[0] == 1:
        st.error("This individual is likely to have diabetes.")
    else:
        st.success("This individual is not likely to have diabetes.")

st.write("Please note that this prediction is based on the model's output and should not be considered a definitive medical diagnosis. Consult a healthcare professional for medical advice.")

# Custom CSS to style the app (e.g., background color, text color)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput {
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)
