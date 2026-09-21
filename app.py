import streamlit as st
import pandas as pd
import joblib


# Page settings
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰"
)


# Title
st.title("💰 Loan Approval Prediction")

st.write("Enter the applicant details to predict loan approval.")


# Load trained model
model = joblib.load("model_joblib.pkl")


# Income
income = st.number_input(
    "Income",
    min_value=0,
    value=50000
)


# Credit Score
creditscore = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)


# Employment Type
employment_type = st.selectbox(
    "Employment Type",
    [
        "Salaried",
        "Self-Employed",
        "Unemployed"
    ]
)


# Region
region = st.selectbox(
    "Region",
    [
        "Chennai",
        "Bangalore",
        "Coimbatore",
        "Madurai",
        "Salem"
    ]
)


# Prediction button
if st.button("Predict Loan Approval"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [
            {
                "income": income,
                "creditscore": creditscore,
                "_employment_type": employment_type,
                "region": region
            }
        ]
    )


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Get probability
    probability = model.predict_proba(input_data)[0]


    # Display result
    if prediction == 1:

        st.success("✅ Loan Approved")

        st.write(
            f"Approval Probability: {probability[1] * 100:.2f}%"
        )

    else:

        st.error("❌ Loan Rejected")

        st.write(
            f"Approval Probability: {probability[1] * 100:.2f}%"
        )