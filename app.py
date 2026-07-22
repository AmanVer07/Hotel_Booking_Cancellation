import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("hotel_booking_cancellation_model.pkl")

st.title("🏨 Hotel Booking Cancellation Predictor")

st.write("Enter booking details")

lead_time = st.slider("Lead Time", 0, 500, 30)

avg_price = st.slider("Average Price Per Room", 0, 20000, 2500)

special_requests = st.slider("Special Requests", 0, 5, 1)

arrival_date = st.slider("Arrival Date", 1, 31, 15)

arrival_month = st.slider("Arrival Month", 1, 12, 6)

total_nights = st.slider("Total Nights", 1, 30, 2)

total_guests = st.slider("Total Guests", 1, 10, 2)

arrival_year = st.selectbox("Arrival Year", [2017, 2018, 2019])

meal_plan = st.selectbox(
    "Meal Plan Not Selected",
    [0, 1]
)

parking = st.selectbox(
    "Parking Required",
    [0, 1]
)

if st.button("Predict"):

    input_data = np.array([[
        lead_time,
        avg_price,
        special_requests,
        arrival_date,
        arrival_month,
        total_nights,
        total_guests,
        arrival_year,
        meal_plan,
        parking
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == 0:

        st.success("✅ Booking Will NOT Be Cancelled")

        st.metric(
            "Booking Success Probability",
            f"{probability[0] * 100:.2f}%"
        )

        st.progress(float(probability[0]))

    else:

        st.error("❌ Booking Will Be Cancelled")

        st.metric(
            "Cancellation Probability",
            f"{probability[1] * 100:.2f}%"
        )

        st.progress(float(probability[1]))


