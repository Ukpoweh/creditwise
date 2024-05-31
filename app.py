import streamlit as st

import pickle

import pandas as pd
import numpy as np
import pickle

from datetime import datetime
import time
import os
from dotenv import load_dotenv
import google.generativeai as ggi

st.set_page_config(page_title="CreditWise", page_icon=":credit_card:")



load_dotenv(".env")
fetched_api_key = os.getenv("API_KEY")
ggi.configure(api_key=fetched_api_key)
llm_model = ggi.GenerativeModel("gemini-pro")


model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

data = pd.read_csv("data.csv")

gender_opt = list(data["CODE_GENDER"].unique())
car_opt = list(data["FLAG_OWN_CAR"].unique())
realty_opt = list(data["FLAG_OWN_REALTY"].unique())
income = list(data["NAME_INCOME_TYPE"].unique())
occupation = list(data["OCCUPATION_TYPE"].unique())
education = list(data["NAME_EDUCATION_TYPE"].unique())
family = list(data["NAME_FAMILY_STATUS"].unique())
housing = list(data["NAME_HOUSING_TYPE"].unique())

def main():
    st.title("CreditWise")
    st.image("credit_card.jpg")
    st.subheader("A Credit Card Approval Prediction System")

    st.markdown("**Welcome to CreditWise!, a credit card approval prediction system offering a valuable solution to both credit card applicants and issuing institutions, leveraging Data Science and Machine Learning techniques to streamline the application process, improve approval rates, as well as enhance financial well-being.**")
    st.sidebar.markdown("""
    ### Key Features:
    - **Real-Time Credit Approval Prediction**: Instantly predicts the likelihood of credit card approval based on applicant data.
    - **Personalized Recommendations**: Offers actionable advice for improving creditworthiness if the application is likely to be rejected.
    - **User-Friendly Interface**: Provides an intuitive web-based interface built with Streamlit for easy data input and feedback.
    - **Machine Learning Integration**: Utilizes advanced machine learning algorithms to ensure accurate and reliable predictions.
    - **Continuous Monitoring and Updates**: Regularly monitors performance and updates the model to maintain and enhance accuracy.
    """)
    st.sidebar.markdown("**Data source can be found on [Kaggle](https://www.kaggle.com/code/hajarlbhyry/credit-card-approval-prediction99-acc-99)**")
    
    #input prompts
    st.write(" ")
    st.write(" ")
    st.markdown("### Get Started!")
    st.write('----')
    st.markdown("*To get started, fill in your necessary financial details below;*")
    col1, col2 = st.columns([1,1])
    with col1:
        gender = st.radio("Select your gender", gender_opt)
    with col2:
        annual_income = st.number_input("Enter your annual income")

    col3, col4 = st.columns([1,1])
    with col3:
        own_car = st.radio("Do you have a car", car_opt)
    with col4:
        own_realty = st.radio("Do you have a property", realty_opt)
    
    col5, col6 = st.columns([1,1])
    with col5:
        income_type = st.selectbox("Select your type of income", income)
    with col6:
        occupation_type = st.selectbox("Select your type of occupation", occupation)
    
    col7, col8 = st.columns([1,1])
    with col7:
        education_level = st.selectbox("Select your level of education", education)
    with col6:
        family_status = st.selectbox("Select your family status", family)

    col9, col10 = st.columns([1,1])
    with col9:
        housing_type = st.selectbox("Select your type of housing", housing)
    with col10:
        start_employment_date = st.date_input("Enter the start date for your current emloyment")
        today = datetime.now().date()
        days_difference = (start_employment_date-today).days
    
    col11, col12 = st.columns([1,1])
    with col11:
        children_count = st.number_input("Count of chldren", step=1, min_value=0)
    with col12:
        fam_count = st.number_input("Count of family members", step=1, min_value=0)    
    col13, col14 = st.columns([1,1])
    with col13:
        own_phone = st.radio("Do you have a phone (0 for No, 1 for Yes)", [0, 1])
    with col14:
        own_email = st.radio("Do you have an email (0 for No, 1 for Yes)", [0, 1])
    st.write('----')

    labeled_categories = np.array(encoder.fit_transform([gender, own_car, own_realty, income_type, education_level, family_status, housing_type, occupation_type])).reshape(1, -1)
    numerical_features = np.array([annual_income, children_count, fam_count, own_phone, own_email, days_difference]).reshape(1,-1)


    processed_categories = np.concatenate([labeled_categories, numerical_features], axis=1).reshape(1, -1)
    scaled_features = scaler.transform(processed_categories)

    prediction = model.predict_proba(scaled_features)
    approved_prob = np.round((prediction[:, 1] * 100), 2)[0]

    response = llm_model.generate_content(["A model predicted that the probability for my credit card approval is {approved_prob}%, give me personalized recommendations on how to improve it. Start with: Here's how you can improve your credit card approval probability;"], stream=True)
    response.resolve()

    output = f"The probability that your credit card will be approved is {approved_prob}%"

    if st.button('Predict your credit card approval probability'):
        st.success(output)
    
    if st.button("Generate recommendations"):
        st.markdown(response.text)

if __name__ == "__main__":
    main()

