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
    st.subheader("A Credit Card Approval Prediction System")

    #input prompts
    st.write('----')
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
        #date_obj = datetime.strptime(start_employment_date, "%Y/%m/%d")
        today = datetime.now().date()
        days_difference = (start_employment_date-today).days
    
    col11, col12 = st.columns([1,1])
    with col11:
        children_count = st.number_input("Count of chldren", step=1, min_value=0)
    with col12:
        fam_count = st.number_input("Count of family members", step=1, min_value=0)
    
    col13, col14 = st.columns([1,1])
    with col13:
        own_phone = st.radio("Do you have a phone", [0, 1])
    with col14:
        own_email = st.radio("Do you have an email", [0, 1])
    st.write('----')

    labeled_categories = np.array(encoder.fit_transform([gender, own_car, own_realty, income_type, education_level, family_status, housing_type, occupation_type])).reshape(1, -1)
    numerical_features = np.array([annual_income, children_count, fam_count, own_phone, own_email, days_difference]).reshape(1,-1)


    processed_categories = np.concatenate([labeled_categories, numerical_features], axis=1).reshape(1, -1)
    scaled_features = scaler.transform(processed_categories)

    prediction = model.predict_proba(scaled_features)
    approved_prob = np.round((prediction[:, 1] * 100), 2)[0]

    response = llm_model.generate_content(["A model predicted that the probability for my credit card approval is {approved_prob}%, give me recommendations on how to improve it"], stream=True)
    response.resolve()

    output = f"The probability that your credit card will be approved is {approved_prob}%"

    if st.button('Predict your credit card approval probability'):
        st.success(output)
    
    if st.button("Generate recommendation"):
        st.markdown(response.text)

if __name__ == "__main__":
    main()

