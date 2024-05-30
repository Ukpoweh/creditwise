import streamlit as st

import pickle

import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def main():
    st.title("CreditWise")
    st.subheader("A Credit Approval Prediction System")

    #input prompts
    col1, col2 = st.columns([1,1])
    with col1:
        gender = st.radio("Select your gender", ['M', 'F'])
    with col2:
        annual_income = st.number_input("Enter your annual income")

    col3, col4 = st.columns([1,1])
    with col3:
        own_car = st.radio("Do you have a car", ["Y", "N"])
    with col4:
        own_realty = st.radio("Do you have a property", ["Y", "N"])
    
    col5, col6 = st.columns([1,1])
    with col5:
        income=['Working', 'Commercial associate', 'State servant', 'Student',
       'Pensioner']
        income_type = st.selectbox("Select your type of income", income)
    with col6:
        occupation=['Security staff', 'Sales staff', 'Accountants', 'Laborers',
       'Managers', 'Drivers', 'Core staff', 'High skill tech staff',
       'Cleaning staff', 'Private service staff', 'Cooking staff',
       'Low-skill Laborers', 'Medicine staff', 'Secretaries',
       'Waiters/barmen staff', 'HR staff', 'Realty agents', 'IT staff']
        occupation_type = st.selectbox("Select your type of occupation", occupation)
    
    col7, col8 = st.columns([1,1])
    with col7:
        education=['Secondary / secondary special', 'Higher education',
       'Incomplete higher', 'Lower secondary', 'Academic degree']``````````````
        education_level = st.selectbox("Select your level of education", education)
    with col6:
        family=['Married', 'Single / not married', 'Civil marriage', 'Separated',
       'Widow']
        family_status = st.selectbox("Select your family status", family)

    col9, col10 = st.columns([1,1])
    with col9:
        housing=['House / apartment', 'Rented apartment', 'Municipal apartment',
       'With parents', 'Co-op apartment', 'Office apartment']
        housing_type = st.selectbox("Select your type of housing", housing)
    with col10:
        start_employment_date = st.date_input("Enter the start date for your current emloyment")
    
    col11, col12 = st.columns([1,1])
    with col11:
        children_count = st.number_input("Count of chldren", step=1, min_value=0)
    with col12:
        fam_count = st.number_input("Count of family members", step=1, min_value=0)
    
    col13, col14 = st.columns([1,1])
    with col13:
        own_phone = st.radio("Do you have a phone", ["0", ""])
    with col14:
        own_email = st.radio("Do you have an email", ["0", "1"])


    labeled_categories = np.array(encoder.transform([gender, own_car, own_realty, income_type, education_level, family_status, housing_type, occupation_type])).reshape(1, -1)
    numerical_features = np.array([annual_income, children_count, fam_count, own_phone, own_email])

if __name__ == "__main__":
    main()

