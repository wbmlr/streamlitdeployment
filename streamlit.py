import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

# Define the coefficients and odds ratios (replace with your actual values!)
coefficients = {
    'const': -0.006893,
    'Pclass': -0.616975,
    'Sex': 1.506248,
    'Age': -0.027390,
    'SibSp': -0.500616,
    'Parch': -0.372352,
    'Fare': 0.005577,
    'Title_Dr': -0.236137,
    'Title_Master': 1.454651,
    'Title_Miss': -0.145968,
    'Title_Mr': -1.370176,
    'Title_Mrs': 1.034044,
    'Title_Ms': 0.184981,
    'Title_Rev': -0.832875,
    'First_Letter_Cabin_B': 0.286805,
    'First_Letter_Cabin_C': -0.175621,
    'First_Letter_Cabin_D': 0.698538,
    'First_Letter_Cabin_E': 1.101272,
    'First_Letter_Cabin_F': 0.318583,
    'First_Letter_Cabin_G': -0.857520,
    'First_Letter_Cabin_χ': -0.387585
}


def calculate_probability(input_data, coefficients):
    """
    Calculates the probability of survival using logistic regression.

    Args:
        input_data (dict): Dictionary of input features.
        coefficients (dict): Dictionary of model coefficients.

    Returns:
        float: Probability of survival (between 0 and 1).
    """

    # Calculate the linear combination of features and coefficients
    linear_combination = coefficients['const']  # Start with the constant
    for feature, value in input_data.items():
        if feature in coefficients:
            linear_combination += coefficients[feature] * value  # Multiply coefficient by feature value
    # Apply the sigmoid function
    probability = 1 / (1 + math.exp(-linear_combination))
    return probability


def main():
    st.title("Titanic Survival Prediction")

    # Input features
    st.sidebar.header("Input Features")

    pclass = st.sidebar.selectbox("Pclass (Passenger Class)", [1, 2, 3], index=0) # Correct index, default to 1
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 100, 30) #Added reasonable age range
    sibsp = st.sidebar.slider("SibSp (Number of Siblings/Spouses Aboard)", 0, 8, 0)
    parch = st.sidebar.slider("Parch (Number of Parents/Children Aboard)", 0, 6, 0)
    fare = st.sidebar.slider("Fare", 0.0, 512.3292, 25.0)  # Added range consistent with the provided info
    title = st.sidebar.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Ms"]) #Add more possible Titles

    cabin = st.sidebar.selectbox("Cabin First Letter", ['B', 'C', 'D', 'E', 'F', 'G', 'χ', 'None'], index=7) #None as Default


    # Create input dictionary
    input_data = {
        'Pclass': pclass,
        'Sex': 1 if sex == "male" else 0, #Convert to Numerical
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Title_Dr': 1 if title == "Dr" else 0,
        'Title_Master': 1 if title == "Master" else 0,
        'Title_Miss': 1 if title == "Miss" else 0,
        'Title_Mr': 1 if title == "Mr" else 0,
        'Title_Mrs': 1 if title == "Mrs" else 0,
        'Title_Ms': 1 if title == "Ms" else 0,
        'Title_Rev': 1 if title == "Rev" else 0,
        'First_Letter_Cabin_B': 1 if cabin == "B" else 0,
        'First_Letter_Cabin_C': 1 if cabin == "C" else 0,
        'First_Letter_Cabin_D': 1 if cabin == "D" else 0,
        'First_Letter_Cabin_E': 1 if cabin == "E" else 0,
        'First_Letter_Cabin_F': 1 if cabin == "F" else 0,
        'First_Letter_Cabin_G': 1 if cabin == "G" else 0,
        'First_Letter_Cabin_χ': 1 if cabin == "χ" else 0,
    }



    # Prediction
    if st.button("Predict Survival"):
        probability = calculate_probability(input_data, coefficients)
        survival = "Yes" if probability >= 0.5 else "No"  # Set the threshold to 0.5

        st.subheader("Prediction")
        st.write(f"Probability of Survival: {probability:.2f}")
        st.write(f"Survival: {survival}")



if __name__ == "__main__":
    main()
