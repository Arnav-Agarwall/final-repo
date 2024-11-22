import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('adult.brl.csv', na_values=" ?")

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
x = data.drop("income", axis=1)
y = data["income"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=63)

# Train the model
model = RandomForestClassifier(n_estimators=150, random_state=63)
model.fit(x_train, y_train)

# Streamlit App
st.title("Income Prediction App")
st.write("Enter the following details to predict if an individual's income exceeds $50K:")

# User input for prediction
age = st.number_input("Enter age:", min_value=1, max_value=120)
workclass = st.selectbox("Select workclass:", options=["Private", "Self-emp-not-inc", "Local-gov", "State-gov", 
                                                        "Federal-gov", "Self-emp-inc", "Without-pay", "Never-worked"])
fnlwgt = st.number_input("Enter final weight (fnlwgt):", min_value=1)
education = st.selectbox("Select education:", options=["Bachelors", "HS-grad", "Masters", "Doctorate", 
                                                        "Associates", "Some-college", "10th", "12th", 
                                                        "9th", "7th-8th", "Preschool", "5th-6th", "1st-4th"])
education_num = st.number_input("Enter education number (1-16):", min_value=1, max_value=16)
marital_status = st.selectbox("Select marital status:", options=["Married-civ-spouse", "Never-married", 
                                                                 "Divorced", "Separated", "Widowed", "Married-spouse-absent", 
                                                                 "Married-af-spouse"])
occupation = st.selectbox("Select occupation:", options=["Tech-support", "Sales", "Exec-managerial", 
                                                          "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", 
                                                          "Other-service", "Farming-fishing", "Transport-moving", 
                                                          "Craft-repair", "Adm-clerical", "Protective-serv", 
                                                          "Armed-Forces", "Priv-house-serv", "Security", "Healthcare"])
relationship = st.selectbox("Select relationship:", options=["Husband", "Wife", "Not-in-family", 
                                                              "Other-relative", "Unmarried", "Own-child"])
race = st.selectbox("Select race:", options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
sex = st.selectbox("Select sex:", options=["Male", "Female"])
capital_gain = st.number_input("Enter capital gain:", min_value=0)
capital_loss = st.number_input("Enter capital loss:", min_value=0)
hours_per_week = st.number_input("Enter hours per week:", min_value=1, max_value=100)
native_country = st.selectbox("Select native country:", options=["United-States", "Mexico", "Germany", "Philippines", 
                                                                  "Canada", "Puerto-Rico", "El-Salvador", "India", 
                                                                  "Cuba", "England", "Jamaica", "South", "Italy", 
                                                                  "China", "Japan", "Greece", "Vietnam", "Taiwan", 
                                                                  "Iran", "Ireland", "France", "South Africa", 
                                                                  "Columbia", "Haiti", "Nicaragua", "Scotland", 
                                                                  "Thailand", "Hungary", "Ghana", "Dominican-Republic", 
                                                                  "Laos", "Portugal", "Cambodia", "Zambia", "Norway", 
                                                                  "Iceland", "Peru", "Outlying-US(Guam-USVI-etc)", 
                                                                  "Trinadad&Tobago", "Yugoslavia", "El Salvador"])

# Prepare input for prediction
input_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'education.num': education_num,
    'marital.status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital.gain': capital_gain,
    'capital.loss': capital_loss,
    'hours.per.week': hours_per_week,
    'native.country': native_country
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Encode categorical features for the input DataFrame
for column, le in label_encoders.items():
    if column in input_df.columns:
        input_df[column] = le.transform(input_df[column])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    predicted_class = "<=50K" if prediction[0] == 0 else ">50K"
    st.write(f"The predicted income class is: {predicted_class}")