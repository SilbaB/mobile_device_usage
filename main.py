import streamlit as st
# import streamlit as st
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
     np.random.randn(20, 4),
     columns=['a', 'b', 'c','D'])

st.line_chart(chart_data)
st.title("MY APP")
df = pd.DataFrame(
    {
        'first column': [1, 2, 3, 4, 5],
        'second column': [6, 7, 8, 9, 10]
    }
)
df
st.write("my data frame")
st.write("Choose a number between 1 - 100")
number=st.slider("Number",min_value=0, max_value=100, value=50)
st.write(f"you choose {number}")
if number <50:
    st.write("you are a kalenjin")
elif 50 <= number <60:
    st.write("You are a chinese")
elif 60 <= number <=70:
    st.write("You are a kikuyu")
else :
    st.write("you are a kamba")

if st.button("click me"):
    st.write("Arsenal will win  the premier league")
elif st.button("Nko njaa"):
    st.write("nani hana kwao")

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming the scaler was saved as 'scaler.pkl'

# User inputs
device_model = st.selectbox("Device Model",[0,1,2,3,4])
os = st.selectbox("Operating System",[0,1])
usage_time = st.number_input("App Usage Time (min/day)", min_value=0)
screen_time_per_hr = st.number_input("Screen On Time (hours/day)", min_value=0.0)
battery_train = st.number_input("Battery Drain (mAh/day)", min_value=0)
no_of_apps = st.number_input("Number of Apps Installed", min_value=0)
data_used = st.number_input("Data Usage (MB/day)", min_value=0)
age=st.number_input("Your Age", min_value=0,max_value=100)
Gender=st.selectbox("Gender",[0,1])
# user_behavior = st.number_input("User Behavior Class", min_value=0)

# Prepare the input data for prediction
if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "device_model" : [device_model],
            "os": [os],
            "usage_time": [usage_time],

            "screen_time_per_hr": [screen_time_per_hr],
            "battery_train": [battery_train],
            "apps": [no_of_apps],
            "data_used": [data_used],
            "Age": [age],
            "Gender": [Gender],
            
        }
    )

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result based on prediction
    if prediction[0] == 1:
        st.success("A")
    elif prediction[0] == 2:
        st.info("B")
    elif prediction[0] == 3:
        st.warning("C")
    elif prediction[0] == 4:
        st.warning("D")
    else:
        st.success("E")


# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# import joblib

# model=joblib.load('svm_model.pkl')

# if st.button("Predict"):
#     st.write("Prediction will be done")
#     input_data=pd.DataFrame(
#         {
#             "Device Model":"device_model",
#             "Operating System":"os",
#             "App Usage Time (min/day)":"usage_time",
#             "Screen On Time (hours/day)" : "screen_time_per_hr",
#             "Battery Drain (mAh/day)":"battery_train",
#             "Number of Apps Installed":"apps",
#             "Data Usage (MB/day)":"data_used",
#             "User Behavior Class":"user_behavior"
#         }
#     )
#     scaler=StandardScaler()
#     input_data_scaled=scaler.fit_transform(input_data)
#     #making prediction
#     prediction=model.predict(input_data_scaled)
#     if prediction[0] == 1:
#         st.success("A")
#     elif prediction[0] == 2:
#         st.info("B")
#     elif prediction[0] == 3:
#         st.warning("C")
#     elif prediction[0] == 4:
#         st.warning("D")    
#     else:
#         st.success("E")




