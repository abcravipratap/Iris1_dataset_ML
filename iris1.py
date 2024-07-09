import streamlit as st
import pickle
import numpy as np
with open("iris.pkl","rb") as file:
    iris_model= pickle.load(file)
def predict_function(SL,SW,PL,PW):
    input_array= np.array([[SL,SW,PL,PW]])
    iris_prediction= iris_model.predict(input_array)
    return iris_prediction
    

st.title("Iris Dataset Prediction")
SL= st.slider("sepal_length",min_value=0.1, max_value=99.9)
SW= st.slider("sepal_width",min_value=0.1, max_value=99.9)
PL= st.slider("petal_length",min_value=0.1, max_value=99.9)
PW= st.slider("petal_width",min_value=0.1, max_value=99.9)


st.button("Prediction")
st.write(f"user values are {SL,SW,PL,PW}")
Prediction= predict_function(SL,SW,PL,PW)
Iris_dct= {0:"setosa",1:"versicolor",2:"virginica"}
st.write(f"\n The prediction is {Iris_dct[Prediction[0]]}")


