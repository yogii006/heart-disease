
import streamlit as st
import sklearn
import pickle
import numpy as np
import pandas as pd


st.header("Heart disease predictor")
new_model = pickle.load(open('model1.pkl','rb'))


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = new_model.predict(input_data_reshaped)
st.write(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
