import streamlit as st
import pickle as pkl
import numpy as np

class_list = {'1': 'Pass', '0': 'Out'}

st.title('Admit Student')
input = open('lr_admit.pkl', 'rb')
model = pkl.load(input)

st.header('Write Name')
txt = st.text_area("","")
                         
if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.text(class_list[label])
