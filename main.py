import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('USA college admission rate prediction')
input = open('lr_admit.pkl', 'rb')
model = pkl.load(input)

st.header('Input admission information')
gre = st.number_input('Insert GRE Score')
toefl = st.number_input('Insert TOEFL Score')
uni_rate = st.number_input('Insert University Rating')
sop = st.number_input('Insert SOP')
lor = st.number_input('Insert LOR')
cgpa = st.number_input('Insert CGPA')
research = st.radio('Choose Research', [0, 1], index=None)

if gre is not None and toefl is not None and uni_rate is not None and sop is not None and lor is not None and cgpa is not None and research is not None:
    if st.button('Predict'):
        feature_vector = np.array([gre, toefl, uni_rate, sop, lor, cgpa, research]).reshape(1,-1)
        result = str((model.predict(feature_vector)[0])[0])

        st.header('Result')
        st.text(result)
