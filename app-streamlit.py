import streamlit as st
import pandas as pd
from src.train import train_model
from src.streamlit_utils import check_if_model_exists
from src.data import prepare_data_to_predict_from_uri, deconding_prediction
from src.predict import predict_from_trained_model

st.title('''
Breast Cancer Diagnosis with Python
''')
st.write('''
<p style="text-align:justify">According to the Brazilian National Cancer Institute (INCA), breast cancer is a disease caused by the disordered multiplication of breast cells. 
There are different types of breast cancer, so each case can evolve differently. 
Despite representing around 1% of cases of the disease, even men can suffer from illness. 
Also according to the INCA, there are an estimated 66,280 new cases for the year 2020.</p>

<p style="text-align:justify">This is a proof of concept using streamlit for a breast cancer diagnostic model, 
since one of the fundamental parts of a project is to allow the end user to have access to the product and be able to use it</p>
''', unsafe_allow_html=True)

if check_if_model_exists():
    st.write('<p style="color:blue;"><b>There is already a trained model! You can predict data now!</b></p>', unsafe_allow_html=True)
else:
    st.write('<p style="color:red;"><b>We didn\'t find a trained model! Please train a model before making a prediction</b></p>', unsafe_allow_html=True)

left_column, center_column, right_column = st.beta_columns(3)

with left_column:
    if st.button('Train Model!'):
        with st.spinner('Starting trainning the Model!'):
            train_model(True)
        st.success('Model Trained and Saved!')


di = dict()
if check_if_model_exists():
    with center_column:
        di['radius_mean'] = st.number_input('Radius Mean')
        di['texture_mean'] = st.number_input('Texture Mean')
        di['perimeter_mean'] = st.number_input('Perimeter Mean')
        di['area_mean'] = st.number_input('Area Mean')
        di['smoothness_mean'] = st.number_input('Smoothness Mean')
        di['compactness_mean'] = st.number_input('Compactness Mean')
        di['concavity_mean'] = st.number_input('Concavity Mean')
        di['concave points_mean'] = st.number_input('Concave Points Mean')
        di['symmetry_mean'] = st.number_input('Symmetry Mean')
        di['fractal_dimension_mean'] = st.number_input('Fractal Dimension Mean')
        di['radius_se'] = st.number_input('Radius SE')
        di['texture_se'] = st.number_input('Texture SE')
        di['perimeter_se'] = st.number_input('Perimeter SE')
        di['area_se'] = st.number_input('Area SE')
        di['smoothness_se'] = st.number_input('Smoothness SE')
    with right_column:
        di['compactness_se'] = st.number_input('Compactness SE')
        di['concavity_se'] = st.number_input('Concavity SE')
        di['concave points_se'] = st.number_input('Concave Points SE')
        di['symmetry_se'] = st.number_input('Symmetry SE')
        di['fractal_dimension_se'] = st.number_input('Fractal Dimension SE')
        di['radius_worst'] = st.number_input('Radius Worst')
        di['texture_worst'] = st.number_input('Texture Worst')
        di['perimeter_worst'] = st.number_input('Perimeter Worst')
        di['area_worst'] = st.number_input('Area Worst')
        di['smoothness_worst'] = st.number_input('Smoothness Worst')
        di['compactness_worst'] = st.number_input('Compactness Worst')
        di['concavity_worst'] = st.number_input('Concavity Worst')
        di['concave points_worst'] = st.number_input('Concave Points Worst')
        di['symmetry_worst'] = st.number_input('Symmetry Worst')
        di['fractal_dimension_worst'] = st.number_input('Fractal Dimension Worst')


    data = prepare_data_to_predict_from_uri(di)

    predict = predict_from_trained_model(data)

    predict = deconding_prediction(predict)

    if predict == 'Benign':
        st.success(predict)
    else:
        st.warning(predict)
else:
    st.warning('Train a Model!')
