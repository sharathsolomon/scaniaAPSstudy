#Streamlit Deployment code for ML Model

import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import time
from PIL import Image


st.title("Scania Trucks APS Failure Prediction")
image = Image.open('scania.jpg')
st.image(image, caption='Scania Trucks')

st.text('This app will predict whether there is failure in the Air Pressure System of Scania Trucks')
st.text('You can upload the data below :')


col1,col2 = st.beta_columns(2)
data = col1.file_uploader('Upload the csv file below',type=['csv'])

col1,col2 = st.beta_columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')


def final_fun_1(X):
  X = X.replace('na',np.NaN)
  X = X.astype(float)
  X = X.drop('cd_000',axis=1)

  features_tobe_eliminated =  ['br_000', 'bq_000', 'bp_000', 'bo_000', 'ab_000', 'cr_000', 'bn_000', 'bm_000']
  median_imp_features = ['ec_00', 'cm_000', 'cl_000', 'ed_000', 'ak_000', 'ca_000', 'dm_000', 'df_000', 'dg_000', 'dh_000', 'dl_000', 'dj_000', 'dk_000', 'eb_000', 'di_000', 'ac_000', 'bx_000', 'cc_000', 'bd_000', 'ds_000', 'dt_000', 'dp_000', 'dq_000', 'dr_000', 'du_000', 'dv_000', 'bc_000', 'cp_000', 'de_000', 'do_000', 'dy_000', 'ef_000', 'ar_000', 'bz_000', 'dx_000', 'dz_000', 'ea_000', 'eg_000', 'be_000', 'dd_000', 'ce_000', 'ax_000', 'ae_000', 'af_000', 'av_000', 'bf_000', 'bs_000', 'cb_000', 'bu_000', 'bv_000', 'cq_000', 'dn_000', 'ba_000', 'ba_001', 'ba_002', 'ba_003', 'ba_004', 'ba_005', 'ba_006', 'ba_007', 'ba_008', 'ba_009', 'cn_000', 'cn_001', 'cn_002', 'cn_003', 'cn_004', 'cn_005', 'cn_006', 'cn_007', 'cn_008', 'cn_009', 'ag_000', 'ag_001', 'ag_002', 'ag_003', 'ag_004', 'ag_005', 'ag_006', 'ag_007', 'ag_008', 'ag_009', 'ay_000', 'ay_001', 'ay_002', 'ay_003', 'ay_004', 'ay_005', 'ay_006', 'ay_007', 'ay_008', 'ay_009', 'az_000', 'az_001', 'az_002', 'az_003', 'az_004', 'az_005', 'az_006', 'az_007', 'az_008', 'az_009', 'ee_000', 'ee_001', 'ee_002', 'ee_003', 'ee_004', 'ee_005', 'ee_006', 'ee_007', 'ee_008', 'ee_009', 'cs_000', 'cs_001', 'cs_002', 'cs_003', 'cs_004', 'cs_005', 'cs_006', 'cs_007', 'cs_008', 'cs_009', 'ah_000', 'bb_000', 'al_000', 'an_000', 'ap_000', 'bg_000', 'bh_000', 'ai_000', 'aj_000', 'am_0', 'as_000', 'at_000', 'au_000', 'ao_000', 'aq_000', 'bi_000', 'bj_000', 'by_000', 'ci_000', 'cj_000', 'ck_000', 'bt_000', 'aa_000']

  median_imputer = pickle.load(open('median_imputer.pkl', 'rb'))
  X_median = median_imputer.transform(X[median_imp_features])
  X[median_imp_features] = X_median
  X = X.drop(features_tobe_eliminated,axis=1)

  scaler = pickle.load(open('normalizer.pkl', 'rb'))
  X_scaled = scaler.transform(X)
  X_mice = pd.DataFrame(X_scaled, columns= X.columns)

  mice_imputer = pickle.load(open('mice_imputer.pkl', 'rb'))
  X_imputed = mice_imputer.transform(X_mice)
  X_imputed = pd.DataFrame(X_imputed,columns=X_mice.columns)

  encoder = load_model('mice_encoder.h5')
  X_encoded = encoder.predict(X_imputed)
  X_final = np.hstack((np.array(X_imputed),X_encoded))

  best_model = pickle.load(open('best_model.pkl','rb'))
  y_prob = best_model.predict_proba(X_final)[:,1]

  y_pred=[]
  for i in y_prob:
    if i>=0.005551333393319321:
      y_pred.append('Failure')
    else:
      y_pred.append('No Failure')
  return y_pred


if predict_button:
    if data is not None:
        df = pd.read_csv(data)
        st.text('Uploaded Data :')
        st.dataframe(df)
        start = time.time()
        y_pred = final_fun_1(df)
        datapoints = np.arange(1,len(y_pred)+1)
        df_pred = pd.DataFrame()
        df_pred['Datapoint'] = datapoints
        df_pred['Prediction'] = y_pred
        st.text('Predictions :')
        st.dataframe(df_pred)
        end = time.time()
        st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    else:
        st.text('Please upload Data')
        
elif test_data:
    test_file = pd.read_csv('sample_data.csv')
    sample = test_file.sample(n=10)
    sample.reset_index(inplace=True,drop=True)
    st.text('Sample Data :')
    st.dataframe(sample)
    start = time.time()
    y_pred = final_fun_1(sample)
    datapoints = np.arange(1,len(y_pred)+1)
    df_pred = pd.DataFrame()
    df_pred['Datapoint'] = datapoints
    df_pred['Prediction'] = y_pred
    st.text('Predictions :')
    st.dataframe(df_pred)
    end = time.time()
    st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    
    
    
    
        
        
        