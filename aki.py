from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
import xgboost as xgb  ###xgboost
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability prediction of postpartum depression risk", layout="centered")
plt.style.use('default')

df=pd.read_csv('AKI1_work_train_LASSO.csv',encoding='utf8')
trainy=df.AKI123
trainx=df.drop('AKI123',axis=1)
xgb = XGBClassifier(colsample_bytree=1,gamma=0.5,learning_rate=0.06,max_depth=2,
                    n_estimators =33,min_child_weight=3,subsample=1,
                    objective= 'binary:logistic',random_state = 1)
xgb.fit(trainx,trainy)

###side-bar
def user_input_features():
    st.title("Probability prediction of postpartum depression risk")
    st.sidebar.header('User input parameters below')
    a1=st.sidebar.selectbox('Smoking',('No','Yes'))
    a2=st.sidebar.selectbox('Intraoperative diuretics use',('No','Yes'))
    a3=st.sidebar.number_input('Norepinephrine')
    a4=st.sidebar.number_input('Autologous blood transfusion')
    a5=st.sidebar.number_input('Blood platelet transfusion')
    a6=st.sidebar.number_input('Intraoperative urine output')
    a7=st.sidebar.number_input('Fraction of hypotension during CPB')
    a8=st.sidebar.number_input('OUT_CPB_MAP_65 time')
    a9=st.sidebar.number_input('OUT_CPB_MAP_AUT_65')

    result=""
    if a1=="Yes":
        a1=1
    else: 
        a1=0 
    if a2=="Yes":
        a2=1
    else: 
        a2=0 
    output=[a1,a2,a3,a4,a5,a6,a7,a8,a9]
    int_features=[int(x) for x in output]
    final_features=np.array(int_features)
    patient1=pd.DataFrame(output)
    patient=pd.DataFrame(patient1.values.T,columns=trainx.columns)
    prediction=xgb.predict_proba(patient)
    prediction=float(prediction[:, 1])
    def predict_PPD():
        prediction=round(user_input_features[:, 1],3)
        return prediction
    result=""
    if st.button("Predict"):
        st.success('The probability of PPD for the mother: {:.1f}%'.format(prediction*100))
        if prediction>0.685:
            b="High risk"
        else:
            b="Low risk"
        st.success('The risk group: '+ b)
        explainer_xgb = shap.TreeExplainer(xgb)
        shap_values= explainer_xgb(patient)
        shap.plots.waterfall(shap_values[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Waterfall plot analysis of PPD for the mother:")
        st.pyplot(bbox_inches='tight')
        st.write("Abbreviations: EPDS, edinburgh postnatal depression scale; PPD, postpartum depression")
    if st.button("Reset"):
        st.write("")
    st.markdown("*Statement: this website will not record or store any information inputed.")
    st.write("2022 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
    st.write("✉ Contact Us: zoujianjun100@126.com")

if __name__ == '__main__':
    user_input_features()
