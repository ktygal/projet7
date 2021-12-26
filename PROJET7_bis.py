import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
#%matplotlib inline
#import warnings
#warnings.filterwarnings('ignore')
####
from PIL import Image
#chemin = 'C:\Users\katyg\Desktop\OPENCLASSROOMS\DATASCIENTISTE\Projet7'

#img = Image.open("C:/Users/katyg/PycharmProjects/pythonProject/\prêt à depenser.png")# en local
img = Image.open("image.png")# pour git
st.image(img, width=200)

#data = pd.read_csv("C:/Users/katyg/PycharmProjects/pythonProject/data_model.csv")# en local
data = pd.read_csv("data_model.csv")#git
#data.index=data['SK_ID_CURR']
#del data['SK_ID_CURR']
# modif
data['PAYMENT_RATE']=round(data['PAYMENT_RATE'],5)


X = pd.DataFrame(data, columns=['PAYMENT_RATE','DAYS_BIRTH','DAYS_EMPLOYED','AMT_ANNUITY'])
y = pd.DataFrame(data, columns = ['TARGET'])

st.title("Dashboard Bank Prediction")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
</div>
"""
status = st.radio("Select client situation: ", ('new_client', 'old_client'))


def user_input_features():
    if (status == 'new_client'):
        st.sidebar.header('New User Input Features')
        payment_rate = st.sidebar.slider('PAYMENT_RATE', X.PAYMENT_RATE.min(), X.PAYMENT_RATE.max())
        days_birth = st.sidebar.slider('DAYS_BIRTH', X.DAYS_BIRTH.min(), X.DAYS_BIRTH.max())
        days_employed = st.sidebar.slider('DAYS_EMPLOYED', X.DAYS_EMPLOYED.min(), X.DAYS_EMPLOYED.max())
        amt_annuity = st.sidebar.slider('AMT_ANNUITY', X.AMT_ANNUITY.min(), X.AMT_ANNUITY.max())
        df = {'PAYMENT_RATE': payment_rate,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'AMT_ANNUITY': amt_annuity}

        features = pd.DataFrame(df, index=[0])
        return features
        #
    else:
    #SELECT_ID_CLIENT = st.sidebar.slider("Choose your SK_ID_CURR", data['SK_ID_CURR'].min(),data['SK_ID_CURR'].max())
        SELECT_ID_CLIENT =  st.sidebar.selectbox("Choose your SK_ID_CURR",data['SK_ID_CURR'].unique())#min(),data['SK_ID_CURR'].max())
        st.write("your selection is SK_ID_CURR:", SELECT_ID_CLIENT)
        payment_rate = data['PAYMENT_RATE'].loc[data['SK_ID_CURR'] == SELECT_ID_CLIENT]
        days_birth = data['DAYS_BIRTH'].loc[data['SK_ID_CURR'] == SELECT_ID_CLIENT]
        days_employed = data['DAYS_EMPLOYED'].loc[data['SK_ID_CURR'] == SELECT_ID_CLIENT]
        amt_annuity = data['AMT_ANNUITY'].loc[data['SK_ID_CURR'] == SELECT_ID_CLIENT]
        df = {'PAYMENT_RATE': payment_rate,
                'DAYS_BIRTH': days_birth,
                'DAYS_EMPLOYED': days_employed,
                'AMT_ANNUITY': amt_annuity}
        features = pd.DataFrame(df)
        return features

input_df = user_input_features()

st.header('Specified input parameters')
st.write(input_df)
st.write('---')

load_clf = pickle.load(open('model_building_bis.pkl','rb')) #

prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.header('Prediction of input values')
if prediction == 0:
    st.write("solvable")
else:
    st.write("Non Solvable")
st.write('---')


st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Visualisation Prediction Probability')
a = prediction_proba[0,0]
b =prediction_proba[0,1]
x = [a, b]
colors = sns.dark_palette("palegoldenrod", 8, reverse=True) #skyblue
fig, ax = plt.subplots(figsize=(10, 4))
colors = sns.color_palette("Paired")
labels= ['Solvable','Non Solvable']
plt.title("Visualisation des prédictions")
plt.pie(x, labels= labels,colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')
plt.show()
st.write(fig)



st.subheader('Visualisation Distributions des Clients')
data_non = data[data['TARGET']==1]
data_solvable=data[data['TARGET']==0]
client = input_df['DAYS_EMPLOYED']
fig, ax = plt.subplots(figsize=(14, 6))
plt.subplot(2, 1, 1); sns.distplot(data_solvable.DAYS_EMPLOYED, label = 'DAYS_EMPLOYED')
plt.text(client,0.00005,"X client")
plt.title("Distribution de DAYS_EMPLOYED des clients solvable")
plt.subplot(2, 1, 2); sns.distplot(data_non.DAYS_EMPLOYED, label = 'DAYS_EMPLOYED')
plt.text(client,0.00005,"X client")
plt.title("Distribution de DAYS_EMPLOYED des clients non solvable")
plt.tight_layout()
plt.show()
st.write(fig)

fig2, ax = plt.subplots(figsize=(14, 6))
client = input_df['AMT_ANNUITY']
plt.subplot(2, 1, 1); sns.distplot(data_solvable.AMT_ANNUITY, label = 'AMT_ANNUITY')
plt.text(client,0.00001,"X client")
plt.title("Distribution de AMT_ANNUITY des clients solvable")
plt.subplot(2, 1, 2); sns.distplot(data_non.AMT_ANNUITY, label = 'AMT_ANNUITY')
plt.text(client,0.00001,"X client")
plt.title("Distribution de AMT_ANNUITY des clients non solvable")
plt.tight_layout()
plt.show()
st.write(fig2)



#streamlit run C:/Users/katyg/PycharmProjects/pythonProject/PROJET7_bis.py
