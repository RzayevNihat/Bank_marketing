import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
df=pd.read_csv('bank2.csv')

    
with open(file='bank_model.pickle',mode='rb') as pickled_model:
    model=pickle.load(file=pickled_model)    
    
bank_image=Image.open(fp='bank_marketing_image.jpg')

st.dataframe(df)

label_encoder = LabelEncoder()

marital_encoding = label_encoder.fit_transform(df['marital'])
marital_mapping = dict(zip(df['marital'], marital_encoding))

education_encoding = label_encoder.fit_transform(df['education'])
education_mapping = dict(zip(df['education'], education_encoding))

month_encoding = label_encoder.fit_transform(df['month'])
month_mapping = dict(zip(df['month'], month_encoding))

day_of_week_encoding = label_encoder.fit_transform(df['day_of_week'])
day_of_week_mapping = dict(zip(df['day_of_week'], day_of_week_encoding))

p_outcome_encoding = label_encoder.fit_transform(df['p_outcome'])
p_outcome_mapping = dict(zip(df['p_outcome'], p_outcome_encoding))

age_categories_encoding = label_encoder.fit_transform(df['age_categories'])
age_categories_mapping = dict(zip(df['age_categories'], age_categories_encoding))

job_categories_encoding = label_encoder.fit_transform(df['job'])
job_categories_mapping = dict(zip(df['job'], job_categories_encoding))

contact_encoding = label_encoder.fit_transform(df['contact'])
contact_mapping = dict(zip(df['contact'], contact_encoding))

loan_encoding = label_encoder.fit_transform(df['loan'])
loan_mapping = dict(zip(df['loan'], loan_encoding))

housing_encoding = label_encoder.fit_transform(df['housing'])
housing_mapping = dict(zip(df['housing'], housing_encoding))


rep_edu={'basic_4y':'Basic 4 year',
         'high_school':'High school',
         'basic_6y':'Basic 6 years',
         'basic_9y':'Basic 9 years',
         'professional_course':'Professional course',
         'unknown':'Unknown',
         'university_degree':'University degree',
         'illiterate':'Illiterate'}

df['education'] = df['education'].map(rep_edu)

interface=st.container()

sidebar=st.sidebar.container()

with interface:
    st.title(body='Bank Marketing')
    st.markdown('***')
    
    st.header(body='Project Description')
    
    st.text(body="""
In recent years, the banking industry has become increasingly competitive, 
with banks vying for customers in a crowded marketplace. As a result, it 
has become more important than ever for banks to develop effective marketing 
strategies that target the right customers with the right messages. One key 
challenge that banks face is determining which customers are most likely to 
subscribe to specific products, such as term deposits, which are a type of 
savings account with a fixed term and interest rate.
    """)
    st.markdown(body='***')
    st.markdown("### Whether the customer subscribed to a term deposit")
    st.write(px.pie(data_frame=df,names='deposit'))
    st.markdown(body='***')
    st.subheader(body='Input Features')
    
    
    marital,education,cons_price_idx,month=st.columns(spec=[1,1,1,1])
    with marital:
        marital=st.selectbox(label='Marital Status',options=df['marital'].str.capitalize().unique())
    
    with education:
        education=st.selectbox(label='Education Level',options=df['education'].unique())
        
    st.markdown('***')

    
    with month:
        month=st.selectbox(label='Last Contact Month',
                           options=df['month'].str.capitalize().unique())
    
    with cons_price_idx:
        cons_price_idx=st.selectbox(label='Consumer Price Index',options=df['cons_price_idx'].sort_values().unique())
            

    day_of_week,p_outcome,job_categories,age_categories=st.columns(spec=[1,1,1,1])
    
    with day_of_week:
        day_of_week=st.selectbox(label='Last Contact Day',options=df['day_of_week'].str.capitalize().unique())
    
    with p_outcome:
        p_outcome=st.selectbox(label='Previous Outcome',options=df['p_outcome'].str.capitalize().unique())                   
                           
    
    with job_categories:
        job_categories=st.selectbox(label='Job of Customer',options=df['job'].str.capitalize().unique())

    with age_categories:
        age_categories=st.selectbox(label='Category of age',options=df['age_categories'].unique())
    st.markdown('***')
    housing,loan,contact,p_days=st.columns(spec=[1,1,1,1])
    with housing:                          
        housing=st.radio(label='The Customer has a Housing Loan',options=df['housing'].unique(),horizontal=True)
    
    with loan:                         
        loan=st.radio(label='The customer has a personal loan',options=df['loan'].unique(),horizontal=True) 
        
    with contact:
        contact=st.radio(label='Contact Type',options=df['contact'].str.capitalize().unique())
    with p_days:
        p_days=st.selectbox(label='Contact Type',options=df['p_days'].unique())
    st.markdown('***')
    
    emp_var_rate,previous,duration,campaign=st.columns(spec=[1,1,1,1])
    with emp_var_rate:
        emp_var_rate=st.selectbox(label='Employment Variation Rate',options=df['emp_var_rate'].sort_values().unique())


    
    with previous:
        previous=st.selectbox(label='Number of Previous',options=df['previous'].unique())
    st.write('\n')        
    #st.markdown(body='***')
    
    with duration:
            duration=st.number_input(label='Last Contact Duration',min_value=df.duration.min(),max_value=df.duration.max(),value=int(df.duration.mean()))
    

    with campaign:
        campaign=st.number_input(label='Number of Campaign ',min_value=df.campaign.min(),max_value=df.campaign.max(),value=int(df.campaign.mean()))

                       
    euribor_3m=st.slider(label='Euribor 3-Month Rate (Daily)',min_value=df.euribor_3m.min(),max_value=df.euribor_3m.max(),value=4.)
    cons_conf_idx=st.slider(label='Euribor 3-Month Rate (Daily)',min_value=df.cons_conf_idx.min(),max_value=df.cons_conf_idx.max())                                  
    nr_employed=st.slider(label='Euribor 3-Month Rate (Daily)',min_value=df.nr_employed.min(),max_value=df.nr_employed.max())                   
    marital_encoded = marital_mapping.get(marital, 0)
    education_encoded = education_mapping.get(education, 0)
    age_categories_encoded = age_categories_mapping.get(age_categories, 0)
    job_categories_encoded = job_categories_mapping.get(job_categories, 0)
    month_encoded = month_mapping.get(month, 0)
    p_outcome_encoded = p_outcome_mapping.get(p_outcome, 0)
    day_of_week_encoded = day_of_week_mapping.get(day_of_week, 0)
    contact_encoded = contact_mapping.get(contact, 0)
    loan_encoded=loan_mapping.get(loan,0)
    housing_encoded=housing_mapping.get(housing,0)
    duration = int(duration)
    campaign = int(campaign)
    previous = int(previous)
    emp_var_rate = int(emp_var_rate)
    cons_price_idx = int(cons_price_idx)
    euribor_3m = float(euribor_3m)
    
    
    input_features = pd.DataFrame({'job_categories':[job_categories_encoded],
                       'marital':[marital_encoded],
                       'education':[education_encoded],
                       'housing':[housing_encoded],
                       'loan':[loan_encoded],
                       'contact':[contact_encoded],
                       'month':[month_encoded],
                       'day_of_week':[day_of_week_encoded],
                       'duration':[duration],
                       'campaign':[campaign],
                       'p_days':[p_days],
                       'previous':[previous],
                       'p_outcome':[p_outcome_encoded],
                       'emp_var_rate':[emp_var_rate],
                       'cons_price_idx':[cons_price_idx],
                       'cons_conf_idx':[cons_conf_idx],
                       'euribor_3m':[euribor_3m],
                       'nr_employed':[nr_employed],
                       'age_categories':[age_categories_encoded]},index=[0],columns=model.feature_names_in_)
    


    st.markdown('***')
    st.subheader(body='Model Prediction')
    
    if st.button('Predict'):
        
        deposit=model.predict(X=input_features)
        
        with st.spinner(text='Sending input features to model...'):
            
            time.sleep(2)
            
            st.success('Your prediction is ready!')
            st.markdown(f'Model output: The Customer Subscribed to a Term Deposit: **{deposit}**')
            
with sidebar:

    st.title(body = 'Variable Dictionary')
    
    st.markdown(body = '- **age** - The age of the customer (numeric)')
    st.markdown(body = '- **job** - The type of job the customer has (categorical)')
    st.markdown(body = '- **marital** - The marital status of the customer (categorical)')
    st.markdown(body = '- **education** - The level of education of the customer (categorical)')
    st.markdown(body = '- **housing** - Whether the customer has a housing loan (categorical)')
    st.markdown(body = '- **loan** - Whether the customer has a personal loan (categorical)')
    st.markdown(body = '- **contact** - The contact communication type (categorical)')
    st.markdown(body = '- **month** - The month of the year when the customer was last contacted (categorical)')
    st.markdown(body = '- **day_of_week** - Last contact day of the week (categorical)')
    st.markdown(body = '- **duration** - The duration of the last contact in seconds (numeric)')
    st.markdown(body = '- **campaign** - The number of contacts performed during this campaign and for this customer (numeric)')                   
    st.markdown(body = '- **previous** - The number of contacts performed before this campaign and for this customer (numeric)')
    st.markdown(body = '- **p_outcome** - The outcome of the previous marketing campaign (categorical)')
    st.markdown(body = '- **emp_var_rate** - Employment variation rate - quarterly indicator (numeric)')
    st.markdown(body = '- **cons_price_idx** - Consumer price index - monthly indicator (numeric)')                  
    st.markdown(body = '- **euribor_3m** - Euribor 3 months rate - daily indicator (numeric)')                   
                       
                       
                      