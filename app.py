import numpy as np
import pandas as pd
import streamlit as st
from sklearn import preprocessing
import pickle

model = pickle.load(open('model.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb'))
cols=['Customer_Age','Income_Category','Education_Level','Marital_Status','Gender','Card_Category','Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Ct_Chng_Q4_Q1',
      'Total_Relationship_Count	','Total_Amt_Chng_Q4_Q1','Avg_Utilization_Ratio','Credit_Limit','Avg_Open_To_Buy','Months_Inactive_12_mon','Months_on_book','Contacts_Count_12_mon',
      'Dependent_count']

def main():
    st.title("Churn Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Churn Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    Customer_Age = st.text_input("Customer Age","0")
    Income_Category = st.selectbox("Income Category", ['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'Less than $40K'])
    Education_Level = st.selectbox("Education Level",['College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate', 'Uneducated', 'Unknown'])
    Marital_Status = st.selectbox("Marital Status",['Divorced', 'Married', 'Single'])
    Gender = st.selectbox("Gender",['F', 'M'])
    Card_Category = st.selectbox("Card Category",['Blue', 'Gold', 'Platinum', 'Silver'])
    Total_Trans_Ct = st.text_input("Total Transactions Count","0")
    Total_Trans_Amt = st.text_input("Total Transactions Amount","0")
    Total_Revolving_Bal = st.text_input("Total Revolving Balance","0")
    Total_Ct_Chng_Q4_Q1 = st.text_input("Total Count Changes Q4 vs Q1","0")
    Total_Relationship_Count = st.text_input("Total Relationship Count","0")
    Total_Amt_Chng_Q4_Q1 = st.text_input("Total Amount Changes Q4 vs Q1","0")
    Avg_Utilization_Ratio = st.text_input("Average Utilization Ratio","0")
    Credit_Limit = st.text_input("Credit Limit","0")
    Avg_Open_To_Buy = st.text_input("Average Open to Buy","0")
    Months_Inactive_12_mon = st.text_input("Months Inactive in 12 months","0")
    Months_on_book = st.text_input("Months On Book","0")
    Contacts_Count_12_mon = st.text_input("Contacts Count in 12 months","0")
    Dependent_count = st.text_input("Dependent Count","0")

    if st.button("Predict"):
        features = [[Customer_Age,Income_Category,Education_Level,Marital_Status,Gender,Card_Category,Total_Trans_Ct,Total_Trans_Amt,Total_Revolving_Bal,Total_Ct_Chng_Q4_Q1,
      Total_Relationship_Count,Total_Amt_Chng_Q4_Q1,Avg_Utilization_Ratio,Credit_Limit,Avg_Open_To_Buy,Months_Inactive_12_mon,Months_on_book,Contacts_Count_12_mon,
      Dependent_count]]
        data = {'Customer_Age': int(Customer_Age), 'Income_Category': Income_Category, 'Education_Level': Education_Level, 'Marital_Status': Marital_Status, 'Gender': Gender,
                'Card_Category': Card_Category, 'Total_Trans_Ct': int(Total_Trans_Ct), 'Total_Trans_Amt': float(Total_Trans_Amt), 'Total_Revolving_Bal': int(Total_Revolving_Bal),
                'Total_Ct_Chng_Q4_Q1': float(Total_Ct_Chng_Q4_Q1), 'Total_Relationship_Count': int(Total_Relationship_Count), 'Total_Amt_Chng_Q4_Q1': float(Total_Amt_Chng_Q4_Q1),
                'Avg_Utilization_Ratio': float(Avg_Utilization_Ratio), 'Credit_Limit': float(Credit_Limit), 'Avg_Open_To_Buy': float(Avg_Open_To_Buy), 'Months_Inactive_12_mon': int(Months_Inactive_12_mon),
                'Months_on_book': int(Months_on_book), 'Contacts_Count_12_mon': int(Contacts_Count_12_mon), 'Dependent_count': int(Dependent_count)}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['Customer_Age','Income_Category','Education_Level','Marital_Status','Gender','Card_Category','Total_Trans_Ct','Total_Trans_Amt','Total_Revolving_Bal','Total_Ct_Chng_Q4_Q1',
      'Total_Relationship_Count	','Total_Amt_Chng_Q4_Q1','Avg_Utilization_Ratio','Credit_Limit','Avg_Open_To_Buy','Months_Inactive_12_mon','Months_on_book','Contacts_Count_12_mon', 'Dependent_count'])

        category_col =['Income_Category','Education_Level','Marital_Status','Gender','Card_Category']
        for cat in encoder_dict:
            for col in df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = pd.Series(encoder_dict[cat])
                    for unique_item in df[col].unique():
                        if unique_item not in le.classes_:
                            df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
                    df[col] = le.transform(df[col])

        features_list = df.values.tolist()
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 1:
            text = "Active"
        else:
            text = "Churn"

        st.success('The Customer is likely to {}'.format(text))

if __name__=='__main__':
    main()
