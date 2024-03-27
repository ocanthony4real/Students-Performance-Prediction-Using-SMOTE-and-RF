import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle

def load_model_2():
    with open('saved_steps_2.pkl', 'rb') as file:
        data_2 = pickle.load(file)
    return data_2

data_2 = load_model_2()

y_test = data_2["y_test"]
y_pred_rf = data_2["y_pred_rf"]
y_test_res = data_2["y_test_res"]
y_pred_rf_res = data_2["y_pred_rf_res"]
categories = data_2["categories"]
categories_res = data_2["categories_res"]
values = data_2["values"]
values_res = data_2["values_res"]

def load_data():
    df = pd.read_csv('dataset.csv')

    df_new = df.drop(axis=1, columns=['Curricular units 1st sem (credited)',
        'Previous qualification',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (without evaluations)','International','Application mode', 'Application order',  'Target'])

    conditions = [
    (df_new['Curricular units 1st sem (grade)'] >= 14.0),
    (df_new['Curricular units 1st sem (grade)'] >= 11.0) & (df_new['Curricular units 1st sem (grade)'] < 14.0),
    (df_new['Curricular units 1st sem (grade)'] >= 5.0) & (df_new['Curricular units 1st sem (grade)'] < 11.0),
    (df_new['Curricular units 1st sem (grade)'] >= 0.0) & (df_new['Curricular units 1st sem (grade)'] < 5.0),
    ]

    choices = [0, 1, 2, 3]

    df_new['1st sem Grade remark'] = np.select(conditions, choices, default='null')

    df_new['1st sem Grade remark'] = df_new['1st sem Grade remark'].astype(int)

    df_new = df_new.drop(axis=1, columns=['Curricular units 2nd sem (grade)','Curricular units 1st sem (grade)'])

    return df_new

df = load_data()

#with SMOTE
df_newest_filtered = df
df_newest_filtered = df_newest_filtered[(df_newest_filtered['1st sem Grade remark'] == 0) | (df_newest_filtered['1st sem Grade remark'] == 1) | (df_newest_filtered['1st sem Grade remark'] == 2) | (df_newest_filtered['1st sem Grade remark'] == 3)]
X_newest = df_newest_filtered.iloc[:,0:18]
y_newest = df_newest_filtered.iloc[:,18:]
sm = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = sm.fit_resample(X_newest, y_newest)
df_newest_res = pd.concat([X_res, y_res], axis = 1)
f_newest_dropped = df[(df['1st sem Grade remark'] != 0) & (df['1st sem Grade remark'] != 1) & (df_newest_filtered['1st sem Grade remark'] != 2) & (df_newest_filtered['1st sem Grade remark'] != 3)]


def show_explore_page():
    st.title("Key Analysis of Model")

    st.write(
        """
    ### Data distribution before resampling
        """  
    )
    
    fig, ax = plt.subplots()
    
    ax.bar(categories, values)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    st.pyplot(fig)

    st.write(
        """
    ### Data distribution after resampling
        """  
    )
    fig, ax = plt.subplots()
    
    ax.bar(categories_res, values_res)
    plt.xlabel('Categories')
    plt.ylabel('Values')
    st.pyplot(fig)

    st.write(
        """
    ### Classification report before resampling of data
    """
    )
    target_names = ["Excellent", "Good", "Fair", "Poor"]

    st.dataframe(
    pd.DataFrame(
        classification_report(y_test, y_pred_rf, target_names=target_names, output_dict=True)
    ).transpose()
    )
    

    st.write(
        """
    ### Classification report after resampling of data
    """
    )
    target_names = ["Excellent", "Good", "Fair", "Poor"]

    st.dataframe(
    pd.DataFrame(
        classification_report(y_test_res, y_pred_rf_res, target_names=target_names, output_dict=True)
    ).transpose()
    )

    