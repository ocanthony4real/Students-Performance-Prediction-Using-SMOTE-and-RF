import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

# def load_model_1():
#     with open('saved_steps_1.pkl', 'rb') as file:
#         data_1 = pickle.load(file)
#     return data_1

# data_1 = load_model_1()

# rf_new_model = data_1["model"]
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
X = df_new.iloc[:,0:18]
y = df_new.iloc[:,18:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.3, shuffle=True)
y_train = np.array(y_train).flatten()
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
grade_0 = 0
for i in df_new['1st sem Grade remark']:
  if i == 0:
    grade_0 += 1
grade_0
grade_1 = 0
for i in df_new['1st sem Grade remark']:
  if i == 1:
    grade_1 += 1
grade_1
grade_2 = 0
for i in df_new['1st sem Grade remark']:
  if i == 2:
    grade_2 += 1
grade_2
grade_3 = 0
for i in df_new['1st sem Grade remark']:
  if i == 3:
    grade_3 += 1
grade_3
categories = ['Excellent', 'Good', 'Fair', 'Poor']
values = [grade_0, grade_1, grade_2, grade_3]

# wiith smote

df_newest_filtered = df_new
# df_newest_res = df_newest_res[df_newest_res['1st sem Grade remark']==2]
df_newest_filtered = df_newest_filtered[(df_newest_filtered['1st sem Grade remark'] == 0) | (df_newest_filtered['1st sem Grade remark'] == 1) | (df_newest_filtered['1st sem Grade remark'] == 2) | (df_newest_filtered['1st sem Grade remark'] == 3)]
X_newest = df_newest_filtered.iloc[:,0:18]
y_newest = df_newest_filtered.iloc[:,18:]
sm = SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = sm.fit_resample(X_newest, y_newest)
df_newest_res = pd.concat([X_res, y_res], axis = 1)
df_newest_dropped = df_new[(df_new['1st sem Grade remark'] != 0) & (df_new['1st sem Grade remark'] != 1) & (df_newest_filtered['1st sem Grade remark'] != 2) & (df_newest_filtered['1st sem Grade remark'] != 3)]
df_newest = pd.concat([df_newest_dropped, df_newest_res], axis = 0)
X = df_newest.iloc[:,0:18]
y = df_newest.iloc[:,18:]
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X, y, random_state = 0, test_size=0.3, shuffle=True)
y_train_res = np.array(y_train_res).flatten()
rf_new_model = RandomForestClassifier().fit(X_train_res, y_train_res)
y_pred_rf_res = rf_new_model.predict(X_test_res)
grade_0_res = 0
for i in df_newest['1st sem Grade remark']:
  if i == 0:
    grade_0_res += 1
grade_0_res
grade_1_res = 0
for i in df_newest['1st sem Grade remark']:
  if i == 1:
    grade_1_res += 1
grade_1_res
grade_2_res = 0
for i in df_newest['1st sem Grade remark']:
  if i == 2:
    grade_2_res += 1
grade_2_res
grade_3_res = 0
for i in df_newest['1st sem Grade remark']:
  if i == 3:
    grade_3_res += 1
grade_3_res
categories_res = ['Excellent', 'Good', 'Fair', 'Poor']
values_res = [grade_0_res, grade_1_res, grade_2_res, grade_3_res]

# def load_model_2():
#     with open('saved_steps_2.pkl', 'rb') as file:
#         data_2 = pickle.load(file)
#     return data_2

# data_2 = load_model_2()

# y_test = data_2["y_test"]
# y_pred_rf = data_2["y_pred_rf"]
# y_test_res = data_2["y_test_res"]
# y_pred_rf_res = data_2["y_pred_rf_res"]
# categories = data_2["categories"]
# categories_res = data_2["categories_res"]
# values = data_2["values"]
# values_res = data_2["values_res"]

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

    