import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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


def show_predict_page():
    st.title("Students' Performance Prediction")
    Marital_Status = (
        "Single", "Married", "Widower", "Divorced", "Facto union", "Legally separated"
    )

    courses = (
    "Biofuel Production Technologies",
    "Animation and Multimedia Design",
    "Social Service (evening attendance)",
    "Agronomy",
    "Communication Design",
    "Veterinary Nursing",
    "Informatics Engineering",
    "Equiniculture",
    "Management",
    "Social Service",
    "Tourism",
    "Nursing",
    "Oral Hygiene",
    "Advertising and Marketing Management",
    "Journalism and Communication",
    "Basic Education",
    "Management (evening attendance)Table"
    )

    Daytime_evening_attendance = (
    "daytime",
    "evening"
    )
    Nationality = (
    "Portuguese",
    "German",
    "Spanish",
    "Italian",
    "Dutch",
    "English",
    "Lithuanian",
    "Angolan",
    "Cape Verdean",
    "Guinean",
    "Mozambican",
    "Santomean",
    "Turkish",
    "Brazilian",
    "Romanian",
    "Moldova (Republic of)",
    "Mexican",
    "Ukrainian",
    "Russian",
    "Cuban",
    "Colombian"
    )

    Mother_qualification = (
    "Secondary Education—12th Year of Schooling or Equivalent",
    "Higher Education—bachelor’s degree",
    "Higher Education—degree",
    "Higher Education—master’s degree",
    "Higher Education—doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling—not completed",
    "11th Year of Schooling—not completed9",
    "7th Year (Old)",
    "Other—11th Year of Schooling",
    "2nd year complementary high school course",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
    "15—Complementary High School Course",
    "Technical-professional course",
    "Complementary High School Course—not concluded",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling—not completed",
    "8th year of schooling",
    "General Course of Administration and Commerce",
    "Supplementary Accounting and Administration",
    "Unknown",
    "Cannot read or write",
    "Can read without having a 4th year of schooling",
    "27—Basic education 1st cycle (4th/5th year) or equivalent",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
    "29—Technological specialization course",
    "Higher education—degree (1st cycle)",
    "31—Specialized higher studies course",
    "32—Professional higher technical course",
    "Higher Education—master’s degree (2nd cycle)",
    "Higher Education—doctorate (3rd cycle)"
    )

    Father_qualification = (
    "Secondary Education—12th Year of Schooling or Equivalent",
    "Higher Education—bachelor’s degree",
    "Higher Education—degree",
    "Higher Education—master’s degree",
    "Higher Education—doctorate",
    "Frequency of Higher Education",
    "12th Year of Schooling—not completed",
    "11th Year of Schooling—not completed9",
    "7th Year (Old)",
    "Other—11th Year of Schooling",
    "2nd year complementary high school course",
    "10th Year of Schooling",
    "General commerce course",
    "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
    "15—Complementary High School Course",
    "Technical-professional course",
    "Complementary High School Course—not concluded",
    "7th year of schooling",
    "2nd cycle of the general high school course",
    "9th Year of Schooling—not completed",
    "8th year of schooling",
    "General Course of Administration and Commerce",
    "Supplementary Accounting and Administration",
    "Unknown",
    "Cannot read or write",
    "Can read without having a 4th year of schooling",
    "27—Basic education 1st cycle (4th/5th year) or equivalent",
    "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
    "29—Technological specialization course",
    "Higher education—degree (1st cycle)",
    "31—Specialized higher studies course",
    "32—Professional higher technical course",
    "Higher Education—master’s degree (2nd cycle)",
    "Higher Education—doctorate (3rd cycle)"
    )

    Mother_occupation = (
    "Student",
    "Representatives of the Legislative Power and Executive Bodies,Directors, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers, and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries,and Forestry",
    "Skilled Workers in Industry, Construction, and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Armed Forces Officers",
    "Armed Forces Sergeants",
    "Other Armed Forces personnel",
    "Directors of administrative and commercial services",
    "Hotel, catering, trade, and other services directors",
    "Specialists in the physical sciences, mathematics, engineering,and related techniques",
    "Health professionals",
    "Teachers",
    "Specialists in finance, accounting, administrative organization,and public and commercial relations",
    "Intermediate level science and engineering techniciansand professions",
    "Technicians and professionals of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural,and similar services",
    "Information and communication technology technicians",
    "Office workers, secretaries in general,and data processing operators",
    "Data, accounting, statistical, financial services, andregistry-related operators",
    "Other administrative support staff",
    "Personal service workers",
    "Sellers",
    "Personal care workers and the like",
    "Protection and security services personnel",
    "Market-oriented farmers and skilled agricultural and animalproduction workers",
    "Farmers, livestock keepers, fishermen, hunters and gatherers,and subsistence",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in metallurgy, metalworking, and similar",
    "Skilled workers in electricity and electronics",
    "Workers in food processing, woodworking, and clothing andother industries and crafts",
    "Fixed plant and machine operators",
    "Assembly workers",
    "Vehicle drivers and mobile equipment operators",
    "Unskilled workers in agriculture, animal production, andfisheries and forestry",
    "Unskilled workers in extractive industry, construction,manufacturing, and transport",
    "Meal preparation assistants"
    "Street vendors (except food) and street service providers"
    )
    Father_occupation = (
    "Student",
    "Representatives of the Legislative Power and Executive Bodies,Directors, Directors and Executive Managers",
    "Specialists in Intellectual and Scientific Activities",
    "Intermediate Level Technicians and Professions",
    "Administrative staff",
    "Personal Services, Security and Safety Workers, and Sellers",
    "Farmers and Skilled Workers in Agriculture, Fisheries,and Forestry",
    "Skilled Workers in Industry, Construction, and Craftsmen",
    "Installation and Machine Operators and Assembly Workers",
    "Unskilled Workers",
    "Armed Forces Professions",
    "Other Situation",
    "(blank)",
    "Armed Forces Officers",
    "Armed Forces Sergeants",
    "Other Armed Forces personnel",
    "Directors of administrative and commercial services",
    "Hotel, catering, trade, and other services directors",
    "Specialists in the physical sciences, mathematics, engineering,and related techniques",
    "Health professionals",
    "Teachers",
    "Specialists in finance, accounting, administrative organization,and public and commercial relations",
    "Intermediate level science and engineering techniciansand professions",
    "Technicians and professionals of intermediate level of health",
    "Intermediate level technicians from legal, social, sports, cultural,and similar services",
    "Information and communication technology technicians",
    "Office workers, secretaries in general,and data processing operators",
    "Data, accounting, statistical, financial services, andregistry-related operators",
    "Other administrative support staff",
    "Personal service workers",
    "Sellers",
    "Personal care workers and the like",
    "Protection and security services personnel",
    "Market-oriented farmers and skilled agricultural and animalproduction workers",
    "Farmers, livestock keepers, fishermen, hunters and gatherers,and subsistence",
    "Skilled construction workers and the like, except electricians",
    "Skilled workers in metallurgy, metalworking, and similar",
    "Skilled workers in electricity and electronics",
    "Workers in food processing, woodworking, and clothing andother industries and crafts",
    "Fixed plant and machine operators",
    "Assembly workers",
    "Vehicle drivers and mobile equipment operators",
    "Unskilled workers in agriculture, animal production, andfisheries and forestry",
    "Unskilled workers in extractive industry, construction,manufacturing, and transport",
    "Meal preparation assistants"
    "Street vendors (except food) and street service providers" 
    )
    Displaced = (
        "yes", "no"
    )
    Educational_special_needs = (
        "yes", "no"
    )
    Debtor = (
        "yes", "no"
    )
    Tuition_fees_up_to_date = (
        "yes", "no"
    )
    Gender = (
        "male", "female"
    )
    Scholarship_holder = (
        "yes", "no"
    )

    Marital_Status  = st.selectbox("Marital Status", Marital_Status)
    
    courses = st.selectbox("Courses", courses)
    Daytime_evening_attendance = st.selectbox("Daytime/evening attendance", Daytime_evening_attendance)
    Nationality = st.selectbox("Nationality", Nationality)
    Mother_qualification = st.selectbox("Mother's qualification", Mother_qualification)
    Father_qualification = st.selectbox("Father's qualification", Father_qualification)
    Mother_occupation = st.selectbox("Mother's occupation", Mother_occupation)
    Father_occupation = st.selectbox("Father's occupation", Father_occupation)
    Displaced = st.selectbox("Displaced", Displaced)
    Educational_special_needs = st.selectbox("Educational special needs", Educational_special_needs)
    Debtor = st.selectbox("Debtor", Debtor)
    Tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", Tuition_fees_up_to_date)
    Gender = st.selectbox("Gender", Gender)
    Scholarship_holder = st.selectbox("Scholarship holder", Scholarship_holder)
    Age_at_enrollment = st.number_input("Age at enrollment")
    Unemployment_rate = st.number_input("Unemployment rate")
    Inflation_rate = st.number_input("Inflation rate")
    GDP = st.number_input("GDP")


    predict = st.button("Predict student's grade")

    if predict:
        prediction = np.array([[Marital_Status, courses, Daytime_evening_attendance, Nationality, Mother_qualification, Father_qualification, Mother_occupation, Father_occupation, Displaced, Educational_special_needs, Debtor, Tuition_fees_up_to_date, Gender, Scholarship_holder, Age_at_enrollment, Unemployment_rate, Inflation_rate, GDP ]])
        if prediction[:, 0]:
            conditions = [prediction[:, 0] == "Single", prediction[:, 0] == "Married", prediction[:, 0] == "Widower", prediction[:, 0] == "Divorced",prediction[:, 0] == "Facto union",prediction[:, 0] == "Legally separated"]
            choices = [0, 1, 2, 3, 4, 5]
            prediction[:, 0] = np.select(conditions, choices, default='null')
            prediction[:, 0] = prediction[:, 0].astype(int)
        if prediction[:, 1]:
            conditions = [prediction[:, 1]=="Biofuel Production Technologies", prediction[:, 1]=="Animation and Multimedia Design", prediction[:, 1]=="Social Service (evening attendance)", prediction[:, 1]=="Agronomy", prediction[:, 1]=="Communication Design", prediction[:, 1]=="Veterinary Nursing", prediction[:, 1]=="Informatics Engineering", prediction[:, 1]=="Equiniculture", prediction[:, 1]=="Management", prediction[:, 1]=="Social Service", prediction[:, 1]=="Tourism", prediction[:, 1]=="Nursing", prediction[:, 1]=="Oral Hygiene", prediction[:, 1]=="Advertising and Marketing Management", prediction[:, 1]=="Journalism and Communication", prediction[:, 1]=="Basic Education", prediction[:, 1]=="Management (evening attendance)Table"]            
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            prediction[:, 1] = np.select(conditions, choices, default='null')
            prediction[:, 1] = prediction[:, 1].astype(int)
        if prediction[:, 2]:
            conditions =  [prediction[:, 2]=="daytime", prediction[:, 2]=="evening"]
            choices = [1, 0]
            prediction[:, 2] = np.select(conditions, choices, default='null')
            prediction[:, 2] = prediction[:, 2].astype(int)
        if prediction[:, 3]:
            conditions = [prediction[:, 3]=="Portuguese", prediction[:, 3]=="German", prediction[:, 3]=="Spanish", prediction[:, 3]=="Italian", prediction[:, 3]=="Dutch", prediction[:, 3]=="English", prediction[:, 3]=="Lithuanian", prediction[:, 3]=="Angolan", prediction[:, 3]=="Cape Verdean", prediction[:, 3]=="Guinean", prediction[:, 3]=="Mozambican", prediction[:, 3]=="Santomean", prediction[:, 3]=="Turkish", prediction[:, 3]=="Brazilian", prediction[:, 3]=="Romanian", prediction[:, 3]=="Moldova (Republic of)", prediction[:, 3]=="Mexican", prediction[:, 3]=="Ukrainian", prediction[:, 3]=="Russian", prediction[:, 3]=="Cuban", prediction[:, 3]=="Colombian"]
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            prediction[:, 3] = np.select(conditions, choices, default='null')
            prediction[:, 3] = prediction[:, 3].astype(int)
        if prediction[:, 4]:
            conditions =  [prediction[:, 4]=="Secondary Education—12th Year of Schooling or Equivalent", prediction[:, 4]=="Higher Education—bachelor’s degree", prediction[:, 4]=="Higher Education—degree", prediction[:, 4]=="Higher Education—master’s degree", prediction[:, 4]=="Higher Education—doctorate", prediction[:, 4]=="Frequency of Higher Education", prediction[:, 4]=="12th Year of Schooling—not completed", prediction[:, 4]=="11th Year of Schooling—not completed9", prediction[:, 4]=="7th Year (Old)", prediction[:, 4]=="Other—11th Year of Schooling", prediction[:, 4]=="2nd year complementary high school course", prediction[:, 4]=="10th Year of Schooling", prediction[:, 4]=="General commerce course", prediction[:, 4]=="Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent", prediction[:, 4]=="15—Complementary High School Course",
                            prediction[:, 4]=="Technical-professional course", prediction[:, 4]=="Complementary High School Course—not concluded", prediction[:, 4]=="7th year of schooling", prediction[:, 4]=="2nd cycle of the general high school course", prediction[:, 4]=="9th Year of Schooling—not completed", prediction[:, 4]=="8th year of schooling", prediction[:, 4]=="General Course of Administration and Commerce", prediction[:, 4]=="Supplementary Accounting and Administration", prediction[:, 4]=="Unknown", prediction[:, 4]=="Cannot read or write", prediction[:, 4]=="Can read without having a 4th year of schooling", prediction[:, 4]=="27—Basic education 1st cycle (4th/5th year) or equivalent", prediction[:, 4]=="Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent", prediction[:, 4]=="29—Technological specialization course", prediction[:, 4]=="Higher education—degree (1st cycle)", prediction[:, 4]=="31—Specialized higher studies course", prediction[:, 4]=="32—Professional higher technical course", prediction[:, 4]=="Higher Education—master’s degree (2nd cycle)", prediction[:, 4]=="Higher Education—doctorate (3rd cycle)"]
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            prediction[:, 4] = np.select(conditions, choices, default='null')
            prediction[:, 4] = prediction[:, 4].astype(int)
        if prediction[:, 5]:
            conditions =  [prediction[:, 5]=="Secondary Education—12th Year of Schooling or Equivalent", prediction[:, 5]=="Higher Education—bachelor’s degree", prediction[:, 5]=="Higher Education—degree", prediction[:, 5]=="Higher Education—master’s degree", prediction[:, 5]=="Higher Education—doctorate", prediction[:, 5]=="Frequency of Higher Education", prediction[:, 5]=="12th Year of Schooling—not completed", prediction[:, 5]=="11th Year of Schooling—not completed9", prediction[:, 5]=="7th Year (Old)", prediction[:, 5]=="Other—11th Year of Schooling", prediction[:, 5]=="2nd year complementary high school course", prediction[:, 5]=="10th Year of Schooling", prediction[:, 5]=="General commerce course", prediction[:, 5]=="Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent", prediction[:, 5]=="15—Complementary High School Course",
                            prediction[:, 5]=="Technical-professional course", prediction[:, 5]=="Complementary High School Course—not concluded", prediction[:, 5]=="7th year of schooling", prediction[:, 5]=="2nd cycle of the general high school course", prediction[:, 5]=="9th Year of Schooling—not completed", prediction[:, 5]=="8th year of schooling", prediction[:, 5]=="General Course of Administration and Commerce", prediction[:, 5]=="Supplementary Accounting and Administration", prediction[:, 5]=="Unknown", prediction[:, 5]=="Cannot read or write", prediction[:, 5]=="Can read without having a 4th year of schooling", prediction[:, 5]=="27—Basic education 1st cycle (4th/5th year) or equivalent", prediction[:, 5]=="Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent", prediction[:, 5]=="29—Technological specialization course", prediction[:, 5]=="Higher education—degree (1st cycle)", prediction[:, 5]=="31—Specialized higher studies course", prediction[:, 5]=="32—Professional higher technical course", prediction[:, 5]=="Higher Education—master’s degree (2nd cycle)", prediction[:, 5]=="Higher Education—doctorate (3rd cycle)"]
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            prediction[:, 5] = np.select(conditions, choices, default='null')
            prediction[:, 5] = prediction[:, 5].astype(int)
        if prediction[:, 6]:
            conditions = [prediction[:, 6]=="Student", prediction[:, 6]=="Representatives of the Legislative Power and Executive Bodies,Directors, Directors and Executive Managers", prediction[:, 6]=="Specialists in Intellectual and Scientific Activities", prediction[:, 6]=="Intermediate Level Technicians and Professions", prediction[:, 6]=="Administrative staff", prediction[:, 6]=="Personal Services, Security and Safety Workers, and Sellers", prediction[:, 6]=="Farmers and Skilled Workers in Agriculture, Fisheries,and Forestry", prediction[:, 6]=="Skilled Workers in Industry, Construction, and Craftsmen", prediction[:, 6]=="Installation and Machine Operators and Assembly Workers", prediction[:, 6]=="Unskilled Workers", prediction[:, 6]=="Armed Forces Professions", prediction[:, 6]=="Other Situation", prediction[:, 6]=="(blank)", prediction[:, 6]=="Armed Forces Officers", prediction[:, 6]=="Armed Forces Sergeants", prediction[:, 6]=="Other Armed Forces personnel", prediction[:, 6]=="Directors of administrative and commercial services", prediction[:, 6]=="Hotel, catering, trade, and other services directors", prediction[:, 6]=="Specialists in the physical sciences, mathematics, engineering,and related techniques", prediction[:, 6]=="Health professionals", prediction[:, 6]=="Teachers", prediction[:, 6]=="Specialists in finance, accounting, administrative organization,and public and commercial relations", prediction[:, 6]=="Intermediate level science and engineering techniciansand professions", prediction[:, 6]=="Technicians and professionals of intermediate level of health", prediction[:, 6]=="Intermediate level technicians from legal, social, sports, cultural,and similar services", prediction[:, 6]=="Information and communication technology technicians", prediction[:, 6]=="Office workers, secretaries in general,and data processing operators", prediction[:, 6]=="Data, accounting, statistical, financial services, andregistry-related operators", prediction[:, 6]=="Other administrative support staff", prediction[:, 6]=="Personal service workers", prediction[:, 6]=="Sellers", prediction[:, 6]=="Personal care workers and the like", prediction[:, 6]=="Protection and security services personnel", prediction[:, 6]=="Market-oriented farmers and skilled agricultural and animalproduction workers", prediction[:, 6]=="Farmers, livestock keepers, fishermen, hunters and gatherers,and subsistence", prediction[:, 6]=="Skilled construction workers and the like, except electricians", prediction[:, 6]=="Skilled workers in metallurgy, metalworking, and similar", prediction[:, 6]=="Skilled workers in electricity and electronics", prediction[:, 6]=="Workers in food processing, woodworking, and clothing andother industries and crafts", prediction[:, 6]=="Fixed plant and machine operators", prediction[:, 6]=="Assembly workers", prediction[:, 6]=="Vehicle drivers and mobile equipment operators", prediction[:, 6]=="Unskilled workers in agriculture, animal production, andfisheries and forestry", prediction[:, 6]=="Unskilled workers in extractive industry, construction,manufacturing, and transport", prediction[:, 6]=="Meal preparation assistants", prediction[:, 6]=="Street vendors (except food) and street service providers"]
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
            prediction[:, 6] = np.select(conditions, choices, default='null')
            prediction[:, 6] = prediction[:, 6].astype(int)
        if prediction[:, 7]:
            conditions = [prediction[:, 7]=="Student", prediction[:, 7]=="Representatives of the Legislative Power and Executive Bodies,Directors, Directors and Executive Managers", prediction[:, 7]=="Specialists in Intellectual and Scientific Activities", prediction[:, 7]=="Intermediate Level Technicians and Professions", prediction[:, 7]=="Administrative staff", prediction[:, 7]=="Personal Services, Security and Safety Workers, and Sellers", prediction[:, 7]=="Farmers and Skilled Workers in Agriculture, Fisheries,and Forestry", prediction[:, 7]=="Skilled Workers in Industry, Construction, and Craftsmen", prediction[:, 7]=="Installation and Machine Operators and Assembly Workers", prediction[:, 7]=="Unskilled Workers", prediction[:, 7]=="Armed Forces Professions", prediction[:, 7]=="Other Situation", prediction[:, 7]=="(blank)", prediction[:, 7]=="Armed Forces Officers", prediction[:, 7]=="Armed Forces Sergeants", prediction[:, 7]=="Other Armed Forces personnel", prediction[:, 7]=="Directors of administrative and commercial services", prediction[:, 7]=="Hotel, catering, trade, and other services directors", prediction[:, 7]=="Specialists in the physical sciences, mathematics, engineering,and related techniques", prediction[:, 7]=="Health professionals", prediction[:, 7]=="Teachers", prediction[:, 7]=="Specialists in finance, accounting, administrative organization,and public and commercial relations", prediction[:, 7]=="Intermediate level science and engineering techniciansand professions", prediction[:, 7]=="Technicians and professionals of intermediate level of health", prediction[:, 7]=="Intermediate level technicians from legal, social, sports, cultural,and similar services", prediction[:, 7]=="Information and communication technology technicians", prediction[:, 7]=="Office workers, secretaries in general,and data processing operators", prediction[:, 7]=="Data, accounting, statistical, financial services, andregistry-related operators", prediction[:, 7]=="Other administrative support staff", prediction[:, 7]=="Personal service workers", prediction[:, 7]=="Sellers", prediction[:, 7]=="Personal care workers and the like", prediction[:, 7]=="Protection and security services personnel", prediction[:, 7]=="Market-oriented farmers and skilled agricultural and animalproduction workers", prediction[:, 7]=="Farmers, livestock keepers, fishermen, hunters and gatherers,and subsistence", prediction[:, 7]=="Skilled construction workers and the like, except electricians", prediction[:, 7]=="Skilled workers in metallurgy, metalworking, and similar", prediction[:, 7]=="Skilled workers in electricity and electronics", prediction[:, 7]=="Workers in food processing, woodworking, and clothing andother industries and crafts", prediction[:, 7]=="Fixed plant and machine operators", prediction[:, 7]=="Assembly workers", prediction[:, 7]=="Vehicle drivers and mobile equipment operators", prediction[:, 7]=="Unskilled workers in agriculture, animal production, andfisheries and forestry", prediction[:, 7]=="Unskilled workers in extractive industry, construction,manufacturing, and transport", prediction[:, 7]=="Meal preparation assistants", prediction[:, 7]=="Street vendors (except food) and street service providers"]
            choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
            prediction[:, 7] = np.select(conditions, choices, default='null')
            prediction[:, 7] = prediction[:, 7].astype(int)
        if prediction[:, 8]:
            conditions = [prediction[:, 8]=="no", prediction[:, 8]=="yes"]
            choices = [0, 1]
            prediction[:, 8] = np.select(conditions, choices, default='null')
            prediction[:, 8] = prediction[:, 8].astype(int)
        if prediction[:, 9]:
            conditions = [prediction[:, 9]=="no", prediction[:, 9]=="yes"]
            choices = [0, 1]
            prediction[:, 9] = np.select(conditions, choices, default='null')
            prediction[:, 9] = prediction[:, 9].astype(int)
        if prediction[:, 10]:
            conditions = [prediction[:, 10]=="no", prediction[:, 10]=="yes"]
            choices = [0, 1]
            prediction[:, 10] = np.select(conditions, choices, default='null')
            prediction[:, 10] = prediction[:, 10].astype(int)
        if prediction[:, 11]:
            conditions = [prediction[:, 11]=="no", prediction[:, 11]=="yes"]
            choices = [0, 1]
            prediction[:, 11] = np.select(conditions, choices, default='null')
            prediction[:, 11] = prediction[:, 11].astype(int)
        if prediction[:, 12]:
            conditions = [prediction[:, 12]=="female", prediction[:, 12]=="male"]
            choices = [0, 1]
            prediction[:, 12] = np.select(conditions, choices, default='null')
            prediction[:, 12] = prediction[:, 12].astype(int)
        if prediction[:, 13]:
            conditions = [prediction[:, 13]=="no", prediction[:, 13]=="yes"]
            choices = [0, 1]
            prediction[:, 13] = np.select(conditions, choices, default='null')
            prediction[:, 13] = prediction[:, 13].astype(int)
        
        Grade = rf_new_model.predict(prediction)
         
        if Grade[0] == 0:
            grade_label = "Excellent"
        elif Grade[0] == 1:
            grade_label = "Good"
        elif Grade[0] == 2:
            grade_label = "Fair"
        else:
            grade_label = "Poor"
        
        st.subheader("The Student's performance will be " + grade_label)

            