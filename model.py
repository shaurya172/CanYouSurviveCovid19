# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv(r"C:\Users\Shaurya Vardhan\Documents\Deployment-flask\covid.csv")

df['entry_date'] = pd.to_datetime(df['entry_date'],infer_datetime_format= True)

df['date_symptoms'] = pd.to_datetime(df['date_symptoms'],infer_datetime_format=True)

df['date_died'].replace('9999-99-99','Not Applicable',inplace=True)

df.rename(columns={'covid_res':'Test result'},inplace=True)

df.replace([97,98,99],np.nan,inplace=True)

df['Test result'] = df['Test result'].map({1:'Positive',2:'Negative',3:'Results Awaited'})

df.drop(['id','patient_type'],axis = 1,inplace=True)


def age_band(age):
    
   
    if age<2:
        return 'Less than 2'
    elif (age>1) and (age<11):
        return '2-10'
    elif (age>10 and age<21):
        return '10-20'
    elif (age>20 and age<31):
        return '20-30'
    elif (age>30 and age<41):
        return '30-40'
    elif (age>40 and age<51):
        return '40-50'
    elif (age>50 and age<61):
        return '50-60'
    elif (age>60 and age<81):
        return '60-80'
    else:
        return 'Above 80'
    
df['age'] = df['age'].apply(age_band)

df['Fatal'] = np.nan


for i in range(0,len(df)):
    if df['date_died'][i] != 'Not Applicable':
        df['Fatal'][i] = 'Yes'
        
df['Fatal'] = df['Fatal'].replace(np.nan,'No')

print(df['Test result'].value_counts())

df_final  = df.copy()

for i in range(0,len(df_final)):
    if df_final['Test result'][i] == 'Results Awaited':
        if df_final['Fatal'][i] == 'Yes':
            #print(df_final['Test result'][i],' ',df_final['Fatal'][i])
            
            df_final['Test result'][i] = 'Positive'

df_final.drop(['intubed','icu','pregnancy'],axis=1,inplace=True)
print('pre processing done!')
###### FEATURE ENGINEERING ############################

from sklearn.preprocessing import LabelEncoder

#features = df.columns
#features = features.to_list()
#features = features[:-1]

df_final.drop(['entry_date','date_symptoms','date_died'],axis =1, inplace= True)
df_final.fillna(method='pad',inplace = True)

X = df_final.drop('Fatal',axis=1)
X = X.apply(LabelEncoder().fit_transform)
y = df_final['Fatal']

features = ['sex',
 'pneumonia',
 'age',
 'diabetes',
 'copd',
 'asthma',
 'inmsupr',
 'hypertension',
 'other_disease',
 'cardiovascular',
 'obesity',
 'renal_chronic',
 'tobacco',
 'contact_other_covid',
 'Test result'
 ]

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

pickle.dump(lr, open('model_lr.pkl','wb'))

model = pickle.load(open('model_lr.pkl', 'rb'))



