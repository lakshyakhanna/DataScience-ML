# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:01:16 2017

@author: lakshya.khanna
"""


import os
import pandas as pd
import pandasql as pdsql
import numpy as np
from pandas.util.testing import assert_frame_equal
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
import datetime as dt
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot  as plt
import seaborn as sns
import pydot
import io
from sklearn.metrics import roc_curve, auc
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

mingw_path= 'E:\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev2\mingw64\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

pysql = lambda q: pdsql.sqldf(q, globals())

os.chdir("E:\POC\Hospital Readmission POC\MIMIC Data")


#import admission
admission = pd.read_csv('ADMISSIONS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
admission['SEQUENCE'] = admission.sort_values(by =['SUBJECT_ID','ADMITTIME','HADM_ID'],ascending = [True,True,True]).groupby(['SUBJECT_ID']).cumcount() + 1
admission.head()
admission.info()
admission.shape
admission.columns
admission.loc[admission.SUBJECT_ID==36,['ROW_ID','SUBJECT_ID','SEQUENCE']]


#import services
services = pd.read_csv('SERVICES.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
services.info()
services.loc[services.HADM_ID == 191941]


#import diagnoses_icd 
diagnosis = pd.read_csv('DIAGNOSES_ICD.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
diagnosis.info()


#import D_ICD_DIAGNOSES.csv.gz
diagnosis_name = pd.read_csv('D_ICD_DIAGNOSES.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
diagnosis_name.info()
diagnosis_name.loc[diagnosis_name.ICD9_CODE =='4019' ]


#import lab_events 
"""labevents = pd.read_csv('LABEVENTS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
labevents.info()
labevents.head(20)
labevents.FLAG.unique()
a = sorted([str(i) for i in list(labevents.VALUE)])
np.array(a) """


#import ICUstays
icustays = pd.read_csv('ICUSTAYS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
icustays.info()

#checking if there is only one ICU stay per admission
diff = icustays['HADM_ID'].count() - len(icustays['HADM_ID'].unique())
if diff > 0:
    print("There are " + str(diff) + " admissions have more than one ICU stays")
else:
    print("There is only one ICU stay per admission")

temp_icu_stay = icustays[['HADM_ID', 'ICUSTAY_ID', 'LOS']].groupby(by=['HADM_ID'], as_index=False).agg({"ICUSTAY_ID":"max", "LOS":"sum"})
recent_icu_stay = pd.merge(icustays, temp_icu_stay, on=['HADM_ID', 'ICUSTAY_ID'])[['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'LOS_y']]
recent_icu_stay.columns = ['HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'LOS']

#Validation
recent_icu_stay.loc[recent_icu_stay.HADM_ID == 147559]
icustays.loc[icustays.HADM_ID == 147559]


#import patients
patients = pd.read_csv('PATIENTS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
patients.info()
patients.columns


#import drgcodes
drgcodes = pd.read_csv('DRGCODES.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
drgcodes.info()
len(drgcodes.HADM_ID.unique())
drgcodes[drgcodes.HADM_ID == 100006]
drgcodes_sev_mor = drgcodes[['HADM_ID','DRG_SEVERITY','DRG_MORTALITY']].groupby('HADM_ID').agg({"DRG_SEVERITY":"mean", "DRG_MORTALITY":"mean"}).reset_index()
drgcodes_sev_mor.loc[drgcodes_sev_mor.DRG_SEVERITY.isnull(),['DRG_SEVERITY']] = 0
drgcodes_sev_mor.loc[drgcodes_sev_mor.DRG_MORTALITY.isnull(),['DRG_MORTALITY']] = 0
drgcodes_sev_mor.shape
drgcodes_sev_mor.columns


#create admission_count
admission_count = admission.groupby('SUBJECT_ID').SUBJECT_ID.agg(['count']).reset_index()
admission_count.columns = ['SUBJECT_ID', 'READMISSION_COUNT']
admission_count.head()
admission_count.info()
admission_count.shape
admission_count[admission_count.READMISSION_COUNT>1].count()# 7537 patients that are readmitted
#admission_count = pysql('SELECT SUBJECT_ID, COUNT(SUBJECT_ID) AS COUNT FROM admission group by SUBJECT_ID;')


#create patients_readmitted
patients_readmitted = pd.merge(admission, admission_count,how = 'left', on = ['SUBJECT_ID'])
patients_readmitted = patients_readmitted.loc[patients_readmitted.READMISSION_COUNT > 1,]
patients_readmitted.shape
patients_readmitted.info()
patients_readmitted.head()
#patients_readmitted = pysql('SELECT a.* , b.COUNT AS READMISSION_COUNT FROM admission a left outer join admission_count b on a.SUBJECT_ID = b.SUBJECT_ID;')


#Get Discharge Time
patients_readmitted_readdays = pysql('select a.SUBJECT_ID, a.HADM_ID,a.DISCHTIME as CURR_DISCHTIME,b.ADMITTIME as NXT_ADMITTIME ,(julianday(b.ADMITTIME) - julianday(a.DISCHTIME)) as DAYS_TO_READMISSION  from patients_readmitted a inner join patients_readmitted b on a.SUBJECT_ID = b.SUBJECT_ID  and a.SEQUENCE = b.SEQUENCE - 1')
patients_readmitted_readdays.shape
patients_readmitted_readdays.loc[patients_readmitted_readdays.SUBJECT_ID==107]
#patients_readmitted_readdays['DAYS_TO_READMISSION'] = patients_readmitted_readdays['NXT_ADMITTIME'].astype('datetime64[ns]') - patients_readmitted_readdays['CURR_DISCHTIME'].astype('datetime64[ns]')


#patients admiited which have taken cardiac services
patients_readmitted_heart = pysql('SELECT a.* from patients_readmitted_readdays a WHERE a.HADM_ID in (SELECT DISTINCT HADM_ID FROM services where CURR_SERVICE in (\'CMED\',\'CSURG\'));')
patients_readmitted_heart.shape
patients_readmitted_heart.info()
len(patients_readmitted_heart.HADM_ID.unique())
patients_readmitted_heart.loc[patients_readmitted_heart.DAYS_TO_READMISSION < 120.0]
hrt_adm_id = list(patients_readmitted_heart.HADM_ID.unique())


#no of surgeries
surgeries = pysql('SELECT count(1) as NO_OF_SURGERIES, CURR_SERVICE,HADM_ID FROM services where CURR_SERVICE in (\'CSURG\') GROUP BY CURR_SERVICE,HADM_ID  ;')
surgeries.shape
surgeries[surgeries['NO_OF_SURGERIES']>1]
test = surgeries.groupby('HADM_ID').count()
test.shape

#Add no of surgeries to the dataset

patients_readmitted_heart_surgeries = pd.merge(patients_readmitted_heart,surgeries[['HADM_ID','NO_OF_SURGERIES']],how = 'left' , on = ['HADM_ID'])
patients_readmitted_heart_surgeries.NO_OF_SURGERIES.unique()
patients_readmitted_heart_surgeries.NO_OF_SURGERIES.fillna(0.0,inplace = True)
patients_readmitted_heart_surgeries['NO_OF_SURGERIES'] = patients_readmitted_heart_surgeries.NO_OF_SURGERIES.astype('int64')
patients_readmitted_heart_surgeries.info()
patients_readmitted_heart_surgeries.shape


#Merging the dataframe with admission to get the admission attributes
#patients_readmit_admattr = pd.merge(patients_readmitted_heart,admission[['HADM_ID','ADMISSION_TYPE','ADMITTIME', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION','MARITAL_STATUS', 'ETHNICITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admattr = pd.merge(patients_readmitted_heart_surgeries,admission[['HADM_ID','ADMISSION_TYPE','ADMITTIME', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION','MARITAL_STATUS', 'ETHNICITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admattr.shape 
total_missing = patients_readmit_admattr.isnull().sum()
patients_readmit_admattr.columns


#Merging the dataframe with patient to get the patient attributes
patients_readmit_admPat_attr = pd.merge(patients_readmit_admattr,patients[['SUBJECT_ID','GENDER', 'DOB']] ,how = 'left' , on = ['SUBJECT_ID'])
patients_readmit_admPat_attr.shape
patients_readmit_admPat_attr.columns


#Merging the dataframe with drgcodes to get the mortality & severity scores attributes
patients_readmit_admPatDrg_attr = pd.merge(patients_readmit_admPat_attr,drgcodes_sev_mor[['HADM_ID', 'DRG_SEVERITY', 'DRG_MORTALITY']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admPatDrg_attr.shape
patients_readmit_admPatDrg_attr.columns


#Merging the dataframe with ICUstay to get the LOS,Last_icu scores attributes
patients_readmit_admPatDrgICU_attr = pd.merge(patients_readmit_admPatDrg_attr,recent_icu_stay[['HADM_ID','LAST_CAREUNIT', 'LOS']] ,how = 'left' , on = ['HADM_ID'])
patients_readmit_admPatDrgICU_attr.shape
patients_readmit_admPatDrgICU_attr.columns



##diagnosis_expanded
diagnosis_exp = diagnosis.loc[diagnosis.HADM_ID.isin(hrt_adm_id),['HADM_ID','ICD9_CODE']]
diagnosis_exp

#Check if the codes are same in new table after filtering
if (sorted(hrt_adm_id) == sorted(list(diagnosis_exp.HADM_ID.unique()))):print(True)
else:print(False)

len(diagnosis_exp.ICD9_CODE.unique())

diagnosis_exp.info()
diagnosis_exp = pd.crosstab(diagnosis_exp['HADM_ID'],diagnosis_exp['ICD9_CODE'])
diagnosis_exp.reset_index(inplace = True)
#diagnosis_exp.drop(['index'],axis = 1,inplace = True)

diagnosis_exp.head()
diagnosis_exp.columns
diagnosis_exp.shape
diag_exp_cols = list(sorted(diagnosis_exp.columns))
diag_exp_cols.remove('HADM_ID')
diag_exp_cols.insert(0,'HADM_ID')

diag_exp_cols.__contains__('HADM_ID')
diag_exp_cols.index('HADM_ID')
len(diagnosis_exp.columns)

diagnosis_exp = diagnosis_exp.reindex_axis(diag_exp_cols,axis=1)

#get the list of ICD9 codes
icd9_codes = list(sorted(diagnosis_exp.columns))
icd9_codes.remove('HADM_ID')

#Function to make a code deficient dataframe to a complete frame
def icd9_df(df,icd9_codes):
    cln_list = list(sorted(df.columns))
    cln_list.remove('HADM_ID')
    missing_codes = [code for code in icd9_codes if code not in cln_list]
    pd.concat([df,pd.DataFrame(columns=missing_codes)])
    for ftr in missing_codes:
        df[ftr] = 0
    full_cln_list = list(sorted(df.columns))
    full_cln_list.remove('HADM_ID')
    full_cln_list.insert(0,'HADM_ID')
    df = df.reindex_axis(diag_exp_cols,axis=1)
    return df

# Test function
icd_test_df = icd9_df(diagnosis_exp,icd9_codes)
if assert_frame_equal(diagnosis_exp, icd_test_df) == True:print(True)
else:print(True)


#Final Data Set to Get ICD Codes as columns
final_dataset = pd.merge(patients_readmit_admPatDrgICU_attr,diagnosis_exp,how = 'left', on= ['HADM_ID'])
#final_dataset = patients_readmit_admPatDrgICU_attr.copy()
final_dataset.shape
final_dataset.columns
final_dataset.info()
total_missing = final_dataset.isnull().sum()
total_missing[total_missing>0]

# Impute Values for missing data
final_dataset.loc[final_dataset.RELIGION.isnull(),['RELIGION']] = final_dataset.RELIGION.mode()[0]
final_dataset.loc[final_dataset.MARITAL_STATUS.isnull(),['MARITAL_STATUS']] = final_dataset.MARITAL_STATUS.mode()[0]
final_dataset.loc[final_dataset.LAST_CAREUNIT.isnull(),['LAST_CAREUNIT']] = final_dataset.LAST_CAREUNIT.mode()[0]
final_dataset.loc[final_dataset.LOS.isnull(),['LOS']] = 0

#EDA on Final Dataset
final_dataset[final_dataset.DAYS_TO_READMISSION >= 30].shape
final_dataset['READMISSION_60dAYS'] = final_dataset.DAYS_TO_READMISSION.apply(lambda x : 1 if x <= 60 else 0)
final_dataset['READMISSION_30dAYS'] = final_dataset.DAYS_TO_READMISSION.apply(lambda x : 1 if x <= 30 else 0)


#Preprocessing the Final Dataset befor splitting

processed_final_dataset = final_dataset.copy()
processed_final_dataset.shape
final_dataset.shape

##Data Preprocessing
processed_final_dataset.rename(columns={'ADMITTIME':'CURR_ADMITTIME'}, inplace=True)
processed_final_dataset.select_dtypes(include = ['object']).columns

#Convert Datetime features to Datetime datatype
processed_final_dataset.DOB.head()
processed_final_dataset['DOB'] = pd.to_datetime(processed_final_dataset['DOB'])
processed_final_dataset['CURR_DISCHTIME'] = pd.to_datetime(processed_final_dataset['CURR_DISCHTIME'])
processed_final_dataset['NXT_ADMITTIME'] = pd.to_datetime(processed_final_dataset['NXT_ADMITTIME'])
processed_final_dataset['CURR_ADMITTIME'] = pd.to_datetime(processed_final_dataset['CURR_ADMITTIME'],format = "%Y/%m/%d")

"""#Feature Engineering"""



#Creating Hour of the Day column
processed_final_dataset.columns
processed_final_dataset['DICHARGE_HOUR_OF_DAY'] = processed_final_dataset.CURR_DISCHTIME.apply(lambda ts : ts.hour)
processed_final_dataset['ADMISSION_HOUR_OF_DAY'] = processed_final_dataset.CURR_ADMITTIME.apply(lambda ts : ts.hour)
processed_final_dataset[['DICHARGE_HOUR_OF_DAY','CURR_DISCHTIME','CURR_ADMITTIME','ADMISSION_HOUR_OF_DAY']]

processed_final_dataset.CURR_ADMITTIME.dtypes

#Creating one more column to bin the data
pd.cut(processed_final_dataset.DICHARGE_HOUR_OF_DAY,4).unique()

processed_final_dataset.DICHARGE_HOUR_OF_DAY.describe()

def get_IQR(ftr):
    quartile_1, quartile_3 = np.percentile(ftr, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ftr > upper_bound) | (ftr < lower_bound))

get_IQR(processed_final_dataset.DICHARGE_HOUR_OF_DAY)
processed_final_dataset.groupby(['DICHARGE_HOUR_OF_DAY']).DICHARGE_HOUR_OF_DAY.count()

def hour_bins(hour):
    if (hour <= 10):
        return 'EarlyMorning'
    elif ((hour > 10) & (hour <= 12)):
        return 'Morning'
    elif ((hour > 12) & (hour <= 18)):
        return 'Noon'
    else:
        return 'Evening'

#Admission Hour - No pattern
processed_final_dataset.groupby(['READMISSION_60dAYS','ADMISSION_HOUR_OF_DAY']).DICHARGE_HOUR_OF_DAY.count()

processed_final_dataset.DICHARGE_HOUR_OF_DAY.unique()
processed_final_dataset.DICHARGE_HOUR_OF_DAY.head()
processed_final_dataset['DICHARGE_HOUR_BAND'] = processed_final_dataset['DICHARGE_HOUR_OF_DAY'].map(hour_bins)
processed_final_dataset[['DICHARGE_HOUR_BAND','DICHARGE_HOUR_OF_DAY']].head()
processed_final_dataset.DICHARGE_HOUR_BAND.isnull().sum()

processed_final_dataset.DICHARGE_HOUR_OF_DAY.isnull().sum()
#Encoding DICHARGE_HOUR_BAND
le = LabelEncoder()
le.fit(['EarlyMorning','Morning','Noon','Evening'])
le.classes_
processed_final_dataset['DICHARGE_HOUR_BAND_ENC'] = le.transform(processed_final_dataset['DICHARGE_HOUR_BAND'])
processed_final_dataset.DICHARGE_HOUR_BAND_ENC.isnull().sum()

#Convert Binned Hour of Day to ordinal categorical
processed_final_dataset['DICHARGE_HOUR_BAND_ENC'] = processed_final_dataset.DICHARGE_HOUR_BAND_ENC.astype('category',categories = [0,1,2,3],ordered = True)
processed_final_dataset.info()
processed_final_dataset.DICHARGE_HOUR_BAND_ENC.value_counts()
processed_final_dataset[processed_final_dataset.DICHARGE_HOUR_BAND_ENC.isnull() == True]
processed_final_dataset.DICHARGE_HOUR_BAND_ENC.isnull().sum()

#processed_final_dataset.INSURANCE.unique()
#processed_final_dataset['INSURANCE'] = processed_final_dataset.INSURANCE.astype('category')



#Making New feature Age
processed_final_dataset.DOB.dtypes

processed_final_dataset['AGE'] = (((processed_final_dataset.CURR_ADMITTIME - processed_final_dataset.DOB)/ np.timedelta64(1, 'D')) / 365).round().astype('int64')
processed_final_dataset[processed_final_dataset['AGE']<0][['AGE','CURR_ADMITTIME','DOB']]

processed_final_dataset.loc[processed_final_dataset['AGE']<0,['AGE']] = - processed_final_dataset.loc[processed_final_dataset['AGE']<0,['AGE']]



#Convert imestamp columns to Str
processed_final_dataset.info()

processed_final_dataset['CURR_DISCHTIME'] = processed_final_dataset['CURR_DISCHTIME'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
processed_final_dataset['NXT_ADMITTIME'] = processed_final_dataset['NXT_ADMITTIME'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
processed_final_dataset['CURR_ADMITTIME'] = processed_final_dataset['CURR_ADMITTIME'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
processed_final_dataset['DOB'] = processed_final_dataset['DOB'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))

processed_final_dataset.columns

"""#Convert Drg Severity and Drg Mortality to  ordinal categorical
processed_final_dataset.dtypes
processed_final_dataset.DRG_SEVERITY.unique()
processed_final_dataset.DRG_MORTALITY.unique()

processed_final_dataset['DRG_SEVERITY'] = processed_final_dataset.DRG_SEVERITY.astype('category',categories = [0.0,1.0,1.5,2.0,2.5,3.0,3.5,4.0],ordered = True)
processed_final_dataset['DRG_MORTALITY'] = processed_final_dataset.DRG_MORTALITY.astype('category',categories = [0.0,1.0,1.5,2.0,2.5,3.0,3.5,4.0],ordered = True)
"""

processed_final_dataset1 = processed_final_dataset.drop(['CURR_DISCHTIME','NXT_ADMITTIME','CURR_ADMITTIME','DOB'],axis = 1)
processed_final_dataset1.shape

processed_final_dataset1.select_dtypes(include = ['object']).columns
processed_final_dataset1.columns


#Bin LOS into 1,2,3(3+)
def LOS_BIN(LOS):
    if LOS < 0.6:
        return 0
    elif ((LOS > 0.6) & (LOS <= 1.3)):
        return 1
    elif ((LOS > 1.3)  & (LOS <= 2.3)):
        return 2
    else:
        return 3
    
processed_final_dataset1.LOS
processed_final_dataset1['LOS_BIN'] = processed_final_dataset1.LOS.apply(LOS_BIN)
processed_final_dataset1[['LOS_BIN','LOS']]

#Decrease in Accuracy so removing it.
"""
#Bin Ethnicity to White, Black , others
processed_final_dataset1.ETHNICITY.unique().tolist()

ethnic_dict = {'WHITE' : 'WHITE','HISPANIC OR LATINO' : 'HISPANIC', 'UNKNOWN/NOT SPECIFIED' : 'OTHERS', 'BLACK/AFRICAN AMERICAN' : 'BLACK', 'ASIAN' : 'ASIAN', 'PATIENT DECLINED TO ANSWER' : 'OTHERS', 'OTHER' : 'OTHERS', 'HISPANIC/LATINO - PUERTO RICAN' : 'HISPANIC', 'WHITE - RUSSIAN' : 'WHITE', 'WHITE - BRAZILIAN' : 'WHITE', 'AMERICAN INDIAN/ALASKA NATIVE' : 'OTHERS', 'ASIAN - ASIAN INDIAN' : 'ASIAN', 'BLACK/HAITIAN' : 'BLACK', 'MIDDLE EASTERN' : 'OTHERS', 'MULTI RACE ETHNICITY' : 'OTHERS', 'BLACK/CAPE VERDEAN' : 'BLACK', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' : 'OTHERS', 'HISPANIC/LATINO - GUATEMALAN' : 'HISPANIC', 'ASIAN - CHINESE' : 'ASIAN', 'HISPANIC/LATINO - COLOMBIAN' : 'HISPANIC', 'UNABLE TO OBTAIN' : 'OTHERS', 'HISPANIC/LATINO - CUBAN' : 'HISPANIC', 'ASIAN - OTHER' : 'ASIAN', 'ASIAN - CAMBODIAN' : 'ASIAN', 'HISPANIC/LATINO - DOMINICAN' : 'HISPANIC', 'PORTUGUESE' : 'OTHERS', 'ASIAN - VIETNAMESE' : 'ASIAN', 'BLACK/AFRICAN' : 'BLACK', 'WHITE - OTHER EUROPEAN' : 'WHITE', 'ASIAN - THAI' : 'ASIAN', 'WHITE - EASTERN EUROPEAN' : 'WHITE' }

processed_final_dataset1['ETHNICITY_GRP'] = processed_final_dataset1.ETHNICITY.map(ethnic_dict)
"""

##EDA before applying dummies
corrmat = processed_final_dataset1.corr()
f, ax = plt.subplots(figsize=(12, 9))
plt.xticks(rotation=90)
plt.yticks(rotation=90)
sns.heatmap(corrmat, square=True)

plt.matshow(processed_final_dataset1.corr())
    
#Plotting for categorical columns
sns.factorplot(x="ADMISSION_TYPE", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="ADMISSION_LOCATION", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="DISCHARGE_LOCATION", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="INSURANCE", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="RELIGION", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="MARITAL_STATUS", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="ETHNICITY", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="GENDER", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="LAST_CAREUNIT", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)
sns.factorplot(x="MARITAL_STATUS", hue="READMISSION_60dAYS", data=processed_final_dataset1, kind="count", size=6)


#EDA for contnuous columns
sns.FacetGrid(processed_final_dataset1, row="READMISSION_60dAYS",size=8).map(sns.kdeplot, "DRG_SEVERITY").add_legend()
sns.FacetGrid(processed_final_dataset1, row="READMISSION_60dAYS",size=8).map(sns.kdeplot, "AGE").add_legend()


#Adding log of Age column
sns.kdeplot(processed_final_dataset1['AGE'])
sns.kdeplot(np.log(processed_final_dataset1['AGE']))
processed_final_dataset1['LOG_AGE'] = np.log(processed_final_dataset1['AGE'])


processed_final_dataset1.DRG_MORTALITY.unique()

#Applying PCA on 
processed_final_dataset1.dtypes


# One-Hot Encoding
final_dummy_data = pd.get_dummies(processed_final_dataset1, columns = ['ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION','INSURANCE','RELIGION','MARITAL_STATUS','ETHNICITY','GENDER','LAST_CAREUNIT'])
final_dummy_data.shape


#Exporting Final Dataset
final_dummy_data.to_csv('readmission_final_dataset.csv',index =  False)
final_dummy_data.READMISSION_30dAYS.value_counts()
final_dummy_data.READMISSION_60dAYS.value_counts()

#Splitting the data into Test & Train
X = final_dummy_data.drop(['READMISSION_60dAYS', 'READMISSION_30dAYS' , 'DAYS_TO_READMISSION','AGE','DICHARGE_HOUR_BAND'],axis = 1)
y = final_dummy_data['READMISSION_30dAYS']
final_dummy_data.shape
final_dummy_data.columns
X.shape
y.shape


X_train_pre, X_test_pre, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.25 , stratify = y , random_state=2017)

# Export the Train & Test Datasets
"""X_train_pre.to_csv('X_train_pre.csv',index =  False)
y_train.to_csv('y_train.csv',index =  False)
X_test_pre.to_csv('X_test_pre.csv',index =  False)
y_test.to_csv('y_test.csv',index =  False)"""

#Export the Admission Id's of Test & Traing set. They are unique and can be checked upon splitting again that split is same.
"""X_train_pre.to_csv('train_HADM_ID.csv',index = False, columns = ['HADM_ID'])
X_test_pre.to_csv('test_HADM_ID.csv',index = False, columns = ['HADM_ID'])"""

#Import Admission ID's in X_train & X_test as exported from v12 RF.
train_hadm = pd.read_csv('train_HADM_ID.csv')
test_hadm = pd.read_csv('test_HADM_ID.csv')

#Check if the split has happened on same admission id's
train_hadm_check =  [hadm for hadm in train_hadm.HADM_ID.tolist() if hadm not in X_train_pre.HADM_ID.tolist()]
if len(train_hadm_check) == 0:
    print('The train split has happened correctly')
else:
    print('There are some different train HADM_ID')    
    
test_hadm_check =  [hadm for hadm in test_hadm.HADM_ID.tolist()  if hadm not in X_test_pre.HADM_ID.tolist()]
if len(test_hadm_check) == 0:
    print('The test split has happened correctly')
else:
    print('There are some different test HADM_ID')  


#check if all admission id's are unique
(X_train_pre.HADM_ID.value_counts()>1).any() == True
(X_test_pre.HADM_ID.value_counts()>1).any() == True

#get X_train, X_test
X_train = X_train_pre.drop(['HADM_ID'],axis = 1)
X_test = X_test_pre.drop(['HADM_ID'],axis = 1)

X_train.shape
y_train.shape
X_test.columns
X_test.shape
y_test.shape
y_test[y_test == 0].count()
y_test.value_counts()
y_train.value_counts()

X_train.info()
X_train.select_dtypes(include = ['category']).columns
X_train.DICHARGE_HOUR_BAND_ENC.isnull().sum()

##Feature Selection
#Remove faeatures with near to zero variance
df_var = np.var(X_train).reset_index()
df_var.columns = ['Feature','Var']
df_var.shape
var_col = list(df_var[df_var.Var > 0.001].Feature)
len(var_col)

df_var.head()

X_train_var = X_train[var_col]
X_train_var.shape
X_test_var = X_test[var_col]


#Get distinct features from RF & GB 200 ftrs.
rf_gb_top200_ftr = list(set(rf_top200_ftr ) | set(gb_top200_ftr))
len(rf_gb_top200_ftr)

#Taking subset of features from X_train_var and X_test_var on the basis of the random forest ftr_importance dataframe.
X_train_var_imp = X_train_var[rf_gb_top200_ftr]
X_test_var_imp = X_test_var[rf_gb_top200_ftr]
X_train_var_imp.shape
X_test_var_imp.shape
X_test_var_imp.__contains__('LOS_BIN')
X_test_var_imp.__contains__('EHTNICITY_GRP')

#The features with non-zero coefficient from Lasso
X_train_lasso_coeff = X_train[lasso_non_zero_coeff]
X_test_lasso_coeff = X_test[lasso_non_zero_coeff]

#Combining features from RF,GB(from var) and lasso (from whole X_train).
rf_gb_lasso_ftr = list(set(rf_top200_ftr ) | set(gb_top200_ftr) | set(lasso_non_zero_coeff))
X_train_RGL = X_train[rf_gb_lasso_ftr]
X_test_RGL = X_test[rf_gb_lasso_ftr]

#Apply SMOTE
sm = SMOTE(random_state = 2017,ratio = 'minority',kind = 'svm')
X_train_res , y_train_res = sm.fit_sample(X_train_var_imp,y_train)
#X_train_res , y_train_res = sm.fit_sample(X_train_RGL,y_train)
X_train_res.shape

#Scaling the data
std_scl = StandardScaler()
X_train_scl = std_scl.fit_transform(X_train_res)

#Creating a PCA object
pca = decomposition.PCA(n_components = 82)
pca.fit(X_train_scl)
captured_variance = list(pca.explained_variance_ratio_.cumsum())

X_train_pca = pca.transform(X_train_scl)

#Creating LDA object
lda = LinearDiscriminantAnalysis(n_components = 101)
lda.fit(X_train_scl,y_train)
captured_var_lda = list(lda.explained_variance_ratio_.cumsum())

X_train_lda = lda.transform(X_train_scl)


#Checking the columns that are not in X_test but are X_train because of one-hot encoding
col_test_missing = [ftr for ftr in list(X_train.columns) if ftr not in list(X_test.columns)]
len(col_test_missing)

#Scaling & PCA
X_test_scl = std_scl.transform(X_test_var_imp)
X_test_pca = pca.transform(X_test_scl)
X_test_lda = lda.transform(X_test_scl)

#Plotting Histogram of Probabilities
def create_hist(y_prob):
    plt.rcParams['font.size'] = 14
    plt.hist(y_prob, bins=10)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of readmission')
    plt.ylabel('Frequency')


#Plot ROC-AUC curve
def get_roc_curve(y_pred_prob):
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print('ROC AUC: %0.2f' % roc_auc)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
    return fpr,tpr,thresholds
    #plt.savefig('roc_and_threshold.png')
    #plt.close()
    
# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold,y_pred_prob):
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
    
#exporting data for Keras
os.chdir('E:\POC\Hospital Readmission POC\Code\Keras')
X_train_var_imp.to_csv('X_train_var_imp.csv',index =  False)
y_train.to_csv('y_train.csv',index =  False)
X_test_var_imp.to_csv('X_test_var_imp.csv',index =  False)
y_test.to_csv('y_test.csv',index =  False)

    
#Model Building

#Random Forest
rf_estimator = ensemble.RandomForestClassifier(random_state = 2017)
param_grid = dict(n_estimators = range(50,750,50) , criterion = ['gini','entropy'] , max_features = [41,42,43,44,45],max_depth = range(7,12,1))
param_grid_test = {'criterion': ['gini'], 'max_features': [44], 'n_estimators': [250], 'max_depth': [10]}
param_grid_test2 = {'criterion': ['gini'], 'max_features': [43], 'n_estimators': [250], 'max_depth': [11]}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator, param_grid_test2, cv =10 , verbose = 1, n_jobs = 10,scoring = "roc_auc" )
rf_grid_estimator.fit(X_train_res,y_train_res)
print(rf_grid_estimator.best_score_)
print(rf_grid_estimator.score(X_train_res, y_train_res))
print(rf_grid_estimator.best_estimator_)
print(rf_grid_estimator.best_params_)
print(rf_grid_estimator.best_estimator_.feature_importances_)

#extracting all the trees build by random forest algorithm
n_tree = 0
for est in rf_grid_estimator.best_estimator_: 
    dot_data = io.StringIO()
    tmp = est.tree_
    tree.export_graphviz(tmp, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf("rftree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1

#Prediction
y_pred = rf_grid_estimator.predict(X_test_var_imp)
y_pred_rf_prob = rf_grid_estimator.predict_proba(X_test_var_imp)[:,1]
#y_pred = rf_grid_estimator.predict(X_test_RGL)
#y_pred_rf_prob = rf_grid_estimator.predict_proba(X_test_RGL)[:,1]

#Accuracy
metrics.accuracy_score(y_test, y_pred)
metrics.roc_auc_score(y_test, y_pred_rf_prob)
metrics.confusion_matrix(y_test,y_pred)
 
#ROC-AUC curve
get_roc_curve('y_pred_rf_prob.csv')

#Export, Import & Check
"""joblib.dump(rf_grid_estimator.best_estimator_,'RF_66_ROC-AUC.pkl')
rf_import = joblib.load('RF_66_ROC-AUC.pkl') 
y_pred_rf_import_prob = rf_import.predict_proba(X_test_var)[:,1]
y_pred_rf_import = rf_import.predict(X_test_var)
metrics.roc_auc_score(y_test, y_pred_rf_import_prob)
metrics.confusion_matrix(y_test,y_pred_rf_import)"""

#Build feature Importance dataframe
ftr_name = X_train_var.columns
ftr_imp_rf = rf_grid_estimator.best_estimator_.feature_importances_
ftr_importance = pd.DataFrame({'ftr_name' : ftr_name, 'ftr_imp_rf' : ftr_imp_rf},columns = ['ftr_name','ftr_imp_rf'])
ftr_importance.sort_values(['ftr_imp_rf'],ascending = False,inplace = True)
ftr_importance.reset_index(inplace = True)
rf_top200_ftr = ftr_importance.ix[0:250,'ftr_name'].tolist()
ftr_importance[ftr_importance.ftr_imp_rf > 0].ftr_name



#ROC-AUC diagram for RF

type(y_pred)
roc_df = pd.concat([y_test.reset_index().READMISSION_60dAYS,pd.Series(y_pred),pd.Series(y_pred_rf_prob)] , axis = 1)
roc_df.columns = ['y_test','y_pred','y_pred_prob']
roc_df[(roc_df.y_test ==1) & (roc_df.y_pred==1)].y_pred_prob

True_Positive = roc_df[(roc_df.y_test ==1) & (roc_df.y_pred==1)].y_pred_prob
True_Negative = roc_df[(roc_df.y_test ==0) & (roc_df.y_pred==0)].y_pred_prob
sns.kdeplot(True_Positive)
sns.kdeplot(True_Negative)

roc_df.info()

y_pred_rf_prob.max()
y_pred_rf_prob[y_pred_rf_prob<0.600]

sns.jointplot(y_pred_rf_prob)



#Gradient Boosting
gb_estimator = ensemble.GradientBoostingClassifier(random_state = 2017)
#gb_param_grid = dict(n_estimators = range(50,750,50),max_depth = [1,2,3,4])
gb_param_grid_test  = {'max_depth': [3], 'n_estimators': [100]}
gb_grid_estimator = model_selection.GridSearchCV(gb_estimator, gb_param_grid_test, cv =10 , verbose = 1, n_jobs = 15,scoring = "roc_auc")
gb_grid_estimator.fit(X_train_res,y_train_res)
print(gb_grid_estimator.best_score_)
print(gb_grid_estimator.score(X_train, y_train))
print(gb_grid_estimator.best_params_)

#Getting the feature importance with Gradient Boosting.
ftr_name = X_train_var.columns
ftr_imp_gb = gb_grid_estimator.best_estimator_.feature_importances_
ftr_importance_gb = pd.DataFrame({'ftr_name' : ftr_name, 'ftr_imp_gb' : ftr_imp_gb},columns = ['ftr_name','ftr_imp_gb'])
ftr_importance_gb.sort_values(['ftr_imp_gb'],ascending = False,inplace = True)
ftr_importance_gb.reset_index(inplace = True)
gb_top200_ftr = ftr_importance_gb.ix[0:250,'ftr_name'].tolist()
ftr_importance_gb[ftr_importance_gb.ftr_imp_gb > 0].ftr_name
gb_top200_ftr.__contains__('LOS_BIN')


y_pred_gb = gb_grid_estimator.predict(X_test_var)
y_pred_gb_prob = gb_grid_estimator.predict_proba(X_test_var)[:,1]

metrics.accuracy_score(y_test, y_pred_gb)
metrics.roc_auc_score(y_test, y_pred_gb_prob)
metrics.confusion_matrix(y_test, y_pred_gb)



#Neural Network
nn_estimator = MLPClassifier(random_state = 2017)
nn_param_grid = {'hidden_layer_sizes' : [(20,2),(25,3)], 'activation' : ['relu','tanh'],'solver' : ['sgd','adam'], 'learning_rate' : ['constant','adaptive'] }
nn_grid_estimator = model_selection.GridSearchCV(nn_estimator, nn_param_grid, cv =10 , verbose = 1, n_jobs = 15 ,scoring = "roc_auc")
nn_grid_estimator.fit(X_train_scl,y_train)
print(nn_grid_estimator.best_score_)
print(nn_grid_estimator.score(X_train_res, y_train_res))
print(nn_grid_estimator.best_params_)

y_pred_nn = nn_grid_estimator.predict(X_test_scl)
y_pred_nn_prob = nn_grid_estimator.predict_proba(X_test_scl)[:,1]

metrics.accuracy_score(y_test, y_pred_nn)
metrics.roc_auc_score(y_test, y_pred_nn_prob)
metrics.confusion_matrix(y_test, y_pred_nn)




#SGD
sgd_estimator = linear_model.SGDClassifier(class_weight = 'balanced',random_state = 2017)
sgd_param_grid = dict(loss = ['log','modified_huber'], penalty = ['L1','L2','elasticnet'] , alpha =  [10 ** a for a in range(-6, 7)])
sgd_grid_Estimator = model_selection.GridSearchCV(sgd_estimator, sgd_param_grid, cv =10 , verbose = 1, n_jobs = 15 ,scoring = "roc_auc")
sgd_grid_Estimator.fit(X_train_scl,y_train)
print(sgd_grid_Estimator.best_score_)
print(sgd_grid_Estimator.score(X_train_scl, y_train))
print(sgd_grid_Estimator.best_params_)

sgd_coef = list(sgd_grid_Estimator.best_estimator_.coef_.ravel())
sgd_columns = list(X_train.columns)
sgd_ftr_imp = pd.DataFrame({'col' : sgd_columns , 'Imp' : sgd_coef})

#Predict
y_pred_sgd = sgd_grid_Estimator.predict(X_test_scl)
y_pred_sgd_prob = sgd_grid_Estimator.predict_proba(X_test_scl)[:,1]

#Accuracies
metrics.accuracy_score(y_test, y_pred_sgd)
metrics.roc_auc_score(y_test, y_pred_sgd_prob)



#Lasso Regression
lasso_estimator = linear_model.LassoCV(alphas = [10 ** a for a in range(-6, 7)],random_state=2017)
lasso_param_grid = dict(max_iter = [900])
lasso_grid_estimator = model_selection.GridSearchCV(lasso_estimator, lasso_param_grid, cv =10 , verbose = 1, n_jobs = 15 ,scoring = "roc_auc")
lasso_grid_estimator.fit(X_train_scl,y_train)
print(lasso_grid_estimator.best_score_)
print(lasso_grid_estimator.score(X_train_scl, y_train))
print(lasso_grid_estimator.best_params_)
print(lasso_grid_estimator.best_estimator_)

#Predict
y_pred_lasso = lasso_grid_estimator.predict(X_test_scl)
y_pred_lasso[y_pred_lasso>1]
#y_pred_lasso_test = pd.Series(y_pred_lasso).apply(lambda x : 0 if x < 0.5 else 1)

#Accuracies
metrics.roc_auc_score(y_test, y_pred_lasso)

#get coefficients and make a dataframe

ftr_name = X_train.columns
ftr_coef_lasso = lasso_grid_estimator.best_estimator_.coef_.tolist()
ftr_importance_lasso = pd.DataFrame({'ftr_name' : ftr_name, 'ftr_coef_lasso' : ftr_coef_lasso},columns = ['ftr_name','ftr_coef_lasso'])
lasso_non_zero_coeff = ftr_importance_lasso.loc[ftr_importance_lasso.ftr_coef_lasso != 0,'ftr_name'].tolist()


#Log regression
log_estimator = linear_model.LogisticRegression(max_iter = 900,random_state = 2017)
log_param_grid = dict(penalty = ['l2'], C  = [10 ** a for a in range(-6, 7)])
log_grid_estimator = model_selection.GridSearchCV(log_estimator, log_param_grid, cv =10 , verbose = 1 ,scoring = "roc_auc")
log_grid_estimator.fit(X_train_lda,y_train)
print(log_grid_estimator.best_score_)
print(log_grid_estimator.score(X_train_lda, y_train))
print(log_grid_estimator.best_params_)
print(log_grid_estimator.best_estimator_)

y_pred_log = log_grid_estimator.predict(X_test_lda)
y_pred_log_prob = log_grid_estimator.predict_proba(X_test_lda)[:,1]

metrics.roc_auc_score(y_test,y_pred_log_prob)
metrics.confusion_matrix(y_test,y_pred_log)
metrics.roc_curve(y_test,y_pred_log)

create_hist(y_pred_log_prob)   
get_roc_curve(y_pred_log_prob)
    
evaluate_threshold(0.5)
evaluate_threshold(0.2)


#Xgboost
#Converting categorical variable in X_train_var and X_test_var to int as Xgboost does not take categorical datatype.
X_train_var_xgb = X_train_var.copy()
X_train_var_xgb['DICHARGE_HOUR_BAND_ENC'] = X_train_var_xgb.DICHARGE_HOUR_BAND_ENC.astype('int64')
X_train_var_xgb.info()

X_train_xgb_res , y_train_xgb_res = sm.fit_sample(X_train_var_xgb,y_train)
X_train_xgb_res

X_test_var_xgb = X_test_var.copy()
X_test_var_xgb['DICHARGE_HOUR_BAND_ENC'] = X_test_var_xgb.DICHARGE_HOUR_BAND_ENC.astype('int64')
X_test_var_xgb

xgb_estimator = xgb.XGBClassifier(random_state=2017)
xgb_param_grid = dict(learning_rate = [0.1,0.01,0.2,0.3,0.4,0.5,0.6], max_depth = [1,2,3],n_estimators = range(50,2000,50))
xgb_param_grid_test = {'learning_rate': [0.3], 'max_depth': [1], 'n_estimators': [100]}

xgb_grid_estimator = model_selection.GridSearchCV(xgb_estimator, xgb_param_grid_test, cv =10 , verbose = 1 ,scoring = "roc_auc", n_jobs = 15)
xgb_grid_estimator.fit(X_train_xgb_res,y_train_xgb_res)
print(xgb_grid_estimator.best_score_)
print(xgb_grid_estimator.score(X_train_xgb_res, y_train_xgb_res))
print(xgb_grid_estimator.best_params_)
print(xgb_grid_estimator.best_estimator_)

[clm for clm in list(X_train_xgb_res.columns) if clm not in list(X_test_var_xgb.columns)]

y_pred_xgb = xgb_grid_estimator.predict(X_test_var_xgb)
y_pred_xgb_prob = xgb_grid_estimator.predict_proba(X_test_var_xgb)[:,1]


metrics.accuracy_score(y_test,y_pred_xgb)
metrics.roc_auc_score(y_test,y_pred_xgb_prob)
metrics.confusion_matrix(y_test,y_pred_xgb)

#Stacking
       
clf1 =  ensemble.RandomForestClassifier(criterion =  'gini', max_features = 44, n_estimators =  250,n_jobs=4, random_state=2017, max_depth = 10)
clf2 = ensemble.GradientBoostingClassifier(max_depth = 3, n_estimators = 100,random_state=2017)
clf3 = xgb.XGBClassifier(random_state=2017,learning_rate = 0.3, max_depth = 1, n_estimators = 100)
lr = linear_model.LogisticRegression( penalty = 'l2',random_state=2017)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)#,use_probas=True,average_probas=False)

stack_grid = {'meta-logisticregression__C' : [10 ** a for a in range(-6, 7)]}
stack_grid_test = {'meta-logisticregression__C' : [0.001]}

stack_grid_estimator = model_selection.GridSearchCV(estimator=sclf, param_grid=stack_grid_test,                                                  cv=10,refit=True,scoring = "roc_auc",verbose = 1)

stack_grid_estimator.fit(X_train_scl,y_train)
print(stack_grid_estimator.best_score_)
print(stack_grid_estimator.score(X_train_scl, y_train))
print(stack_grid_estimator.best_params_)
print(stack_grid_estimator.best_estimator_)

y_pred_stack = stack_grid_estimator.predict(X_test_scl)
y_pred_stack_prob = stack_grid_estimator.predict_proba(X_test_scl)[:,1]

metrics.accuracy_score(y_test,y_pred_stack)
metrics.roc_auc_score(y_test,y_pred_stack_prob)
metrics.confusion_matrix(y_test,y_pred_stack)


#Voting ensemble

rf =  ensemble.RandomForestClassifier(criterion =  'gini', max_features = 44, n_estimators =  250,n_jobs=4, random_state=2017, max_depth = 10)
gb = ensemble.GradientBoostingClassifier(max_depth = 3, n_estimators = 100,random_state=2017)
xgb = xgb.XGBClassifier(random_state=2017,learning_rate = 0.3, max_depth = 1, n_estimators = 100)
lr = linear_model.LogisticRegression( penalty = 'l2',random_state=2017)

voting_grid = dict()   
v_estimator1 = ensemble.VotingClassifier([('gb',gb), ('rf',rf), ('xgb',xgb)],voting= 'soft',weights = [2,1,1.5])
voting_grid_estimator = model_selection.GridSearchCV(estimator=v_estimator1, param_grid=voting_grid, cv=10,scoring = "roc_auc",verbose = 1)
voting_grid_estimator.fit(X_train,y_train)
print(voting_grid_estimator.best_score_)
print(voting_grid_estimator.score(X_train, y_train))
print(voting_grid_estimator.best_params_)
print(voting_grid_estimator.best_estimator_)

y_pred_vote = voting_grid_estimator.predict(X_test)
y_pred_vote_prob = voting_grid_estimator.predict_proba(X_test)[:,1]

metrics.accuracy_score(y_test,y_pred_vote)
metrics.roc_auc_score(y_test,y_pred_vote)
metrics.confusion_matrix(y_test,y_pred_vote)
