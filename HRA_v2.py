# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:16:44 2017

@author: lakshya.khanna
"""

import os
import pandas as pd
import pandasql as pdsql
import numpy as np
from pandas.util.testing import assert_frame_equal

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
labevents = pd.read_csv('LABEVENTS.csv.gz',compression='gzip',quotechar='"', error_bad_lines=False)
labevents.info()
labevents.head(20)
labevents.FLAG.unique()
a = sorted([str(i) for i in list(labevents.VALUE)])
np.array(a)


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


#Merging the dataframe with admission to get the admission attributes
patients_readmit_admattr = pd.merge(patients_readmitted_heart,admission[['HADM_ID','ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION','MARITAL_STATUS', 'ETHNICITY']] ,how = 'left' , on = ['HADM_ID'])
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
final_dataset.shape
final_dataset.columns
final_dataset.info()
total_missing = final_dataset.isnull().sum()
total_missing[total_missing>0]



###############################################################
#EDA - For DiagnosisGet ICD Codes
patients_readmitted_heart_icd1 = pd.merge(patients_readmitted_heart,diagnosis[['HADM_ID','ICD9_CODE']],how = 'left', on= ['HADM_ID'])
patients_readmitted_heart_icd1.ICD9_CODE.value_counts()
len(patients_readmitted_heart_icd1.ICD9_CODE.unique())
patients_readmitted_heart_icd1.groupby('HADM_ID').agg('count').ICD9_CODE.sort_values(ascending = False)
patients_readmitted_heart_icd1.info()
patients_readmitted_heart_icd1.loc[patients_readmitted_heart_icd1.ICD9_CODE=='4280']


#EDA - Check for patient subsequent admission row id's
admission.loc[admission.SUBJECT_ID==102,['ROW_ID','SUBJECT_ID','sequence']]
admission.loc[admission.SUBJECT_ID==99982,]
admission['sequence'] = admission.sort_values(by =['SUBJECT_ID','ADMITTIME'],ascending = [True,True]).groupby(['SUBJECT_ID']).cumcount() + 1
admission.head()

#Rough
pysql('SELECT count(1) FROM  admission a left outer join admission_count b where a.SUBJECT_ID = b.SUBJECT_ID and b.COUNT > 1;')
