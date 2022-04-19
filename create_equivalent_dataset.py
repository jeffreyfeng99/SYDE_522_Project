import pandas as pd
import numpy as np
import os

# This dictionary is not used. The information is just for reference 
equivalence_uci_to_zigong_dict = {
    'age': 'ageCat', #int to string. '(int,int]' (categorized in decades)
    'anaemia': 'red.blood.cell', #bool to float. red.blood.cell has reference rance as 3.5-5.5. Below is anemic??  
    'creatinine_phosphokinase': None, #int
    'diabetes': 'diabetes', #bool to bool
    'ejection_fraction': 'LVEF', # int to int (may need normalization based on referance ranges)
    'high_blood_pressure': 'systolic.blood.pressure', # bool to int
    'high_blood_pressure': 'diastolic.blood.pressure', # bool to int
    'platelets': 'platelet',  #int to int. need normalization. also mean.platelet.volume
    'serum_creatinine': 'creatinine.enzymatic.method', # float to float. may need normalization
    'serum_sodium': 'sodium', # int to float
    'sex': 'gender', # bool to float. 0 Female, 1 Male.
    'smoking': None, # bool
    'time': 're.admission.time..days.from.admission.', #int to int. some N/A
    'DEATH_EVENT': 'time.of.death..days.from.admission.', #bool to ints some N/A. Also: death.within.28.days. death.within.3.months, death.within.6.months. some are positive enen though time of death is N/A
}

def process_uci(uci_full_df):
    # Take the full uci dataset and reduce it to the features that exist in both uci and zigong

    # Create the new dataframe
    uci_partial_df = pd.DataFrame()

    # Copy some of the features directly
    uci_partial_df['age'] = uci_full_df['age']
    uci_partial_df['anaemia'] = uci_full_df['anaemia']
    uci_partial_df['diabetes'] = uci_full_df['diabetes']
    uci_partial_df['ejection_fraction'] = uci_full_df['ejection_fraction'].astype('float')
    uci_partial_df['high_blood_pressure'] = uci_full_df['high_blood_pressure']
    uci_partial_df['platelets'] = uci_full_df['platelets']
    
    # Creatinine needs scaling by a factor of 100 to convert to umol/L 
    uci_partial_df['creatinine'] = uci_full_df['serum_creatinine']
    uci_partial_df['creatinine'] = uci_partial_df['creatinine'].apply(lambda x: x*100)

    # Copy some of the features directly
    uci_partial_df['sodium'] = uci_full_df['serum_sodium'].astype('float')
    uci_partial_df['sex'] = uci_full_df['sex']
    uci_partial_df['readmissiontime'] = uci_full_df['time'].astype('float')
    
    # Readmission time and death time requires some processing
    readmit_time = uci_partial_df['readmissiontime'].to_list()
    death_occurence = uci_full_df['DEATH_EVENT'].to_list()
    readmit_multiclass = []
    death_multiclass = []
    death_times_all = []

    for i in range(len(readmit_time)):
        '''
        readmission time multiclass
            class 0: readmission in less than 28 days
            class 1: readmission between 28 days and 3 months
            class 2: readmission between 3 months and 6 months
            class 3: readmission over 6 months later
        
        death time multiclass
            class 0: death in less than 28 days
            class 1: death between 28 days and 3 months
            class 2: death between 3 months and 6 months
            class 3: death over 6 months later
            class 4: no death

        death times
            - since we lack information, assume that death occurs at the same time as readmission
            - time of 0 means no death     
        '''

        if readmit_time[i] <= 28:
            readmit_multiclass.append(0)
            if death_occurence[i] == True:
                death_multiclass.append(0)
                death_times_all.append(readmit_time[i])
            else:
                death_multiclass.append(4)
                death_times_all.append(0)

        elif readmit_time[i] <= 90 and readmit_time[i] >= 28:
            readmit_multiclass.append(1)
            if death_occurence[i] == True:
                death_multiclass.append(1)
                death_times_all.append(readmit_time[i])
            else:
                death_multiclass.append(4)
                death_times_all.append(0)

        elif readmit_time[i] <= 180 and readmit_time[i] >= 90:
            readmit_multiclass.append(2)
            if death_occurence[i] == True:
                death_multiclass.append(2)
                death_times_all.append(readmit_time[i])
            else:
                death_multiclass.append(4)
                death_times_all.append(0)

        elif readmit_time[i] >= 180:
            readmit_multiclass.append(3)
            if death_occurence[i] == True:
                death_multiclass.append(3)
                death_times_all.append(readmit_time[i])
            else:
                death_multiclass.append(4)
                death_times_all.append(0)

        else:
            raise NotImplementedError

    uci_partial_df['readmissiontime_multiclass'] = readmit_multiclass
    uci_partial_df['death'] = uci_full_df['DEATH_EVENT']
    uci_partial_df['deathtime'] = death_times_all
    uci_partial_df['deathtime_multiclass'] = death_multiclass

    uci_full_df = pd.merge(uci_partial_df, uci_full_df)
    uci_full_df = uci_full_df.drop(columns=['serum_creatinine','serum_sodium','time','DEATH_EVENT'])
    return uci_partial_df, uci_full_df

def process_zigong(zigong_full_df):
    # Take the full zigong dataset and reduce it to the features that exist in both uci and zigong

    # Create the new dataframe
    zigong_partial_df = pd.DataFrame()

    # Zigong gives an age range. To convert it to an integer value, just take the average of the range
    zigong_partial_df['age'] = zigong_full_df['ageCat']
    zigong_partial_df['age'] = zigong_partial_df['age'].apply(lambda x: (int(x.split(',')[0][1:])+int(x.split(',')[1][:-1]))//2).astype('float')

    # Zigong gives rbc. To conver to anaemia boolean, compare to the physiological range. Less than 3.5 is anemic.
    zigong_partial_df['anaemia'] = zigong_full_df['red.blood.cell']
    zigong_partial_df['anaemia'] = zigong_partial_df['anaemia'].apply(lambda x: 1 if float(x)<3.5 else 0)

    zigong_partial_df['diabetes'] = zigong_full_df['diabetes']

    # Zigong gives left ventrical ejection fraction. About half of the values are missing. Replace with a 'normal' physiological value of 60
    zigong_partial_df['ejection_fraction'] = zigong_full_df['LVEF']
    zigong_partial_df['ejection_fraction'] = zigong_partial_df['ejection_fraction'].replace('NA', 60)
    zigong_partial_df['ejection_fraction'] = zigong_partial_df['ejection_fraction'].replace(np.nan, 60)

    # Zigong gives systolic and diastolic bp. To convert to high pressure boolean, compare to physiological range. Systolic more than 130 OR diastolic more than 80 is high.
    systolic_blood_pressure = zigong_full_df['systolic.blood.pressure'].to_list()
    diastolic_blood_pressure = zigong_full_df['diastolic.blood.pressure'].to_list()
    high_blood_pressure = []
    for i in range(len(systolic_blood_pressure)):
        if systolic_blood_pressure[i] >= 130 or diastolic_blood_pressure[i] >= 80:
            high_blood_pressure.append(1)
        else:
            high_blood_pressure.append(0)
    zigong_partial_df['high_blood_pressure'] = high_blood_pressure

    # Platelets needs scaling to convert to 10^6/L
    zigong_partial_df['platelets'] = zigong_full_df['platelet']
    zigong_partial_df['platelets'] = zigong_partial_df['platelets'].apply(lambda x: x*1000)

    zigong_partial_df['creatinine'] = zigong_full_df['creatinine.enzymatic.method']

    zigong_partial_df['sodium'] = zigong_full_df['sodium']
    
    # Zigong gives string values. Convert 'Female' to 0, and 'Male' to 1
    zigong_partial_df['sex'] = zigong_full_df['gender']
    zigong_partial_df['sex'] = zigong_partial_df['sex'].apply(lambda x: 0 if 'Female' in x else 1)
    
    readmit_28 = zigong_full_df['re.admission.within.28.days'].to_list()
    readmit_3 = zigong_full_df['re.admission.within.3.months'].to_list()
    readmit_6 = zigong_full_df['re.admission.within.6.months'].to_list()
    readmit_time = zigong_full_df['re.admission.time..days.from.admission.'].to_list()

    readmit_multiclass = []
    readmit_times_all = []

    for i in range(len(readmit_28)):
        '''
        readmission time multiclass
            class 0: readmission in less than 28 days
            class 1: readmission between 28 days and 3 months
            class 2: readmission between 3 months and 6 months
            class 3: readmission over 6 months later
            class 4: no readmission

            - time of 0 means no readmission     
        '''

        if ~np.isnan(readmit_time[i]):
            readmit_times_all.append(readmit_time[i])

            # Trust the readmission time over the boolean columns
            if readmit_time[i] <= 28:
                readmit_multiclass.append(0)
            elif readmit_time[i] <= 90 and readmit_time[i] >= 28:
                readmit_multiclass.append(1)
            elif readmit_time[i] <= 180 and readmit_time[i] >= 90:
                readmit_multiclass.append(2)
            elif readmit_time[i] >= 180:
                readmit_multiclass.append(3)
            else:
                raise NotImplementedError

        else:
            if readmit_28[i] == 1:
                readmit_multiclass.append(0)
                readmit_times_all.append(28) 
            elif readmit_3[i] == 1:
                readmit_multiclass.append(1)
                readmit_times_all.append(90) 
            elif readmit_6[i] == 1:
                readmit_multiclass.append(2)
                readmit_times_all.append(180) 
            else:
                readmit_multiclass.append(4)
                readmit_times_all.append(0) # never dead

    zigong_partial_df['readmissiontime'] = readmit_times_all
    zigong_partial_df['readmissiontime_multiclass'] = readmit_multiclass

    death_28 = zigong_full_df['death.within.28.days'].to_list()
    death_3 = zigong_full_df['death.within.3.months'].to_list()
    death_6 = zigong_full_df['death.within.6.months'].to_list()
    death_time = zigong_full_df['time.of.death..days.from.admission.'].to_list()

    death_multiclass = []
    death_times_all = []
    death_occurence = []

    for i in range(len(death_28)):
        '''
        death time multiclass
            class 0: death in less than 28 days
            class 1: death between 28 days and 3 months
            class 2: death between 3 months and 6 months
            class 3: death over 6 months later
            class 4: no death

            - time of 0 means no death     
        '''

        if ~np.isnan(death_time[i]):
            death_times_all.append(death_time[i])
            death_occurence.append(1)

            # Trust the death time over the boolean columns
            if death_time[i] <= 28:
                death_multiclass.append(0)
            elif death_time[i] <= 90 and death_time[i] >= 28:
                death_multiclass.append(1)
            elif death_time[i] <= 180 and death_time[i] >= 90:
                death_multiclass.append(2)
            elif death_time[i] >= 180:
                death_multiclass.append(3)
            else:
                raise NotImplementedError

        else:
            if death_28[i] == 1:
                death_multiclass.append(0)
                death_occurence.append(1)
                death_times_all.append(28) 
            elif death_3[i] == 1:
                death_multiclass.append(1)
                death_occurence.append(1)
                death_times_all.append(90) 
            elif death_6[i] == 1:
                death_multiclass.append(2)
                death_occurence.append(1)
                death_times_all.append(180) 
            else:
                death_multiclass.append(4)
                death_occurence.append(0)
                death_times_all.append(0) # never dead
    
    zigong_partial_df['death'] = death_occurence
    zigong_partial_df['deathtime'] = death_times_all
    zigong_partial_df['deathtime_multiclass'] = death_multiclass
    
    zigong_full_df = zigong_full_df.drop(columns=['Unnamed: 0','ageCat','LVEF','platelet','creatinine.enzymatic.method','sodium','gender','diabetes','inpatient.number'])
    zigong_full_df = pd.concat([zigong_full_df, zigong_partial_df], axis=1)
    cols = ['DestinationDischarge', 'admission.ward','admission.way','occupation','discharge.department','type.of.heart.failure','NYHA.cardiac.function.classification','Killip.grade',
            'type.II.respiratory.failure','consciousness','respiratory.support.','oxygen.inhalation','outcome.during.hospitalization']
    zigong_full_df[cols] = zigong_full_df[cols].apply(lambda x: pd.factorize(x)[0] + 1)
    
    return zigong_partial_df, zigong_full_df

# This dictionary is not used. The information is just for reference 
equivalence_uci_to_zigong_dict = {
    'age': 'ageCat', #int to string. '(int,int]' (categorized in decades)
    'anaemia': 'red.blood.cell', #bool to float. red.blood.cell has reference rance as 3.5-5.5. Below is anemic??  
    'creatinine_phosphokinase': None, #int
    'diabetes': 'diabetes', #bool to bool
    'ejection_fraction': 'LVEF', # int to int (may need normalization based on referance ranges)
    'high_blood_pressure': 'systolic.blood.pressure', # bool to int
    'high_blood_pressure': 'diastolic.blood.pressure', # bool to int
    'platelets': 'platelet',  #int to int. need normalization. also mean.platelet.volume
    'serum_creatinine': 'creatinine.enzymatic.method', # float to float. may need normalization
    'serum_sodium': 'sodium', # int to float
    'sex': 'gender', # bool to float. 0 Female, 1 Male.
    'smoking': None, # bool
    'time': 're.admission.time..days.from.admission.', #int to int. some N/A
    'DEATH_EVENT': 'time.of.death..days.from.admission.', #bool to ints some N/A. Also: death.within.28.days. death.within.3.months, death.within.6.months. some are positive enen though time of death is N/A
}


def normalization(df, keep_fields=[], exclude_fields=[]):
    # perform min-max normalization for the desired columns of data
    assert (len(keep_fields) == 0) ^ (len(exclude_fields) == 0)
    normalized_df = df

    if len(keep_fields) > 0:
        for field in keep_fields:
            normalized_df[field]=(df[field]-df[field].min())/(df[field].max()-df[field].min())
    else:
        for field in df:
            if field not in exclude_fields:
                normalized_df[field]=(df[field]-df[field].min())/(df[field].max()-df[field].min())
    
    return normalized_df

if __name__ == '__main__':

    zigong_full_df = pd.read_csv('./data_zigong/dat.csv')
    uci_full_df = pd.read_csv('./data_uci/heart_failure_clinical_records_dataset.csv')

    zigong_partial_df, zigong_full_df = process_zigong(zigong_full_df)
    uci_partial_df, uci_full_df = process_uci(uci_full_df)

    uci_and_zigong_df = pd.concat([zigong_partial_df, uci_partial_df], ignore_index=True)

    normalized_zigong_full_df = normalization(zigong_full_df, exclude_fields=['readmissiontime_multiclass','deathtime_multiclass'])
    normalized_uci_full_df = normalization(uci_full_df, exclude_fields=['readmissiontime_multiclass','deathtime_multiclass'])

    normalized_uci_df = normalization(uci_partial_df, exclude_fields=['readmissiontime_multiclass','deathtime_multiclass'])
    normalized_zigong_df = normalization(zigong_partial_df, exclude_fields=['readmissiontime_multiclass','deathtime_multiclass'])
    normalized_uci_and_zigong_df = normalization(uci_and_zigong_df, exclude_fields=['readmissiontime_multiclass','deathtime_multiclass'])

    print(normalized_zigong_full_df)
    print(normalized_uci_full_df)
    print(normalized_uci_df)
    print(normalized_zigong_df)
    print(normalized_uci_and_zigong_df)

    os.makedirs('normalized_datasets/', exist_ok=True)
    normalized_zigong_full_df.to_csv('normalized_datasets/normalized_zigong_full_df.csv')
    normalized_uci_full_df.to_csv('normalized_datasets/normalized_uci_full_df.csv')
    normalized_uci_df.to_csv('normalized_datasets/normalized_uci_df.csv')
    normalized_zigong_df.to_csv('normalized_datasets/normalized_zigong_df.csv')
    normalized_uci_and_zigong_df.to_csv('normalized_datasets/normalized_uci_and_zigong_df.csv')
