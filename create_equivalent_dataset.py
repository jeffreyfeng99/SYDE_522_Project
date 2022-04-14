import pandas as pd

if __name__ == '__main__':

    zigong_df = pd.read_csv('./data_zigong/dat.csv')

    uci_df = pd.read_csv('./data_uci/heart_failure_clinical_records_dataset.csv')

    print(zigong_df)
    print(uci_df)

    equivalence_uci_to_zigong_dict = {
        'age': 'ageCat', #int to string. '(int,int]' (categorized in decades)
        'anaemia': 'red.blood.cell', #bool to float. red.blood.cell has reference rance as 3.5-5.5. Below is anemic??  
        'creatinine_phosphokinase': 'creatinine.enzymatic.method', #int to float. different ghings, but mayube noramlize?? 
        'diabetes': 'diabetes', #bool to bool
        'ejection_fraction': 'LVEF', # int to int (may need normalization based on referance ranges)
        'high_blood_pressure': 'systolic.blood.pressure', # bool to int
        'high_blood_pressure': 'diastolic.blood.pressure', # bool to int
        'high_blood_pressure': 'map', # bool to int ####################### might need to analyze blood pressures
        'platelets': 'platelet',  #int to int. need normalization. also mean.platelet.volume
        'serum_creatinine': 'creatinine.enzymatic.method', # float to float. may need normalization
        'serum_sodium': 'sodium', # int to float
        'sex': 'gender', # bool to float. 0 Female, 1 Male.
        'smoking': None, # bool
        'time': 're.admission.time..days.from.admission.', #int to int. some N/A
        'DEATH_EVENT': 'time.of.death..days.from.admission.', #bool to ints some N/A. Also: death.within.28.days. death.within.3.months, death.within.6.months. some are positive enen though time of death is N/A
    }

    print(equivalence_uci_to_zigong_dict)