import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.lstm_encoder import LSTMEmbedding

def read_last_date(file_path='src/data_backup/last_updated_creation_date.txt'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
    return None

def append_last_line(new_line,file_path='src/data_backup/last_updated_creation_date.txt'):
    with open(file_path, 'a') as file:
        file.write('\n' + new_line)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def store_dfs(df_visit, df_service,df_diagnose,df_episode,path_date):
    path_date = path_date + '/'
    path_visit = path_date + 'visit.parquet'; path_service =  path_date + 'service.parquet';path_diag =  path_date + 'diag.parquet'; path_episode = path_date + 'episode.parquet'
    df_visit.to_parquet(path_visit);df_service.to_parquet(path_service);df_diagnose.to_parquet(path_diag);df_episode.to_parquet(path_episode)

def load_stored_dfs(path_date):
    path_date = path_date + '/'
    path_visit = path_date + 'visit.parquet'; path_service =  path_date + 'service.parquet';path_diag =  path_date + 'diag.parquet'; path_episode = path_date + 'episode.parquet'
    df_visit = pd.read_parquet(path_visit); df_service = pd.read_parquet(path_service); df_diagnose = pd.read_parquet(path_diag)
    df_episode = pd.read_parquet(path_episode)
    return df_visit, df_service,df_diagnose, df_episode



class DataLoader:
    def __init__(self, source='HJH'):
        self.source = source
        self.PATH = 'data/'+source +'/'

    def load_local(self):

        PATH_1 = self.PATH + 'scanned/Claim_Service.xlsx'
        PATH_2 = self.PATH + 'scanned/Claim_Visit.xlsx'

        df_service = pd.read_excel(PATH_1)
        df_visit   = pd.read_excel(PATH_2)

        return df_service, df_visit

    def _drop_duplicates(self,df_visit,service_columns):
        for col in list(df_visit.columns):
            if col != 'VISIT_ID' and col in service_columns:
                df_visit.drop(columns = [col],inplace=True)
        return df_visit

    def _add_prefix_to_columns(self, df, prefix):
        df2 = df.copy()
        df2.columns = [prefix + col for col in df2.columns]
        return df2

    def load_data(self,PATH):
        df_service, df_visit = self.load_sourced(PATH)

        df_visit = self._drop_duplicates(df_visit,list(df_service.columns)) ## drop columns duplications
        return df_service, df_visit

    def load_sourced(self,PATH):
        PATH_1 = PATH + '/service.parquet'
        PATH_2 = PATH + '/visit.parquet'

        df_service = pd.read_parquet(PATH_1)
        df_visit   = pd.read_parquet(PATH_2)

        return df_service, df_visit

    def merge_visit_service(self,df_service=None, df_visit=None):
        if df_service is None or df_visit is None: ## reload is required
            df_service, df_visit = self.load_data()
        merged_df = pd.merge(df_service, df_visit, left_on='VISIT_ID',how='left') ## merging on VISIT_ID
        return merged_df


def self_drop_duplicates(df_visit, service_columns):
    for col in list(df_visit.columns):
        if col != 'VISIT_ID' and col in service_columns:
            df_visit.drop(columns=[col], inplace=True)
    return df_visit



class MergedDataPreprocessing:
    def __init__(self, df):
        self.df = df
        self.lstm_embedding = LSTMEmbedding()

    def _add_list_to_json(self,list_name, values,file_path = 'src/data_backup/label_encoding_items.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        data[list_name] = values
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _read_list_from_json(self,column_name, file_path = 'src/data_backup/label_encoding_items.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get(column_name, None)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {file_path}.")
            return None

    def _truncate_column_values(self, column):
        '''
        :param column: date column value
        :return: utputs only yy/mm
        '''
        df = self.df.copy()
        df[column] = df[column].astype(str).str[:7]

        return df

    def _eliminate_null_claims(self,df_sorted):
        return df_sorted[df_sorted['OUTCOME'].notnull()] ## assert 'APPROVED' or 'PARTIAL'

    def train_test_split(self, id_column='VISIT_NO', test_size=0.2, random_state=None):
        df = self.df
        df = self._eliminate_null_claims(df) ## assert 'Accepted' and 'Rejected' cases

        unique_ids = df[id_column].unique()
        train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

        train_data = df[df[id_column].isin(train_ids)]
        test_data = df[df[id_column].isin(test_ids)]

        return train_data, test_data

    def train_test_split_time(self,id_column = 'CREATION_DATE',test_size= 0.2):
        df = self.df
        df = self._eliminate_null_claims(df) ## assert 'Accepted' and 'Rejected' cases
        df = df.sort_values([id_column])
        train_size = 1 - test_size
        split_time = int(len(df) * train_size)

        train_data, test_data = df[:split_time], df[split_time:]

        return train_data, test_data

    def _label_encode_column(self, column_name, min_count, replace_value='Other'):
        df = self.df.copy()
        counts = df[column_name].value_counts()
        values_to_replace = counts[counts < min_count].index
        df[column_name] = df[column_name].replace(values_to_replace, replace_value)

        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])

        return df[column_name].values


    def _preprocess_service(self,eng_sentence):
        eng_sentence = eng_sentence.split('-')[0]
        return eng_sentence

    def _replace_strings_in_column(self, undefined_inp:str,replacement_value = 0):
        if type(undefined_inp) == str:
            undefined_inp = replacement_value
        return undefined_inp

    def columns_prep(self, categorical_columns=False):
        df = self.df
        LIST_ENCODED_COLS = ["PATIENT_GENDER","ICD10","EMERGENCY_INDICATOR","PATIENT_NATIONALITY","PATIENT_MARITAL_STATUS","CLAIM_TYPE","NEW_BORN","TREATMENT_TYPE"]
        for column in LIST_ENCODED_COLS:
            column_encoding = self._read_list_from_json(column_name=column)
            df[column]= df[column].map(column_encoding)

            if column != 'ICD10': ## Only ICD10 can have string in those columns
                df[column] = df[column].apply(self._replace_strings_in_column)

        if categorical_columns:
            cols_cats = ['DOCTOR_SPECIALTY_CODE', 'DOCTOR_CODE', 'DEPARTMENT_TYPE', 'PURCHASER_CODE', 'CONTRACT_NO', 'TREATMENT_TYPE_INDICATOR']
            for col in cols_cats:
                    df[col] = df[col].astype(str)

        self.df = df

        return self.df

    def column_embedding(self, df1,drop_after_processing):

        textual_col = ['OASIS_IOS_DESCRIPTION','OASIS_IOS_CODE', 'SERVICE_TYPE','UNIT',
                       'UNIT_TYPE','TIMES','PER', 'PROVIDER_DEPARTMENT']

        df1['CombinedText'] = df1[textual_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        arr2 = self.lstm_embedding.embedding_vector(df1['CombinedText'].tolist(), reload_model=True)
        new_cols_names = ['CombinedText' + str(i + 1) for i in range(arr2.shape[1])]
        df2 = pd.DataFrame(arr2)
        df2.columns = new_cols_names
        for col in df2.columns:
            df1.loc[:, col] = df2[col].values
        to_drop = textual_col + ['CombinedText']

        textual_col = ['ICD10']
        df1['ICDText'] = df1[textual_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        arr2 = self.lstm_embedding.embedding_vector(df1['ICDText'].tolist(), reload_model=True)
        new_cols_names = ['ICDText' + str(i + 1) for i in range(arr2.shape[1])]
        df2 = pd.DataFrame(arr2)
        df2.columns = new_cols_names
        for col in df2.columns:
            df1.loc[:, col] = df2[col].values
        to_drop = to_drop + textual_col + ['ICDText']

        textual_col = ['Chief_Complaint']
        df1['ComplaintText'] = df1[textual_col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        arr2 = self.lstm_embedding.embedding_vector(df1['ComplaintText'].tolist(), reload_model=True)
        new_cols_names = ['ComplaintText' + str(i + 1) for i in range(arr2.shape[1])]
        df2 = pd.DataFrame(arr2)
        df2.columns = new_cols_names
        for col in df2.columns:
            df1.loc[:, col] = df2[col].values
        to_drop = to_drop + textual_col + ['ComplaintText']
        if drop_after_processing:
            df1.drop(columns=to_drop, inplace=True)
        else:
            df1.drop(columns=['CombinedText','ICDText','ComplaintText'], inplace=True)

        return df1

    def fast_icd_encode(self):
        df = self.df
        column_encoding = self._read_list_from_json(column_name='ICD10')
        df['ICD10'] = df['ICD10'].map(column_encoding)
        self.df = df

    def store_current_columns(self,df_index,encoding_values:dict):
        self._add_list_to_json(list_name=df_index,values=encoding_values)

def fast_icd_encode(df):
    encoding = MergedDataPreprocessing(df)
    column_encoding = encoding._read_list_from_json(column_name='ICD10')
    df['ICD10'] = df['ICD10'].map(column_encoding)
    return df