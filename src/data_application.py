import json

model_columns = ['item_CreatedDate','transaction_RequestId','transaction_PatientAge', 'transaction_PatientEnGender','item_NameEn','item_Price','item_Status','item_Sequence','item_RequestQuantity','item_ResponseQuantity','transaction_DiagnosisIds','transaction_PhysicianIds','item_ResponseState']
train_columns = ['transaction_PatientAge','transaction_PatientEnGender', 'item_NameEn', 'item_Price', 'item_Sequence', 'item_RequestQuantity', 'transaction_DiagnosisIds', 'item_ResponseState']

class DataFrameProcessor:
    def __init__(self, df):
        self.df = df

    def _add_list_to_json(self,list_name, values,file_path = 'src/data_backup/stored_info.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}
        data[list_name] = values
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def _read_list_from_json(self,list_name, file_path = 'src/data_backup/stored_info.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get(list_name, None)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the file {file_path}.")
            return None

    def get_types(self):
        df2 = self.df.dtypes.reset_index()
        df2.columns = ['column_name', 'data_type']
        return df2

    def _get_no_variance_cols(self):
        unique_counts = self.df.nunique()
        single_value_columns = unique_counts[unique_counts <= 1].index.tolist()
        return single_value_columns

    def _drop_columns_with_substring(self, substrings:list):
        df2 = self.df.copy()
        for substring in substrings:
            columns_to_drop = [col for col in df2.columns if substring in col]
            df2 = df2.drop(columns=columns_to_drop)
        return df2

    def save_dropped_cols(self,df_index:str,other_columns=[]):
        dropped_columns = self._get_no_variance_cols() + other_columns
        list_name = df_index + '_dropped'
        self._add_list_to_json(list_name=list_name,values=dropped_columns)

    def filter_high_variance_features(self):  ## To keep high variance columns only and remove low varaince ones
        no_vars = list(self._get_no_variance_cols())
        high_var_cols = [col for col in self.df.columns if col not in no_vars]
        self.df = self.df[high_var_cols]
        self.df = self._drop_columns_with_substring('Ar')
        return self.df

    def drop_columns(self,list_name):
        dropped_cols = self._read_list_from_json(list_name=list_name)
        for col in dropped_cols:
            if col in self.df.columns:
                self.df.drop(columns=[col],inplace=True)
        return self.df

    def store_current_columns(self,df_index):
        list_name = df_index + '_columns'
        self._add_list_to_json(list_name=list_name,values=list(self.df.columns))


### Use Case
# processor = DataFrameProcessor(df)
# types_df = processor.get_types()
# df = processor.filter_high_variance_features()
# processor.save_dropped_cols(other_columns)