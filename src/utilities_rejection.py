import re
import numpy as np

class DataFrameRejection:
  def __init__(self, df):
    self.df = df


  def map_rejection_label(self):
    df = self.df

    medical_pattern = r"MN-1-[1-2]"
    admin_pattern   = r"AD-[1-3]-[1-8]"
    cover_pattern   = r"CV-[1-3]-(10|[1-9])"
    support_pattern = r"SE-1-(10|[1-9])"
    billing_pattern = r"BE-1-(10|[1-9])"

    # Iterate through each row in the DataFrame and perform the label assignment
    labels = []
    for i in range(len(df)):
        note1 = df['NOTES'].iloc[i]
        if re.search(medical_pattern, note1):
            label_nphies = 'Medical'
        elif re.search(admin_pattern, note1):
            label_nphies = 'Administrative'
        elif re.search(cover_pattern, note1):
            label_nphies = 'Coverage'
        elif  re.search(support_pattern, note1):
            label_nphies = 'Support'
        elif  re.search(billing_pattern, note1):
            label_nphies = 'Billing'
        else:
            label_nphies = 'NoLabel'
        labels.append(label_nphies)

    df['NPHIES_LABEL'] = labels
    self.df = df
    return df

  def get_label_rejection(self):
      df_label = self.df
      labels = []
      for i in range(len(df_label)):
        label_prep = df_label['NPHIES_LABEL'].iloc[i]
        if label_prep == 'Medical':
          label = 1
        else:
          label = 0
        labels.append(label)
      df_label['NPHIES_LABEL'] = labels
      return df_label


class RejectionReasonLabeling:
    def __init__(self, df):
        self.df = df
        self.df['NPHIES_CODE'] = None

    def recoginze_label(self):
      df = self.df

      patterns = {
          'medical': r"MN-1-[1-2]",
          'admin': r"AD-[1-3]-[1-8]",
          'cover': r"CV-[1-3]-(10|[1-9])",
          'support': r"SE-1-(10|[1-9])",
          'billing': r"BE-1-(10|[1-9])"  }
      list_labels = [None] *len(df)

      for i in range(len(df)):
          text = str(df['NOTES'].iloc[i])
          for label, pattern in patterns.items():
                match1 = re.search(pattern, text)
                if match1:
                    list_labels[i] = match1.group(0)
      df['NPHIES_CODE'] = list_labels
      self.df = df

    def recognize_service(self,df_original):
        df = self.df
        df = df[list(df_original.columns)]

        list_labels = [0] * len(df)
        for i in range(len(df)):
            row_df = df.iloc[0]
            visit = row_df['VISIT_ID']; service = row_df['SERVICE_DESCRIPTION']

            sub_df = df_original[df_original['VISIT_ID']==visit]
            counter = list(sub_df['SERVICE_DESCRIPTION']).count(service)

            if counter > 1:
                list_labels[i] = 1

        self.df['DUPLICATED_SERVICE'] = list_labels

def label_reasons(df_train, df_test, reasons):
    df_train['NPHIES_LABEL'] = 0; df_test['NPHIES_LABEL'] =0
    for i in range(len(reasons)):
        df_train['NPHIES_LABEL'] = np.where(df_train['NPHIES_CODE'] == reasons[i], i+2, df_train['NPHIES_LABEL'])
        df_test['NPHIES_LABEL'] = np.where(df_test['NPHIES_CODE'] == reasons[i], i+2, df_test['NPHIES_LABEL'])

    return df_train, df_test

def label_reasons_column(y):
    outcomes_train = []
    for i in range(len(y)):
        if y['NPHIES_LABEL'].iloc[i] > 1:
            outcomes_train.append( y['NPHIES_LABEL'].iloc[i] )
        else:
            val = y['OUTCOME'].iloc[i]
            outcomes_train.append(val)
    return outcomes_train
