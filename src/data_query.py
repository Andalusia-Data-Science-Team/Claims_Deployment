from datetime import datetime, timedelta
import urllib
import sqlalchemy
import pandas as pd
import json

with open('..\Claims_Rejection\src\data_backup\passcode.json', 'r') as file:
    data_dict = json.load(file)

db_names = data_dict['DB_NAMES']

with open('..\Claims_Rejection\src\data_backup\stored_info.json', 'r') as file:
    table_info_dict = json.load(file)

def get_connection_from_source(source):
    passcodes = db_names[source]
    server, db, uid, pwd, driver = passcodes['Server'],passcodes['Database'],passcodes['UID'],passcodes['PWD'],passcodes['driver']

    conn_str = f'''
    DRIVER={driver};Server={server};Database={db};UID={uid};PWD={pwd};charset="utf-8";
    '''
    return conn_str

def load_query(TABLE_NAME='Claim_Visit', source='BI'):
    """
    :param TABLE_NAME: Claim_Visit,Claim_Service, purchaser_Contract, Insurance_AHJ, Diagnosis
    :param source: BI -> BI-03
    :return: df
    """
    conn_str = get_connection_from_source(source)

    if f'{TABLE_NAME.lower()}_columns' in table_info_dict.keys():
        table_columns = table_info_dict[f'{TABLE_NAME.lower()}_columns']
        columns_string = ", ".join(table_columns)

    else:
        columns_string = '*'


    query = f'''SELECT  {columns_string}  FROM DWH_Claims.dbo.{TABLE_NAME}'''

    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)


def load_merged_query_by_date(FIRST_DATE, LAST_DATE, source='BI'):
    conn_str = get_connection_from_source(source)

    visit_columns = table_info_dict['claim_visit_columns']
    service_columns = table_info_dict['claim_service_columns']
    diagnose_columns = table_info_dict["episode-diagnose_columns"]

    visit_columns_str = ', '.join([f'V.[{col}]' for col in visit_columns])
    service_columns_str = ', '.join([f'S.[{col}]' for col in service_columns])
    diagnose_columns_str = ', '.join([f'E.[{col}]' for col in diagnose_columns])

    query = f"""
        SELECT 
            {visit_columns_str}, 
            {service_columns_str}, 
            {diagnose_columns_str}
        FROM Claim_Visit V
        LEFT JOIN Claim_Service S
            ON V.VISIT_ID=S.VISIT_ID
        LEFT JOIN [dbo].[Episode-Diagnose] E
            ON E.EPISODE_KEY=CONCAT('1_',V.VISIT_NO)
        WHERE 
            (V.[CREATION_DATE] >= '{FIRST_DATE}' AND V.[CREATION_DATE] <= '{LAST_DATE}'
            OR V.[AMEND_LAST_DATE] >= '{FIRST_DATE}' AND V.[AMEND_LAST_DATE] <= '{LAST_DATE}')
    """
    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def load_query_by_date(TABLE_NAME, FIRST_DATE, LAST_DATE, source='BI'):
    conn_str = get_connection_from_source(source)

    if f'{TABLE_NAME.lower()}_columns' in table_info_dict.keys():
        table_columns = table_info_dict[f'{TABLE_NAME.lower()}_columns']
        columns_string = ", ".join(table_columns)
    else:
        columns_string = '*'

    if 'Episode-Diagnose' != TABLE_NAME:
        query = f'''SELECT  {columns_string}  FROM DWH_Claims.dbo.[{TABLE_NAME}]
          WHERE [CREATION_DATE] > '2024-04-30' AND [CREATION_DATE] < '2024-06-12' '''
    else:
        query = f'''SELECT  {columns_string}  FROM DWH_Claims.dbo.[{TABLE_NAME}]'''

    connect_string = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={connect_string}', fast_executemany=True)

    with engine.connect() as connection:
        return pd.read_sql(query, engine)


def update_data(FIRST_DATE, LAST_DATE):
    df_episode  = load_query_by_date(TABLE_NAME='Episode-Diagnose', FIRST_DATE=FIRST_DATE, LAST_DATE=LAST_DATE)
    df_service = load_query_by_date(TABLE_NAME='Claim_Service',FIRST_DATE=FIRST_DATE, LAST_DATE=LAST_DATE)
    df_visit    = load_query_by_date(TABLE_NAME='Claim_Visit',FIRST_DATE=FIRST_DATE, LAST_DATE=LAST_DATE)
    df_diagnose = load_query_by_date(TABLE_NAME='Diagnosis',FIRST_DATE=FIRST_DATE, LAST_DATE=LAST_DATE)
    return df_visit, df_service, df_diagnose, df_episode




def load_claims_bisample(source='BI'):
    
    db_name = db_names[source]

    query = '''
    SELECT *
    FROM [DWH_Claims].[DBO].[Claims_Sample]'''

    df = load_query(query,db_name)
    return df

def load_claims(source='SNB',TABLE_NAME='ClaimTransaction',SAMPLE_SIZE=100):
    db_name = db_names[source]
    query = f'''SELECT TOP({SAMPLE_SIZE}) *  FROM Nphies.{TABLE_NAME}  ORDER BY CreatedDate DESC'''
    df = load_query(query, db_name)
    return df


# df_requests, df_response = load_merged(source='SNB')
#df = load_claims(source='SNB',TABLE_NAME='ClaimTransaction',SAMPLE_SIZE=50)