import sys
import pandas as pd
from constants import *
import psycopg2
from psycopg2 import sql, connect, errors
import re
from datetime import datetime
import numpy as np
from psycopg2.extras import execute_batch, execute_values
from credentials import user, password, host, dbname


def load_data(file_path, relevant_columns):
    # only load the columns we care about
    try:
        # attempt to read the CSV using 'utf-8' encoding first
        df = pd.read_csv(file_path, usecols=relevant_columns, encoding='utf-8')
    except UnicodeDecodeError:
        # if 'utf-8' fails, try 'ISO-8859-1' or 'latin1'
        df = pd.read_csv(file_path, usecols=relevant_columns, encoding='ISO-8859-1')

    return df


def extract_year_from_filename(file_path):
    """Extract year from scorecard filename (e.g., 'MERGED_2021_22_PP.csv' -> 2022)."""
    match = re.search(r'MERGED_(\d{4})_(\d{2})_', file_path)
    if match:
        return int(match.group(2))  # Extracts the second year (e.g., 22 in 2021_22 -> 2022)
    return None


def handle_empty_string(value):
    return None if value == '' else value


def preprocess_data(df):

    # get the year from the filename
    year = extract_year_from_filename(file_path)
    if year:
        df['YEAR'] = year

    # replace -999 with None
    df.replace(-999, None, inplace=True)

    # apply handle_empty_string to all relevant columns in the dataframe
    df.replace(" ", np.nan, inplace=True)

    df.replace([np.inf, -np.inf, np.nan], None, inplace=True)


    for column in df.columns:
        df[column] = df[column].apply(handle_empty_string)


    # big int range
    BIGINT_MIN = -9223372036854775808
    BIGINT_MAX = 9223372036854775807

    # handing BIG INT ERROR
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        # Convert non-numeric entries to NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Cap values to BIGINT range and convert out-of-range values to NaN
        df[column] = df[column].apply(
            lambda x: np.nan if pd.notnull(x) and (x < BIGINT_MIN or x > BIGINT_MAX) else x
        )

    # drop rows where any required fields have NaN values 
    df.dropna(subset=['UNITID', 'OPEID'], inplace=True)


    # preprocessing column types 
    columns_to_coerce = ['SATVRMID', "SATMTMID", "SATWRMID", "ACTCMMID", "COSTT4_A", 
                         "COSTT4_P", "TUITIONFEE_IN", "TUITIONFEE_OUT", "TUITIONFEE_PROG", 
                         "TUITFTE", "AVGFACSAL", "UGNONDS", "GRADS",  "MD_EARN_WNE_4YR"] 
    for column in columns_to_coerce:
        # If the column contains NaN values, replace them before coercion
        df[column] = df[column].fillna(0).astype(int)  # Replace NaN with 0 or any other suitable value

    # preprocessing boolean column
    boolean_columns = ['MAIN']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].replace({1: True, 0: False})


    if all(col in df.columns for col in ['SATVRMID', 'SATMTMID', 'SATWRMID']):
        df['SATCMMID'] = df[['SATVRMID', 'SATMTMID', 'SATWRMID']].sum(axis=1)

    num_rows = df.shape[0]
    print("rows to insert: ", num_rows)

    return df


def insert_data(df, table_columns, table_name, conn):
    cursor = conn.cursor()
    
    columns = ', '.join(table_columns)
    placeholders = ', '.join(['%s'] * len(table_columns))
    
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    for _, row in df[table_columns].iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()


def insert_data_batch(df, table_columns, table_name, conn, batch_size=100):
    cursor = conn.cursor()
    
    columns = ', '.join(table_columns)
    
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES %s"
    
    # Prepare data for batch insertion
    data_tuples = [tuple(row) for _, row in df[table_columns].iterrows()]

    try:
        # Use execute_values for batch insert
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            execute_values(cursor, insert_query, batch)
            conn.commit()
            print(f"Inserted batch ending at row {i + batch_size}")
    except Exception as e:
        print(f"Error during batch insert into {table_name}: {e}")
        conn.rollback()
    finally:
        cursor.close()


def insert_institution_batch(df, conn, batch_size=100):
    # Deduplicate based on the conflict key 'OPEID'
    df = df.drop_duplicates(subset=['OPEID'])

    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO Institution (
            UNITID, OPEID, INSTNM, ACCREDAGENCY, CONTROL,
            REGION, MAIN, NUMBRANCH, ZIP,
            CITY, ADDR, PREDDEG, HIGHDEG
        ) VALUES %s
        ON CONFLICT (OPEID) DO UPDATE SET
            INSTNM = EXCLUDED.INSTNM,
            ACCREDAGENCY = EXCLUDED.ACCREDAGENCY,
            CONTROL = EXCLUDED.CONTROL,
            REGION = EXCLUDED.REGION,
            MAIN = EXCLUDED.MAIN,
            NUMBRANCH = EXCLUDED.NUMBRANCH,
            ZIP = EXCLUDED.ZIP,
            CITY = EXCLUDED.CITY,
            ADDR = EXCLUDED.ADDR,
            PREDDEG = EXCLUDED.PREDDEG,
            HIGHDEG = EXCLUDED.HIGHDEG
    """
    
    # Prepare the data for batch insertion
    data_tuples = [
        (
            row.UNITID, row.OPEID, row.INSTNM, row.ACCREDAGENCY, row.CONTROL,
            row.REGION, row.MAIN, row.NUMBRANCH, row.ZIP, row.CITY,
            row.ADDR, row.PREDDEG, row.HIGHDEG
        )
        for _, row in df.iterrows()
    ]

    try:
        # Insert data in batches
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            execute_values(cursor, insert_query, batch, 
                           template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
            conn.commit()
            print(f"Inserted batch ending at row {i + batch_size}")
    except Exception as e:
        print(f"Error during batch insert: {e}")
        conn.rollback()
    finally:
        cursor.close()



def main(file_path):
    # relevant columns needed from scorecard data
    relevant_columns = list(set(score_institution_columns + score_loan_columns +
                                score_graduation_columns + score_faculty_columns +
                                score_admission_columns + score_tuition_columns))
    
    df = load_data(file_path, relevant_columns)
    df = preprocess_data(df)
    
    conn = psycopg2.connect(
    host=host, dbname=dbname,
    user=user, password=password
)
    
    try:
        # Insert into each table based on available columns
        if set(score_institution_columns).intersection(df.columns):
            print("institution entered")
            insert_institution_batch(df, conn)
        
        if set(score_loan_columns).intersection(df.columns):
            print("loan entered")
            insert_data_batch(df, score_loan_columns, 'loan', conn)

        if set(score_graduation_columns).intersection(df.columns):
            print("graduation entered")
            insert_data_batch(df, score_graduation_columns, 'graduation', conn)

        if set(score_faculty_columns).intersection(df.columns):
            print("faculty entered")
            insert_data_batch(df, score_faculty_columns, 'faculty', conn)

        if set(score_admission_columns).intersection(df.columns):
            print("admission entered")
            new_score_cols = ["OPEID", "ADM_RATE", "SATCMMID", "ACTCMMID", "ADMCON7"]
            insert_data_batch(df, new_score_cols, 'admission', conn)

        if set(score_tuition_columns).intersection(df.columns):
            print("tuition entered")
            insert_data_batch(df, score_tuition_columns, 'tuition', conn)

        print(f"Data from {file_path} successfully loaded.")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load-scorecard.py <filename.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)

