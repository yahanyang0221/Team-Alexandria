import sys
import pandas as pd
from constants import hd_institution_columns, hd_loan_columns
from constants import hd_graduation_columns, hd_faculty_columns
from constants import hd_admission_columns, hd_tuition_columns
import psycopg2
from psycopg2 import sql, connect, errors
import re
from datetime import datetime
import numpy as np
from psycopg2.extras import execute_batch, execute_values
from credentials import user, password, host, dbname
import gc


# def load_data(file_path, relevant_columns):
#     """
#     Load the relevant columns from the CSV file into a Pandas DataFrame.
#     """
#     # Ensure 'YEAR' is not in relevant_columns since it's not part of the original file
#     relevant_columns = [col for col in relevant_columns if col != 'YEAR']

#     try:
#         # Attempt to read the CSV using 'utf-8' encoding first
#         df = pd.read_csv(file_path, usecols=relevant_columns, encoding='utf-8')
#     except UnicodeDecodeError:
#         # If 'utf-8' fails, try 'ISO-8859-1' or 'latin1'
#         df = pd.read_csv(file_path, usecols=relevant_columns, encoding='ISO-8859-1')

#     return df

def load_data(file_path, relevant_columns):
    """
    Load the relevant columns from the CSV file into a Pandas DataFrame.
    Only include columns that are present in the file.
    """
    try:
        # Attempt to read the CSV using 'utf-8' encoding first
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If 'utf-8' fails, try 'ISO-8859-1' or 'latin1'
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Keep only columns that match relevant_columns
    df = df[[col for col in relevant_columns if col in df.columns]]

    return df


def extract_year_from_filename(file_path):
    """extract the year from the IPEDS filename
    (e.g., HD_2018.csv -> 2018).
    
    Args: 
    file_path (str): file path

    Returns: 
    year (int)
    """
    match = re.search(r'(\d{4})', file_path)
    if match:
        return int(match.group(1))  # extracts 4-digit year
    return None


def handle_empty_string(value):
    return None if value == '' else value


def preprocess_data(df, file_path):
    """
    Preprocess the dataframe:
    - Extract the year from the file name and add it as a column.
    - Handle missing values, data types, and more.
    """
    # Extract the year from the file name
    year = extract_year_from_filename(file_path)
    if year:
        df['YEAR'] = year

    # Replace -999 with None
    df.replace(-999, None, inplace=True)

    # Add CREATED_AT and UPDATED_AT columns
    df['CREATED_AT'] = datetime.today().strftime('%Y-%m-%d')
    df['UPDATED_AT'] = datetime.today().strftime('%Y-%m-%d')

    # Replace empty strings with NaN
    df.replace(" ", np.nan, inplace=True)

    # Handle big integer values
    BIGINT_MIN = -9223372036854775808
    BIGINT_MAX = 9223372036854775807

    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert non-numeric entries to NaN
        df[column] = df[column].apply(
            lambda x: np.nan if pd.notnull(x) and (x < BIGINT_MIN or x > BIGINT_MAX) else x
        )

    # Drop rows with NaN values in important columns
    df.dropna(subset=['UNITID', 'OPEID'], inplace=True)

    print(f"Rows to insert: {df.shape[0]}")
    print("Preprocessing completed.")
    return df


# def insert_data_batch(df, table_columns, table_name, conn, batch_size=100):
#     """
#     Insert data into the specified table in batches.
#     """
#     # Generate SQL query
#     insert_query = f"INSERT INTO {table_name} ({', '.join(table_columns)}) VALUES %s"

#     try:
#         with conn.cursor() as cursor:
#             for i in range(0, len(df), batch_size):
#                 # Generate batch dynamically to avoid memory issues
#                 batch_df = df.iloc[i:i + batch_size]
#                 batch = [tuple(row) for _, row in batch_df[table_columns].iterrows()]

#                 # Log batch details for debugging
#                 print(f"Inserting batch {i} to {i + batch_size}: {batch}")

#                 # Use execute_values for efficient batch insertion
#                 execute_values(cursor, insert_query, batch)

#                 # Commit the batch to the database
#                 conn.commit()

#                 # Explicitly free memory for the batch
#                 del batch
#                 # gc.collect()  # Force garbage collection
#                 print(f"Batch {i} inserted successfully")

#     except Exception as e:
#         print(f"Error during batch insert into {table_name}: {e}")
#         conn.rollback()


def insert_data_batch(df, table_columns, table_name, conn, batch_size=100):
    """
    Insert data into the specified table in batches.
    Dynamically adjusts for missing columns.
    """
    # Filter columns to exclude the problematic C21 columns
    c21_columns = ['C21BASIC', 'C21IPUG', 'C21IPGRD', 'C21UGPRF', 'C21ENPRF', 'C21SZSET']
    available_columns = [col for col in table_columns if col in df.columns and col not in c21_columns]

    insert_query = f"INSERT INTO {table_name} ({', '.join(available_columns)}) VALUES %s"

    try:
        with conn.cursor() as cursor:
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                batch = [tuple(row[col] for col in available_columns) for _, row in batch_df.iterrows()]

                print(f"Inserting batch {i} to {i + batch_size}...")
                execute_values(cursor, insert_query, batch)
                conn.commit()

                print(f"Batch {i} inserted successfully")

    except Exception as e:
        print(f"Error during batch insert into {table_name}: {e}")
        conn.rollback()


def insert_data(df, table_columns, table_name, conn):
    """
    inserting the data into rows individually (all tables except institution) 
    this function was used for testing

    Args:
    df (pandas dataframe)
    table_columns (list): relevant columns in table
    table_name (str): name of table
    conn (psycog postgres connection)
    batch_size (int)
    
    """
    cursor = conn.cursor()

    columns = ', '.join(table_columns)
    placeholders = ', '.join(['%s'] * len(table_columns))

    insert_query = (f"""INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})""")

    for _, row in df[table_columns].iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()


def insert_institution(df, conn):
    """
    inserting the data into main institution table by individual row 
    used for testing

    Args:

    df(pandas dataframe)
    conn (psycog postgres connecction) 
    """
    cursor = conn.cursor()

    for index, row in df.iterrows():
        if index % 100 == 0: print(index)
        try:
            cursor.execute("""
                INSERT INTO Institution (
                    UNITID, OPEID, INSTNM, LATITUDE, LONGITUD,
                    FIPS, CBSA, CBSATYPE, CSA, C21BASIC, C21IPUG,
                    C21IPGRD, C21UGPRF, C21ENPRF, C21SZSET,
                    CREATED_AT, UPDATED_AT
                ) VALUES ( %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (OPEID) DO UPDATE SET
                    INSTNM = EXCLUDED.INSTNM,
                    LATITUDE = EXCLUDED.LATITUDE,
                    LONGITUD = EXCLUDED.LONGITUD,
                    FIPS = EXCLUDED.FIPS,
                    CBSA = EXCLUDED.CBSA,
                    CBSATYPE = EXCLUDED.CBSATYPE,
                    CSA = EXCLUDED.CSA,
                    C21BASIC = EXCLUDED.C21BASIC,
                    C21IPUG = EXCLUDED.C21IPUG,
                    C21IPGRD = EXCLUDED.C21IPGRD,
                    C21UGPRF = EXCLUDED.C21UGPRF,
                    C21ENPRF = EXCLUDED.C21ENPRF,
                    C21SZSET = EXCLUDED.C21SZSET,
                    UPDATED_AT = NOW()
            """, (
                row.UNITID, row.OPEID, row.INSTNM, row.LATITUDE, row.LONGITUD,
                row.FIPS, row.CBSA, row.CBSATYPE, row.CSA, row.C21BASIC,
                row.C21IPUG, row.C21IPGRD, row.C21UGPRF, row.C21ENPRF, row.C21SZSET
            ))
        except Exception as e:
            # i was getting an integer out of range error at row 5567
            print(f"Error inserting row {index}: {e}")
            conn.rollback()
            continue


def insert_institution_batch(df, conn, batch_size=100):
    """
    Insert data into the Institution table in batches.
    Updates duplicate records using ON CONFLICT DO UPDATE.
    """
    # Drop duplicates in the DataFrame based on the primary key (UNITID)
    df = df.drop_duplicates(subset=['UNITID'])

    cursor = conn.cursor()

    insert_query = """
        INSERT INTO Institution (
            UNITID, OPEID, INSTNM, LATITUDE, LONGITUD,
            FIPS, CBSA, CBSATYPE, CSA, C21BASIC, C21IPUG,
            C21IPGRD, C21UGPRF, C21ENPRF, C21SZSET,
            CREATED_AT, UPDATED_AT
        ) VALUES %s
        ON CONFLICT (UNITID) DO UPDATE SET
            OPEID = EXCLUDED.OPEID,
            INSTNM = EXCLUDED.INSTNM,
            LATITUDE = EXCLUDED.LATITUDE,
            LONGITUD = EXCLUDED.LONGITUD,
            FIPS = EXCLUDED.FIPS,
            CBSA = EXCLUDED.CBSA,
            CBSATYPE = EXCLUDED.CBSATYPE,
            CSA = EXCLUDED.CSA,
            C21BASIC = COALESCE(EXCLUDED.C21BASIC, Institution.C21BASIC),
            C21IPUG = COALESCE(EXCLUDED.C21IPUG, Institution.C21IPUG),
            C21IPGRD = COALESCE(EXCLUDED.C21IPGRD, Institution.C21IPGRD),
            C21UGPRF = COALESCE(EXCLUDED.C21UGPRF, Institution.C21UGPRF),
            C21ENPRF = COALESCE(EXCLUDED.C21ENPRF, Institution.C21ENPRF),
            C21SZSET = COALESCE(EXCLUDED.C21SZSET, Institution.C21SZSET),
            UPDATED_AT = NOW()
    """

    # Prepare data for batch insertion
    data_tuples = [
        (
            row.UNITID, row.OPEID, row.INSTNM, row.LATITUDE, row.LONGITUD,
            row.FIPS, row.CBSA, row.CBSATYPE, row.CSA, row.C21BASIC,
            row.C21IPUG, row.C21IPGRD, row.C21UGPRF, row.C21ENPRF, row.C21SZSET
        )
        for _, row in df.iterrows()
    ]

    try:
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            execute_values(cursor, insert_query, batch,
                           template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())")
            conn.commit()
            print(f"Inserted batch ending at row {i + batch_size}")
    except Exception as e:
        print(f"Error during batch insert: {e}")
        conn.rollback()
    finally:
        cursor.close()


def main(file_path):
    """
    main function which: 
    1. reads data 
    2. preprocesses it 
    3. inserts into all six tables (by batch) 
        ** has error handling
    """

    # Define relevant columns needed from IPEDS data
    relevant_columns = list(set(hd_institution_columns + hd_loan_columns +
                                hd_graduation_columns + hd_faculty_columns +
                                hd_admission_columns + hd_tuition_columns))

    df = load_data(file_path, relevant_columns)
    df = preprocess_data(df, file_path)

    print("data preprocessed")

    conn = psycopg2.connect(
        host=host, dbname=dbname,
        user=user, password=password)

    print("connection established")

    try:
        # if set(hd_institution_columns).intersection(df.columns):
        #     print("institution entered")
        #     # insert_data(df, hd_institution_columns, 'institution', conn)
        #     insert_institution_batch(df, conn)
        #     print("1/6 institution inserted")

        if set(hd_loan_columns).intersection(df.columns):
            print("loan entered")
            insert_data_batch(df, hd_loan_columns, 'loan', conn)
            # insert_data(df, hd_loan_columns, 'loan', conn)
            print("2/6 loan inserted")

        if set(hd_graduation_columns).intersection(df.columns):
            print("grad entered")
            insert_data_batch(df, hd_graduation_columns, 'graduation', conn)
            # insert_data(df, hd_graduation_columns, 'graduation', conn)
            print("3/6 grad inserted")

        if set(hd_faculty_columns).intersection(df.columns):
            print("faculty entered")
            insert_data_batch(df, hd_faculty_columns, 'faculty', conn)
            # insert_data(df, hd_faculty_columns, 'faculty', conn)

        if set(hd_admission_columns).intersection(df.columns):
            print("admission entered")
            insert_data_batch(df, hd_admission_columns, 'admission', conn)
            # insert_data(df, hd_admission_columns, 'admission', conn)

        if set(hd_tuition_columns).intersection(df.columns):
            print("tuition entered")
            insert_data_batch(df, hd_tuition_columns, 'tuition', conn)
            # insert_data(df, hd_tuition_columns, 'tuition', conn)

        print(f"Data from {file_path} successfully loaded.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


# def main(file_path):
#     # Define relevant columns needed from IPEDS data
#     relevant_columns = list(set(hd_institution_columns + hd_loan_columns +
#                                 hd_graduation_columns + hd_faculty_columns +
#                                 hd_admission_columns + hd_tuition_columns))

#     df = load_data(file_path, relevant_columns)
#     expected_institution_columns = hd_institution_columns + ['CREATED_AT', 'UPDATED_AT']

#     df = preprocess_data(df, file_path)

#     conn = psycopg2.connect(
#         host=host, dbname=dbname,
#         user=user, password=password)

#     try:
#         if set(hd_institution_columns).intersection(df.columns):
#             print("Institution data insertion started...")
#             insert_data_batch(df, hd_institution_columns, 'institution', conn)
#             print("Institution data inserted successfully.")

#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load-ipeds.py <filename.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)
