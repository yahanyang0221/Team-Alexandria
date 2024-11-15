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


def load_data(file_path, relevant_columns):
    # only load the columns we care about
    try:
        # attempt to read the CSV using 'utf-8' encoding first
        df = pd.read_csv(file_path, usecols=relevant_columns, encoding='utf-8')
    except UnicodeDecodeError:
        # if 'utf-8' fails, try 'ISO-8859-1' or 'latin1'
        df = pd.read_csv(file_path, usecols=relevant_columns,
                         encoding='ISO-8859-1')

    return df


def extract_year_from_filename(file_path):
    """Extract the year from the IPEDS filename
    (e.g., 'ipeds_2018.csv' -> 2018)."""
    match = re.search(r'(\d{4})\.csv$', file_path)
    if match:
        return int(match.group(1))  # Extracts the 4-digit year
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

    # add CREATED_AT and UPDATED_AT columns
    df['CREATED_AT'] = '2024-11-05'
    df['UPDATED_AT'] = datetime.today().strftime('%Y-%m-%d')

    # apply handle_empty_string to all relevant columns in the dataframe
    df.replace(" ", np.nan, inplace=True)

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
            lambda x: np.nan if pd.notnull(x) and (x < BIGINT_MIN or x >
                                                   BIGINT_MAX) else x
        )

    # Drop rows where any required fields have NaN values
    # (you can adjust this based on your needs)
    df.dropna(subset=['UNITID', 'OPEID'], inplace=True)

    num_rows = df.shape[0]
    print("rows to insert: ", num_rows)

    return df


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


def insert_data(df, table_columns, table_name, conn):
    cursor = conn.cursor()

    columns = ', '.join(table_columns)
    placeholders = ', '.join(['%s'] * len(table_columns))

    insert_query = (f"INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})")

    for _, row in df[table_columns].iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()


def insert_institution(df, conn):
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
    # Deduplicate based on the conflict key 'OPEID'
    df = df.drop_duplicates(subset=['OPEID'])

    cursor = conn.cursor()

    insert_query = """
        INSERT INTO Institution (
            UNITID, OPEID, INSTNM, LATITUDE, LONGITUD,
            FIPS, CBSA, CBSATYPE, CSA, C21BASIC, C21IPUG,
            C21IPGRD, C21UGPRF, C21ENPRF, C21SZSET,
            CREATED_AT, UPDATED_AT
        ) VALUES %s
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
    """

    # Prepare the data for batch insertion
    data_tuples = [
        (
            row.UNITID, row.OPEID, row.INSTNM, row.LATITUDE, row.LONGITUD,
            row.FIPS, row.CBSA, row.CBSATYPE, row.CSA, row.C21BASIC,
            row.C21IPUG, row.C21IPGRD, row.C21UGPRF, row.C21ENPRF, row.C21SZSET
        )
        for _, row in df.iterrows()
    ]

    try:
        # Insert data in batches
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
    # Define relevant columns needed from IPEDS data
    relevant_columns = list(set(hd_institution_columns + hd_loan_columns +
                                hd_graduation_columns + hd_faculty_columns +
                                hd_admission_columns + hd_tuition_columns))

    df = load_data(file_path, relevant_columns)
    df = preprocess_data(df)

    print("data preprocessed")

    conn = psycopg2.connect(
    host=host, dbname=dbname,
    user=user, password=password)

    print("connection established")

    try:
        if set(hd_institution_columns).intersection(df.columns):
            print("institution entered")
            #insert_data(df, hd_institution_columns, 'institution', conn)
            insert_institution_batch(df, conn)
            #print("1/6 institution inserted")

        if set(hd_loan_columns).intersection(df.columns):
            print("loan entered")
            insert_data_batch(df, hd_loan_columns, 'loan', conn)
            #print("2/6 loan inserted")

        if set(hd_graduation_columns).intersection(df.columns):
            print("grad entered")
            insert_data_batch(df, hd_graduation_columns, 'graduation', conn)
            #print("3/6 grad inserted")

        if set(hd_faculty_columns).intersection(df.columns):
            print("faculty entered")
            insert_data_batch(df, hd_faculty_columns, 'faculty', conn)

        if set(hd_admission_columns).intersection(df.columns):
            print("admission entered")
            insert_data_batch(df, hd_admission_columns, 'admission', conn)

        if set(hd_tuition_columns).intersection(df.columns):
            print("tuition entered")
            insert_data_batch(df, hd_tuition_columns, 'tuition', conn)

        print(f"Data from {file_path} successfully loaded.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load-ipeds.py <filename.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)
