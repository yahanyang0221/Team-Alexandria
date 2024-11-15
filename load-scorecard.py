import sys
import pandas as pd
from constants import score_institution_columns, score_loan_columns
from constants import score_graduation_columns, score_faculty_columns
from constants import score_admission_columns, score_tuition_columns
import psycopg
from psycopg import sql, connect, errors


def load_data(file_path, relevant_columns):
    # load only the columns we care about
    df = pd.read_csv(file_path, usecols=relevant_columns)
    return df


def preprocess_data(df):
    # replace -999 with None
    df.replace(-999, None, inplace=True)

    # create 'SATCMMID'
    if {'SATVRMID', 'SATMTMID', 'SATWRMID'}.issubset(df.columns):
        df['SATCMMID'] = df[['SATVRMID', 'SATMTMID',
                             'SATWRMID']].sum(axis=1, skipna=True)
        df.drop(columns=['SATVRMID', 'SATMTMID', 'SATWRMID'], inplace=True)

    return df


def insert_data(df, table_columns, table_name, conn):
    cursor = conn.cursor()

    columns = ', '.join(table_columns)
    placeholders = ', '.join(['%s'] * len(table_columns))

    insert_query = (
        f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        )

    for _, row in df[table_columns].iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()


def main(file_path):
    # relevant columns needed from scorecard data
    relevant_columns = list(set(
        score_institution_columns + score_loan_columns +
        score_graduation_columns + score_faculty_columns +
        score_admission_columns + score_tuition_columns))

    df = load_data(file_path, relevant_columns)
    df = preprocess_data(df)

    conn = psycopg.connect(
        host="pinniped.postgres.database.azure.com", dbname="rtshah",
        user="rtshah", password="Hp7HklBKM0")

    try:
        # Insert into each table based on available columns
        if set(score_institution_columns).intersection(df.columns):
            insert_data(df, score_institution_columns, 'institution', conn)

        if set(score_loan_columns).intersection(df.columns):
            insert_data(df, score_loan_columns, 'loan', conn)

        if set(score_graduation_columns).intersection(df.columns):
            insert_data(df, score_graduation_columns, 'graduation', conn)

        if set(score_faculty_columns).intersection(df.columns):
            insert_data(df, score_faculty_columns, 'faculty', conn)

        if set(score_admission_columns).intersection(df.columns):
            insert_data(df, score_admission_columns, 'admission', conn)

        if set(score_tuition_columns).intersection(df.columns):
            insert_data(df, score_tuition_columns, 'tuition', conn)

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
