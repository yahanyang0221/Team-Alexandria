# :school: Team-Alexandria (w College Data)

## Streamlit Dashboard
You can access the Streamlit dashboard [here](http://172.26.121.190:8501) to better understand our data.


## Data Loading Instructions 

This repository contains scripts for loading and preprocessing IPEDS and College Scorecard data into a PostgreSQL database.


### 1. Setup
- Ensure the required Python packages are installed: pandas, psycopg2, numpy, and re
- Update the credentials.py file with your database credentials
- Run create-table.ipynb to create the necessary tables

### 2. Loading Data:
- Use *load_data( )* function to load the CSV file. This function attempts to read the file with ‘utf-8’ encoding and falls back to ‘ISO-8859-1’ or ‘latin1’ if necessary
- *extract_year_from_filename( )* function extracts the year from the filename and adds it to the DataFrame
- Our data includes years from 2019 to 2022

### 3. Preprocessing Data:
- *preprocess_data( )* function handles various preprocessing tasks:
- Adds YEAR, CREATED_AT, and UPDATED_AT columns (for IPEDS data)
- Adds YEAR column (for College Scorecard data)
- Replaces placeholder values (-999) and empty strings with None
- Ensures numeric columns are within the valid range for BIGINT
- Converts specific columns to integer and boolean types as needed (for College Scorecard data)
- Creates a combined SAT score column (for College Scorecard data) 

### 4. Inserting Data:
- Use *insert_data_batch( )* for batch insertion of data into the specified table
- Alternatively, use *insert_data( )* for row-by-row insertion
- *insert_institution( )* and *insert_institution_batch( )* functions handle insertion into the Institution table, with conflict resolution on the OPEID column (for IPEDS data)
- *insert_institution_batch( )* function handles insertion into the Institution table, with conflict resolution on the OPEID column (for College Scorecard data)
- For our updated version, we recommend inserting row by row for "institution" table and batch for others to ensure better error message for debugging.

### 5. Main Function:
- *main( )* function manages the data loading and insertion process:
- Defines the relevant columns needed from the IPEDS/College Scorecard data
- Loads and preprocesses the data
- Inserts data into the appropriate tables 

### 6. Execution
- Run the script:
```
python load-ipeds.py <FILENAME.csv>
python load-scorecard.py <FILENAME.csv>
```
