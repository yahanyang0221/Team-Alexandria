
# tabl 1: institution
institution_columns = [
    "UNITID", "OPEID", "INSTNM", "ACCREDAGENCY", "INST_CONTROL", "REGION",
    "MAIN", "NUMBRANCH", "PREDEG", "HIGHDEG", "ZIP", "CITY", "ADDR",
    "LATITUDE", "LONGITUD", "FIPS", "CBSA", "CBSATYPE", "CSA", "C21BASIC",
    "C21IPUG", "C21IPGRD", "C21UGPRF", "C21ENPRF", "C21ZSZET", "CREATED_AT",
    "UPDATED_AT"
]

hd_institution_columns = [
    "UNITID", "OPEID", "INSTNM", "LATITUDE", "LONGITUD", "FIPS", "CBSA",
    "CBSATYPE", "CSA", "C21BASIC", "C21IPUG", "C21IPGRD", "C21UGPRF",
    "C21ENPRF", "C21SZSET"]

score_institution_columns = [
    "UNITID", "OPEID", "INSTNM", "ACCREDAGENCY", "CONTROL", "REGION",
    "MAIN", "NUMBRANCH", "HIGHDEG", "ZIP", "CITY", "ADDR", "PREDDEG",
    "HIGHDEG"]

# table 2 : loan
loan_columns = ["LOAN_ID", "OPEID", "YEAR", "CDR2", "CDR3", "DBRR5_FED_UG_RT"]
score_loan_columns = ["OPEID", "YEAR", "CDR2", "CDR3", "DBRR5_FED_UG_RT"]


# table 3:  graduation
graduation_columns = [
    "GRADUATION_ID", "OPEID", "YEAR", "UGNONDS", "GRADS", "MD_EARN_WNE_4YR"]
score_graduation_columns = [
    "OPEID", "UGNONDS", "GRADS", "MD_EARN_WNE_4YR", "YEAR"]

# table 4: faculty
faculty_columns = ["FACULTY_ID", "OPEID", "AVGFACSAL", "YEAR"]
score_faculty_columns = ["OPEID", "YEAR", "AVGFACSAL"]

# table 5: admission
admission_columns = [
    "ADMISSION_ID", "OPEID", "YEAR", "ADM_RATE", "SATVRMID",
    "SATMTMID", "SATWRMID"]
score_admission_columns = [
    "OPEID", "YEAR", "ADM_RATE", "SATVRMID",
    "SATMTMID", "SATWRMID", "ACTCMMID", "ADMCON7"]

# table 6:
tuition_columns = [
    "TUITION_ID", "OPEID", "YEAR", "TUITIONFEE_IN", "TUITIONFEE_OUT",
    "TUITIONFEE_PROG", "TUITFTE", "COSTT4_A", "COSTT4_P"]
score_tuition_columns = [
    "OPEID", "YEAR", "TUITIONFEE_IN", "TUITIONFEE_OUT",
    "TUITIONFEE_PROG", "TUITFTE", "COSTT4_A", "COSTT4_P"]
