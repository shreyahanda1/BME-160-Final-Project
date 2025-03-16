import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from lifelines import KaplanMeierFitter

# Base URL for cBioPortal API
BASE_URL = "https://www.cbioportal.org/api"

# Define the cancer study ID (Glioblastoma)
STUDY_ID = "gbm_tcga"

# Function to fetch data from cBioPortal API
def fetch_data(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, headers={"Accept": "application/json"})
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching {endpoint}: {response.status_code}")
        return None

# Fetch available studies
studies = fetch_data("studies")
print("Available Studies Sample:")
print(studies[:5])

# Define the correct molecular profile and sample list from previous output
MUTATION_PROFILE_ID = "gbm_tcga_mutations"  # Update this if needed
SAMPLE_LIST_ID = "gbm_tcga_sequenced"  # Update this if needed

# Fetch mutation data
mutation_response = requests.get(
    f"{BASE_URL}/molecular-profiles/{MUTATION_PROFILE_ID}/mutations",
    params={"sampleListId": SAMPLE_LIST_ID, "projection": "DETAILED"},
    headers={"Accept": "application/json"}
)

# Check response
if mutation_response.status_code == 200:
    mutation_data = mutation_response.json()
    mutation_df = pd.DataFrame(mutation_data)
    print("Mutation Data Sample:")
    print(mutation_df.head())
else:
    print(f"Error fetching mutations: {mutation_response.status_code}, {mutation_response.text}")


# Fetch clinical data
clinical_data = fetch_data(f"studies/{STUDY_ID}/clinical-data")

# Convert to DataFrame
if clinical_data:
    clinical_df = pd.DataFrame(clinical_data)
    print("Clinical Data Sample:")
    print(clinical_df.head())

    # tumor type is inside print(clinical_df["clinicalAttributeId"].unique()) and SAMPLE_TYPE
    # output: ['Primary' 'Recurrence']
    # tumor_type = clinical_df[clinical_df["clinicalAttributeId"] == "SAMPLE_TYPE"]["value"].unique()

    # filtering primary tumor and recurring (metastatic) tumor data
    # extract primary tumor sample IDs
    # convert to list
    primary_tumor = clinical_df[(clinical_df["clinicalAttributeId"] == "SAMPLE_TYPE") & 
    (clinical_df["value"] == "Primary")]["sampleId"].tolist()


    # extract recurring (metastatic) tumor sample IDs
    # convert to list
    metastatic_tumor = clinical_df[(clinical_df["clinicalAttributeId"] == "SAMPLE_TYPE") & 
    (clinical_df["value"] == "Recurrence")]["sampleId"].tolist()

# filter mutation data with primary and metastatic
primary_tumor_df = mutation_df[mutation_df["sampleId"].isin(primary_tumor)]
metastatic_tumor_df = mutation_df[mutation_df["sampleId"].isin(metastatic_tumor)]

print(f"\nPrimary{primary_tumor_df}")
print(f"Metastatic{metastatic_tumor_df}")

# count mutations per gene in primary and metastatic tumors
# if the gene is dict, specifically extract "hugoGeneSymbol" for format syntax
primary_tumor_df["gene_symbol"] = primary_tumor_df["gene"].apply(lambda x: x["hugoGeneSymbol"] if isinstance(x, dict) else x)
metastatic_tumor_df["gene_symbol"] = metastatic_tumor_df["gene"].apply(lambda x: x["hugoGeneSymbol"] if isinstance(x, dict) else x)

# value_counts() is used in Pandas to obtain a series containing counts of unique values/genes
# counts the number of gene mutations being expressed in data
primary_mutation_counts = primary_tumor_df["gene_symbol"].value_counts()
metastatic_mutation_counts = metastatic_tumor_df["gene_symbol"].value_counts()

# fill NaN values with 0
mutation_compare_df = pd.DataFrame({
    "Primary Tumors": primary_mutation_counts, 
    "Metastatic Tumors": metastatic_mutation_counts
}).fillna(0)

print(mutation_compare_df)

top_primary = mutation_compare_df.sort_values("Primary Tumors", ascending=False).head(20)
top_metastatic = mutation_compare_df.sort_values("Metastatic Tumors", ascending=False).head(20)

# combining the two dataframes together
top_unique = pd.concat([top_primary, top_metastatic]).index.unique()

# keep only top selected genes
filtered_mutation_compare_df = mutation_compare_df.loc[top_unique]

'''
mutation_compare_df["Total Mutations"] = mutation_compare_df["Primary Tumors"] + mutation_compare_df["Metastatic Tumors"]
mutation_compare_df = mutation_compare_df.sort_values("Total Mutations", ascending=False)
mutation_compare_df = mutation_compare_df.drop(columns = ["Total Mutations"])

top_genes = mutation_compare_df.head(20)
'''

plt.figure(figsize=(12, 6))
filtered_mutation_compare_df.plot(kind="bar", color=["red", "orange"], alpha=0.7)

plt.xticks(rotation=45, fontsize=10)
plt.xlabel("Gene", fontsize=12)
plt.ylabel("Mutation Count", fontsize=12)
plt.title("Mutation Frequency in Primary vs. Metastatic Tumors in Glioblastoma", fontsize=14)
plt.legend(["Primary Tumors", "Metastatic Tumors"], fontsize=12)

#plt.show()

'''
Kaplan-Meier Algorithm:
Starts patients with 100% survival rate. Every time an event occurs and a patient dies, the survival
rate decreases.

Identify mutations that make metastatic GBM worse. Check if other gene mutations worsen survival in metastatic cases.
Ex: Are some mutations making metastatic cases even deadlier?
If metastatic cases has a average 100 day survival rate but TP53 has a average of 50 day survival, 
this is a huge impact.
Why? 
Tells doctors which metastatic cases are more aggressive.
Tells how quick doctors should take action-- which tests/therapies to implement
Maybe they need to implement more aggressive treatment for high-risk patients
If we find a new gene that makes metastasis worse, pharmaceutical companies can develop new drugs that
target that gene.

1. Run Kaplan-Meier Algorithm within metastatic cases
2. If survival gap is huge, this might be evidence that TP53 worsens metastasis.  
'''

# merging the survival data with clinical_df
months = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_MONTHS")
status = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_STATUS")

# convert to DataFrame
# checking if there is something in months and status
# if there is, continue to convert to DataFrame
if months and status:
    months_df = pd.DataFrame(months)
    status_df = pd.DataFrame(status)

    # makes sure there are only numbers in survival time
    months_df = months_df[pd.to_numeric(months_df["value"], errors="coerce").notna()]

    # merging two dataframes, months_df and status_df so each patient has both survival time (months) and survival status (alive/deceased)
    # remove duplicates for patient entries, and keep only the first
    survival_df = months_df.merge(status_df, on="uniquePatientKey", suffixes=("_months", "_status")).drop_duplicates(subset=["uniquePatientKey"], keep="first")

    # renaming for consistency
    survival_df.rename(columns={"value_months": "survival_time", "value_status": "event"}, inplace=True)

    survival_df["survival_time"] = survival_df["survival_time"].astype(float)

    # for tracking event/survival status
    # 1 is for deceased
    # 0 is for alive
    survival_df["event"] = survival_df["event"].apply(lambda x: 1 if str(x).upper() == "DECEASED" else 0)

    #############################################
    print("Survival_df before merging:")
    print(survival_df[["uniquePatientKey", "survival_time", "event"]].head())

    # get patient list
    patient_list = fetch_data(f"studies/{STUDY_ID}/patients")
    if patient_list:
        patient_df = pd.DataFrame(patient_list)

    # standardize ID to remove whitespaces and turn to uppercase so that merge is successful
    patient_df["uniquePatientKey"] = patient_df["uniquePatientKey"].str.upper().str.strip()
    survival_df["uniquePatientKey"] = survival_df["uniquePatientKey"].str.upper().str.strip()

    # get TCGA ID from clinical_df
    clinical_df["patientId_clean"] = clinical_df["sampleId"].str[:12]

    # merge clinical_df with patient_df so that we have a format we can work with
    clinical_df = clinical_df.merge(
        patient_df[["uniquePatientKey", "patientId"]],
        left_on="patientId_clean",
        right_on="patientId",
    )

    # merging uniquePatientKey
    
    if "uniquePatientKey_x" in clinical_df.columns and "uniquePatientKey_y" in clinical_df.columns:
        clinical_df["uniquePatientKey"] = clinical_df["uniquePatientKey_y"].combine_first(clinical_df["uniquePatientKey_x"])
    elif "uniquePatientKey_x" in clinical_df.columns:
        clinical_df.rename(columns={"uniquePatientKey_x": "uniquePatientKey"}, inplace=True)
    elif "uniquePatientKey_y" in clinical_df.columns:
        clinical_df.rename(columns={"uniquePatientKey_y": "uniquePatientKey"}, inplace=True)
    
    # gets rid of unecessary columns and renames for clarity
    clinical_df.drop(columns=["uniquePatientKey_x", "uniquePatientKey_y", "patientId_clean", "patientId"], inplace=True, errors="ignore")
    clinical_df.rename(columns={"uniquePatientKey": "uniquePatientKey_clinical"}, inplace=True)

    # merging survival data
    clinical_df = clinical_df.merge(
        survival_df[["uniquePatientKey", "survival_time", "event"]],
        left_on="uniquePatientKey_clinical",
        right_on="uniquePatientKey",
    )

    print("Survial data after merging")
    clinical_survival_df = clinical_df[["sampleId", "survival_time", "event"]].drop_duplicates()
    print(clinical_survival_df.head())

# succesfully extracted survival data and their uniquePatientKey
# filter clinical data for only metastatic cases
metastatic_clinical_df = clinical_df[clinical_df["sampleId"].isin(metastatic_tumor)]

# delete rows/patients that do not have survival data found
metastatic_clinical_df = metastatic_clinical_df.dropna(subset=["survival_time", "event"])

# drop duplicates so only one row per patient
metastatic_clinical_df = metastatic_clinical_df.drop_duplicates(subset=["sampleId"], keep="first")

# 421 metastatic mutation records were found
# 13 metastatic cases were successfully filtered from clinical_df
# 6 metastatic patients have survival data but no mutation data

print(f"Total metastatic cases and surival time:")
print(metastatic_clinical_df[["sampleId", "survival_time", "event"]].head())

# Check if all metastatic cases have mutation data
metastatic_mutation_cases = metastatic_tumor_df[metastatic_tumor_df["sampleId"].isin(metastatic_clinical_df["sampleId"])]

print(f"Metastatic mutation records:")
print(metastatic_mutation_cases[["sampleId", "gene_symbol"]].head())

# excludes the 6 missing cases
valid_metastatic_cases = metastatic_clinical_df[metastatic_clinical_df["sampleId"].isin(metastatic_mutation_cases["sampleId"])]

# final metastatic cases that will be used for Kaplan-Meier analysis
# 7 patients-- small sample size
# We have a small sample size because we removed patients with missing survival data and excluded patients without mutation data
# Limitation: for future use of this code, would like to have access to a bigger dataset with more patient samples
print(f"Final metastatic cases for Kaplan-Meier analysis: {valid_metastatic_cases.shape[0]}")
print(valid_metastatic_cases[["sampleId", "survival_time", "event"]].head())

# count mutation occurences in metastatic cases
# these are patients with survival data from cBioPortal
# one patient can have multiple mutations
all_mutated_genes = metastatic_mutation_cases["gene_symbol"].unique()
print("All mutated genes:")
print(all_mutated_genes)

plt.figure(figsize=(12, 8))

for i in all_mutated_genes:
    # filter cases with this specific mutation
    patients_mutations = metastatic_mutation_cases[metastatic_mutation_cases["gene_symbol"] == i]["sampleId"].unique()

    # get the survival data from patients with this gene mutation
    survival_data = metastatic_clinical_df[metastatic_clinical_df["sampleId"].isin(patients_mutations)]

    # plot the Kaplan-Meier Algorithm
    if survival_data.shape[0] > 1:
        kmf = KaplanMeierFitter()
        kmf.fit(survival_data["survival_time"], event_observed=survival_data["event"], label=i)
        kmf.plot(alpha=0.7)

plt.xlabel("Time (Days)")
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Survival Analysis for Mutated Genes in Metastatic Cases")
#plt.show()

# found that no deaths occured, meaning all patients in the dataset are alive. this leads to the Kalpan-Meier curve consistently being at 1.0 for all genes
# if no one dies, the plot won't drop

############################################# DEBUGGING #####################################################################
print(valid_metastatic_cases[['sampleId', 'survival_time', 'event']])
print("Number of events (deceased patients):", valid_metastatic_cases["event"].sum())
print("Survival time range:", valid_metastatic_cases["survival_time"].min(), "-", valid_metastatic_cases["survival_time"].max())

print(metastatic_clinical_df["event"].value_counts())
print(survival_df[["uniquePatientKey", "event"]].value_counts())
print(survival_df["event"].unique())  # Check unique values
print(survival_df["event"].dtype)  # Ensure it's numeric (not string)
#print(survival_df["value_status"].unique())  # Are "DECEASED" values present?

print(f"🔍 New survival time range: {survival_df['survival_time'].min()} - {survival_df['survival_time'].max()}")

print("🔍 Unique values in 'event':")
print(survival_df["event"].unique())  # Should contain 0 and 1

print("\n🔍 Count of deceased patients (event = 1):")
print(survival_df["event"].value_counts())  # Should show count of 1s and 0s

#################
status = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_STATUS")
status_df = pd.DataFrame(status)

print("🔍 Unique values in raw 'status_df[value]':")
print(status_df["value"].unique())

print("🔍 Checking status_df before conversion:")
print(status_df.head())  # Show first few rows

survival_df["event"] = survival_df["event"].apply(lambda x: 1 if str(x).upper() == "DECEASED" else 0)

print("🔍 Unique values in survival_df['event'] before conversion:")
print(survival_df["event"].value_counts())

all_attributes = fetch_data(f"studies/{STUDY_ID}/clinical-attributes")
attributes_df = pd.DataFrame(all_attributes)

print("🔍 Available Clinical Attributes:")
print(attributes_df[["clinicalAttributeId", "displayName"]])

# Fetch all clinical attributes for the study
clinical_attributes = fetch_data(f"studies/{STUDY_ID}/clinical-attributes")

if clinical_attributes:
    clinical_attr_df = pd.DataFrame(clinical_attributes)
    print("🔍 Available Clinical Attributes:")
    print(clinical_attr_df[["clinicalAttributeId", "displayName"]])  # Show relevant columns
else:
    print("⚠️ No clinical attributes found!")

# had an issue. i was using the incorrect dataset from cBioPortal to track alive and deceased patients
# that is why my algorithm was not chaning and remaining stagnant for all genes
# had to go back and find the correct datasets


# Fetch survival-related attributes again
months_raw = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_MONTHS")
status_raw = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_STATUS")
death_raw = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=DAYS_TO_DEATH")
followup_raw = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=DAYS_TO_LAST_FOLLOWUP")

# Convert to DataFrame
if months_raw and status_raw and death_raw and followup_raw:
    months_df = pd.DataFrame(months_raw)
    status_df = pd.DataFrame(status_raw)
    death_df = pd.DataFrame(death_raw)
    followup_df = pd.DataFrame(followup_raw)

    # Check the first 5 rows of each DataFrame
    print("\n🔍 OS_MONTHS Raw Data:")
    print(months_df.head())

    print("\n🔍 OS_STATUS Raw Data:")
    print(status_df.head())

    print("\n🔍 DAYS_TO_DEATH Raw Data:")
    print(death_df.head())

    print("\n🔍 DAYS_TO_LAST_FOLLOWUP Raw Data:")
    print(followup_df.head())

    # Check unique values to see if they contain non-numeric junk
    print("\n🔍 Unique OS_STATUS Values:", status_df["value"].unique())
    print("\n🔍 Unique DAYS_TO_DEATH Values:", death_df["value"].unique())
    print("\n🔍 Unique DAYS_TO_LAST_FOLLOWUP Values:", followup_df["value"].unique())

else:
    print("⚠️ One or more survival attributes failed to fetch!")

print("\n🔍 Columns before merging survival data:")
print(clinical_df.columns)

clinical_df = clinical_df.merge(
    survival_df[["uniquePatientKey", "survival_time", "event"]],
    left_on="uniquePatientKey_clinical",
    right_on="uniquePatientKey",
    how="left"
)

print("\n🔍 Columns after merging survival data:")
print(clinical_df.columns)
