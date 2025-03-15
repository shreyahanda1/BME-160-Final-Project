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

plt.show()

'''
Kaplan-Meier Algorithm:
starts patients with 100% survival rate. Every time an event occurs and a patient dies, the survival
rate decreases.

Ex: Are some mutations making metastatic cases even deadlier?
If metastatic cases has a average 100 day survival rate but TP53 has a average of 50 day survival, 
this is a huge impact.
Why? Tells how quick doctors should take action-- which tests/therapies to implement
Maybe they need to implement more aggressive treatment for high-risk patients
If we find a new gene that makes metastasis worse, pharmaceutical companies can develop new drugs that
target that gene.

1. Run Kaplan-Meier Algorithm within metastatic cases
2. If survival gap is huge, this might be evidence that TP53 worsens metastisis.  


# Fetch overall survival data
os_months_data = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_MONTHS")
os_status_data = fetch_data(f"studies/{STUDY_ID}/clinical-data?clinicalAttributeId=OS_STATUS")

# Convert to DataFrame
if os_months_data and os_status_data:
    os_months_df = pd.DataFrame(os_months_data)
    os_status_df = pd.DataFrame(os_status_data)

    # Ensure only numeric values are in survival time
    os_months_df = os_months_df[pd.to_numeric(os_months_df["value"], errors="coerce").notna()]
    os_months_df["value"] = os_months_df["value"].astype(float)  # Convert to float

    # Merge survival time and event status
    survival_df = os_months_df.merge(os_status_df, on="uniquePatientKey", suffixes=("_months", "_status"))

    # Remove duplicate patient entries (keep the first occurrence)
    survival_df = survival_df.drop_duplicates(subset=["uniquePatientKey"], keep="first")

    # Convert survival time (months to days)
    survival_df["survival_time"] = survival_df["value_months"] * 30
    
    # Convert event status (1 = deceased, 0 = alive)
    survival_df["event"] = survival_df["value_status"].apply(lambda x: 1 if x.upper() == "DECEASED" else 0)
'''