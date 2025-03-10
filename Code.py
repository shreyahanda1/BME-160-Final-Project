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
SAMPLE_LIST_ID = "gbm_tcga_all"  # Update this if needed

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