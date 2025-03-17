# BME 160
# Cancer Genomics: Gene Expression Analysis 
# Group Members: Architha Dhavala (adhavala), Shreya Handa (shanda1), Arya Ashok (abashok)
import os
import warnings
import pandas as pd
import requests

# suppress warnings to avoid nonsense and for clearer output
warnings.filterwarnings('ignore')

# makes sure output directory exists to save and store results
os.makedirs("output", exist_ok=True)

# API for cBioPortal - we're using glioblastoma tcga data
BASE_URL = "https://www.cbioportal.org/api"
STUDY_ID = "gbm_tcga"

class DataFetcher:
    ''' 
    Sets API points and creates variables to store mutation data for later use. 
    '''
    def __init__(self):
        # initialize sample list data
        self.mutation_profile_id = f"{STUDY_ID}_mutations"
        self.sample_list_id = f"{STUDY_ID}_sequenced"
        
        # initialize data containers to store sample ID for primary and metastatic tumors
        # initialize DataFrames to store data and results
        self.primary_tumor = []
        self.metastatic_tumor = []
        self.mutation_df = None
        self.primary_tumor_df = None
        self.metastatic_tumor_df = None
        self.mutation_compare_df = None
        self.top_primary = None
        self.top_metastatic = None
        self.top_unique = None
        
    def fetch_data(self, endpoint, params=None):
        """
        Fetches data from cBioPortal API
        
        Params:
            endpoint: the API endpoint to get data from
            params: parameters to include in the request
            
        Returns:
            JSON data from the API or none if it fails
        """
        try:
            url = f"{BASE_URL}/{endpoint}"
            response = requests.get(url, params=params, headers={"Accept": "application/json"})
            # raise exception if something goes wrong
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return None
            
    def extract_gene_symbol(self, df):
        """
        Extracts gene symbols from nested JSON structure in mutation data
        
        Params:
            df: DataFrame with "gene" column
            
        Modifies df in place to add gene_symbol column with extracted gene names
        """
        # noticed some data is stored in a dictionary
        # if the gene data is in dictionary, specifically extract "hugoGeneSymbol" 
        # if not in dictioanry, simply extract the gene
        try:
            df["gene_symbol"] = df["gene"].apply(
                lambda x: x["hugoGeneSymbol"] if isinstance(x, dict) and "hugoGeneSymbol" in x else str(x)
            )
        except Exception as e:
            print(f"Error extracting gene symbols: {e}")
            df["gene_symbol"] = df.get("hugoGeneSymbol", "Unknown")
            
    def create_mutation_matrix(self, df, sample_list):
        """
        Creates a binary matrix showing which samples have which mutations
        
        Params:
            df: dataframe with mutation data
            sample_list: list of sample ids to include
            
        Returns:
            DataFrame with samples as rows and genes as columns, 1 means mutated
        """
        unique_genes = df["gene_symbol"].unique()
        mutation_matrix = pd.DataFrame(0, index=sample_list, columns=unique_genes)
        for _, row in df.iterrows():
            sample_id = row["sampleId"]
            gene = row["gene_symbol"]
            if sample_id in sample_list:
                mutation_matrix.loc[sample_id, gene] = 1
        return mutation_matrix
    
    def load_data(self):
        # fetch clinical data
        # get the sample types (primary vs recurrent tumors)
        sample_types_raw = self.fetch_data(f"studies/{STUDY_ID}/clinical-data", params={'clinicalAttributeIds': ['SAMPLE_TYPE']})
        # if there is clinical data found, converts to DataFrame
        if sample_types_raw:
            sample_types_df = pd.DataFrame(sample_types_raw)
            print("\033[1m" + '**MUTATION PATTERNS & GENE EXPRESSION ALGORITHM USING GLIOBLASTOMA TGCA DATA**')  # bold text looks cooler
            print(f"Sample types data loaded: {sample_types_df.shape[0]} samples")
            
            # filtering primary tumor and recurring (metastatic) tumor data
            # extract primary tumor sample IDs
            # convert to list
            self.primary_tumor = sample_types_df[sample_types_df["value"] == "Primary"]["sampleId"].tolist()

            # extract recurring (metastatic) tumor sample IDs
            # convert to list
            self.metastatic_tumor = sample_types_df[sample_types_df["value"] == "Recurrence"]["sampleId"].tolist()
            
            print(f"Primary tumors found: {len(self.primary_tumor)}")
            print(f"Metastatic tumors found: {len(self.metastatic_tumor)}")
        # if no data, print error and reset primary_tumor and metastatic_tumor
        else:
            print("Failed to load sample types")
            self.primary_tumor = []
            self.metastatic_tumor = []
        
        # get all the mutation data
        mutation_response = self.fetch_data(
            f"molecular-profiles/{self.mutation_profile_id}/mutations",
            params={"sampleListId": self.sample_list_id, "projection": "DETAILED"}
        )

        # filter mutation data to primary_tumor_df or metastatic_tumor_df based on the patient sampleId 
        if mutation_response:
            # convert to dataframe and do some basic filtering
            self.mutation_df = pd.DataFrame(mutation_response)
            print(f"Mutation data loaded: {self.mutation_df.shape[0]} records")
            self.primary_tumor_df = self.mutation_df[self.mutation_df["sampleId"].isin(self.primary_tumor)]
            self.metastatic_tumor_df = self.mutation_df[self.mutation_df["sampleId"].isin(self.metastatic_tumor)]
            # print out new primary_tumor_df and metastatic_tumor_df to ensure data reflects code's intention and no errors occured
            print(f"Primary tumor mutations: {self.primary_tumor_df.shape[0]}")
            print(f"Metastatic tumor mutations: {self.metastatic_tumor_df.shape[0]}\n")

            # count different gene mutation type distribtution
            mutation_type_counts = self.mutation_df["mutationType"].value_counts()
            print("-----------Mutation Type Distribution:-----------")
            print(mutation_type_counts, '\n')

            # extract gene symbols for all our dataframes
            # noticed some data is stored in a dictionary
            # if the gene data is in dictionary, specifically extract "hugoGeneSymbol" 
            # if not in dictioanry, simply extract the gene
            self.extract_gene_symbol(self.mutation_df)
            self.extract_gene_symbol(self.primary_tumor_df)
            self.extract_gene_symbol(self.metastatic_tumor_df)

            # value_counts() is used in Pandas to obtain a series containing counts of unique values/genes
            # counts the number of mutations per gene for both primary and metastatic tumors
            primary_mutation_counts = self.primary_tumor_df["gene_symbol"].value_counts()
            metastatic_mutation_counts = self.metastatic_tumor_df["gene_symbol"].value_counts()
            
            # create DataFrame that compares gene mutation counts between primary and metastatic tumors
            # fill NaN values with 0
            self.mutation_compare_df = pd.DataFrame({
                "Primary Tumors": primary_mutation_counts, 
                "Metastatic Tumors": metastatic_mutation_counts
            }).fillna(0)

            # find genes that are different between primary and metastatic
            primary_vs_metastatic_comparison = self.mutation_compare_df['Primary Tumors'] > self.mutation_compare_df['Metastatic Tumors']
            primary_mutations = self.mutation_compare_df[primary_vs_metastatic_comparison]
            print("Genes more frequently mutated in primary tumors:")
            print(primary_mutations)
            
            # genes more common in metastatic tumors
            metastatic_mutations = self.mutation_compare_df[~primary_vs_metastatic_comparison]
            print("Genes more frequently mutated in metastatic tumors:")
            print(metastatic_mutations)

            # there are many genes with mutations expressed in patients, but that is a lot of data
            # we want to extract the most important/vital data so extract the top 30 genes with the most mutations
            # this provides critical information
            self.top_primary = self.mutation_compare_df.sort_values("Primary Tumors", ascending=False).head(30)
            self.top_metastatic = self.mutation_compare_df.sort_values("Metastatic Tumors", ascending=False).head(30)
            # combining the two DatFrames, top_primary and top_metastatic, together
            # to put in one graph together
            self.top_unique = pd.concat([self.top_primary, self.top_metastatic]).index.unique()
            print('\n\n-----------Top 30 Primary Tumors:-----------\n', self.top_primary)
            print('\n\n-----------Top 30 Metastatic Tumors:-----------\n', self.top_metastatic, '\n')
            
            return True
        # if errors occured, print statement. debugging
        else:
            print("Failed to load mutation data")
            return False