#BME 160
#Cancer Genomics: Gene Expression Analysis 
# Group Members: Architha Dhavala (adhavala), Shreya Handa (shanda1), Arya Ashok (abashok)
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import requests
from scipy.cluster import hierarchy
from matplotlib.colors import LinearSegmentedColormap

from data import DataFetcher
from PvMMutationFrequency import PvMFrequency
from mutationNetwork import MutationNetwork
from cluster import Cluster

'''
Input: Glioblastoma cancer data from cBioPortal.
Ouput: Graph that finds the most likely gene mutations in primary vs. metastatic tumors. 

Informs doctors about different gene mutations and there potential agressiveness. 
For example, say a patient just got his biopsy exams back and his GBM tumor has a BAG1 gene mutation. Looking at our table, we see that
BAG1 gene is highly expressed in metastatic GBM cancer. This will tell doctors that instact and aggressive treatment is necessary for the 
patient's treatment plan. 

BAG1 gene's affect is still being investigated and researched in GBM. There is not much data about it in relation to GBM. But, our graph shows
that there is signifcant evidence of BAG1 gene mutation and metastatic cancer. This adds more to current research attempts.

Why? Tells doctors how aggressive a patient's gene mutation is.
Tells how quick doctors should take action-- which tests/therapies to implement.
Tells doctors if they need to implement more aggressive treatment for high-risk patients.
'''

def main():
    # creates DataFetcher instace
    # fetches the mutation and clinical data from our class DataFetcher
    data_fetcher = DataFetcher()
    if not data_fetcher.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # generates the primary vs. metastatic gene mutation frequency plot
    freq = PvMFrequency()
    freq.create_plot(data_fetcher.mutation_compare_df, data_fetcher.top_unique)
    
    # generates mutation martix for the network plot analysis
    # conaints columns=genes and rows=samples
    # 1 indicates the presense of a mutation
    samples = list(set(data_fetcher.primary_tumor + data_fetcher.metastatic_tumor))
    mutation_matrix = data_fetcher.create_mutation_matrix(data_fetcher.mutation_df, samples)
    
    # generates mutation network plot
    # shows the relationship between genes that frequently mutate together
    network = MutationNetwork()
    co_occurrence = network.create_plot(data_fetcher.mutation_df, mutation_matrix)
    
    # if the found co-occurence matrix is valid, then generate the clustering plot
    if co_occurrence is not None:
        # Generate hierarchical clustering plot (Figure 3)
        cluster = Cluster()
        cluster.create_plot(co_occurrence)
    
    # display all of the plots
    plt.show()
    
    # confirm that all plots have been successfully created
    print("Analysis completed. All figures loaded.")

if __name__ == "__main__":
    main()