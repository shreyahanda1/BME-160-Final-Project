import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

class Cluster:
    ''' 
    Creates hierarchical heatmap clustering of gene mutations and their co-occurence with other genes.
    Identifies gene clustering patterns and gene mutations that oftentimes occur together. 
    '''
    def __init__(self):
        pass
        
    def create_plot(self, co_occurrence_matrix):
        '''
        Generates clustering heatmap for co-occurences. 
        Params: Matrix with gene mutation co-occurences
        Returns: Boolean. True if plot is created and no errors. False if errors. 
        ''' 
        try:
            # groups genes together based on similar mutation patterns
            row = hierarchy.linkage(co_occurrence_matrix, method='average')

            # plots figure and sets its background color
            figure = plt.gcf()
            figure.set_facecolor('lavender')
            
            # creates a cluster heatmap 
            # sets appearance for heatmap, including size, color, x and y labels, etc. 
            cluster_map = sns.clustermap(
                co_occurrence_matrix,
                figsize=(12, 10),
                cmap="YlOrRd",
                row_linkage=row,
                col_linkage=row,  
                xticklabels=True,
                yticklabels=True,
                linewidths=0.5
            )
            
            # creates plot title
            cluster_map.ax_heatmap.set_title("Hierarchical Clustering of Mutation Co-occurrence", fontsize=16, fontweight='bold')
            
            # saves the figure in output directory-- connects to the data.py file
            # returns True if success
            plt.savefig("output/mutation_clustering.png", dpi=300)
            print("Figure 3: Hierarchical clustering of mutation co-occurrence")
            return True
        # if anything fails, return False and error statement. debugging purposes
        except Exception as e:
            print(f"Error in clustering plot: {e}")
            return False