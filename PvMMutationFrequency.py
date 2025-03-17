import matplotlib.pyplot as plt
import pandas as pd

class PvMFrequency:
    ''' 
    Generates primary vs metastatic gene mutation frequency plot.
    '''
    def __init__(self):
        pass
        
    def create_plot(self, mutation_compare_df, top_unique):
        try:
            # keep only the top selected genes. this is what we will be plotting
            filtered_mutation_compare_df = mutation_compare_df.loc[top_unique]
            # find the genes with highest mutation count in each category
            max_primary_gene = filtered_mutation_compare_df['Primary Tumors'].idxmax()
            max_metastatic_gene = filtered_mutation_compare_df['Metastatic Tumors'].idxmax()

            # create and set color of the plot 
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.set_facecolor('lavender')

            # plot each column separately so we can control colors for individual bars
            bar_width = 0.35
            x = range(len(filtered_mutation_compare_df.index))

            # plot the figure
            # set color red for primary data
            # set color green for metastatic data   

            # plot primary tumors (red bars)  
            primary_color = ['red' if gene == max_primary_gene else 'red' for gene in filtered_mutation_compare_df.index]
            primary_bar = ax.bar([i - bar_width/2 for i in range(len(filtered_mutation_compare_df.index))], 
                                filtered_mutation_compare_df['Primary Tumors'], 
                                width=bar_width, 
                                color=primary_color, 
                                alpha=0.7, 
                                label='Primary Tumors')
            
            # plot metastatic tumors (green bars)
            metastatic_color = ['green' if gene == max_metastatic_gene else 'green' for gene in filtered_mutation_compare_df.index]
            metastatic_bar = ax.bar([i + bar_width/2 for i in range(len(filtered_mutation_compare_df.index))], 
                                    filtered_mutation_compare_df['Metastatic Tumors'], 
                                    width=bar_width, 
                                    color=metastatic_color, 
                                    alpha=0.7, 
                                    label='Metastatic Tumors')
            
            # all code below customizes the plot's appearance, including x- and y-axis title, font size, etc. 

            # adds annotations for the highest values
            max_primary = filtered_mutation_compare_df.index.get_loc(max_primary_gene)
            max_metastatic = filtered_mutation_compare_df.index.get_loc(max_metastatic_gene)
            
            # annotates highest primary
            primary_val = filtered_mutation_compare_df.loc[max_primary_gene, 'Primary Tumors']
            ax.annotate(f"Highest\n{int(primary_val)}", 
                        xy=(max_primary - bar_width/2, primary_val), 
                        xytext=(max_primary - bar_width/2, primary_val + 5),
                        ha='center', 
                        color='darkred', 
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkred'))

            # annotates highest metastatic
            metastatic_val = filtered_mutation_compare_df.loc[max_metastatic_gene, 'Metastatic Tumors']
            ax.annotate(f"Highest\n{int(metastatic_val)}", 
                        xy=(max_metastatic + bar_width/2, metastatic_val), 
                        xytext=(max_metastatic + bar_width/2, metastatic_val + 5),
                        ha='center', 
                        color='darkgreen', 
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkgreen'))
            
            # sets labels and title
            ax.set_xlabel("Gene", fontsize=12)
            ax.set_ylabel("Mutation Count", fontsize=12)
            ax.set_title("Mutation Frequency in Primary vs. Metastatic Tumors", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(filtered_mutation_compare_df.index, rotation=50, fontsize=9, ha='right')
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig("output/mutation_frequency.png")
            print("Figure 1: mutation frequency plot")
            return True
        # if anything in the above fails, print out error message. debugging purposes
        except Exception as e:
            print(f"Error creating mutation frequency plot: {e}")
            return False