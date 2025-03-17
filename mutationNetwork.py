import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

class MutationNetwork:
    def __init__(self):
        pass
        
    def create_plot(self, mutation_df, all_mutation_matrix):
        try:
            # just focus on top 30 genes to keep things manageable
            top_mutated_genes = mutation_df["gene_symbol"].value_counts().head(30).index.tolist()
            
            # make a matrix showing which genes tend to be mutated together
            co_occurrence_matrix = pd.DataFrame(0, index=top_mutated_genes, columns=top_mutated_genes)
            for i, gene1 in enumerate(top_mutated_genes):
                for gene2 in top_mutated_genes[i:]:
                    co_occurrence = (all_mutation_matrix[gene1] & all_mutation_matrix[gene2]).sum()
                    co_occurrence_matrix.loc[gene1, gene2] = co_occurrence
                    co_occurrence_matrix.loc[gene2, gene1] = co_occurrence  # make it symmetric

            # create a network graph to visualize co-occurring mutations
            G = nx.Graph()
            for gene in top_mutated_genes:
                mutation_count = mutation_df["gene_symbol"].value_counts().get(gene, 0)
                G.add_node(gene, size=mutation_count)

            # only add edges if genes co-occur at least twice
            threshold = 2
            for i, gene1 in enumerate(top_mutated_genes):
                for j, gene2 in enumerate(top_mutated_genes[i+1:], i+1):
                    weight = co_occurrence_matrix.loc[gene1, gene2]
                    if weight >= threshold:
                        G.add_edge(gene1, gene2, weight=weight)

            # network visualization stuff
            pos = nx.spring_layout(G, k=0.3, seed=42)  # positions the nodes
            node_sizes = [G.nodes[gene]['size'] * 20 for gene in G.nodes()]  # bigger nodes = more mutations
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]  # thicker edges = more co-occurrence

            plt.figure(figsize=(10, 8))
            fig = plt.gcf()  # get current figure
            fig.set_facecolor('lavender')
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, node_color="skyblue", edgecolors="black")
            nx.draw_networkx_edges(G, pos, width=[w/2 for w in edge_weights], alpha=0.5, edge_color="green")
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            plt.title("Network of Co-occurring Mutations", fontsize=16, fontweight='bold')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("output/mutation_network.png", dpi=300)
            print("Figure 2: mutation network plot")
            
            return co_occurrence_matrix
            
        except Exception as e:
            print(f"Error in network plot: {e}")
            return None