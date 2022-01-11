import numpy as np
import networkx as nx
import sys 

import hypercomparison.networks
import hypercomparison.correlation_and_greedy_routing

import json 

def generate_embedding_dict(embedding_npy_filename, edgelist_filename):
  embeddings_np = np.load(embedding_npy_filename)
  nx_graph = nx.read_edgelist(edgelist_filename)
  embeddings = {str(list(nx_graph.nodes())[i]): embeddings_np[i] for i in range(len(nx_graph))}
  return embeddings

if __name__ == "__main__":
  assert len(sys.argv) == 2
  configFilename = sys.argv[1]
  with open(configFilename, "r") as configFile:
    config = json.load(configFile)

  for embedding_alg in config["algNames"]:
    for graph_metadatum in config["graphMetaData"]:
      for d_str in config["d_s"]:
        embedding_npy_filename = "../embeddings/{}/{}/{}_{}_{}_embedding.npy".format(
          graph_metadatum["name"],
          embedding_alg,
          graph_metadatum["name"],
          embedding_alg,
          d_str)
        edgelist_filename = "../{}".format(graph_metadatum["edgelistFilename"])
        embeddings_dict = generate_embedding_dict(embedding_npy_filename, edgelist_filename)

        network = hypercomparison.networks.RealNetwork(graph_metadatum["name"])

        all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings_dict)
        pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
        print("Embedding Alg: {}\t Graph: {}\t Pearson Correlation: {}".format(
          embedding_alg,
          graph_metadatum["name"],
          pearson_correlation))