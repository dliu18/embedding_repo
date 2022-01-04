import networkx as nx
import scipy.io as sio
import sys

if __name__ == "__main__":
	assert(len(sys.argv) > 2)
	inputpath  = sys.argv[1]
	outputpath = sys.argv[2]

	matfile = sio.loadmat(inputpath)
	nx_graph = nx.from_scipy_sparse_matrix(matfile["network"], create_using=nx.Graph())
	nx.readwrite.edgelist.write_edgelist(nx_graph, outputpath, data=False)
