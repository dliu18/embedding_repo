import pickle
import json
import networkx as nx
import pickle 
import numpy as np
import subprocess
import sys
import traceback

try:
	from gem.evaluation import visualize_embedding as viz
	from gem.embedding.hope import HOPE
	#from gem.embedding.sdne	    import SDNE
	from sklearn.manifold import SpectralEmbedding
	from sklearn.decomposition import TruncatedSVD, PCA
except:
	pass

try:
	from ge import SDNE
except:
	pass

#MACROS
MAX_TRIES = 3
random_state = 42

###########################################################################
#LAMBDAS:
#Input: nx.Graph object corresponding to core and embedding dimension d
#Output: np.ndarray embedding matrix where rows are in the nodelist order specified by cores.nodes()

def hope_lambda(graphName, nx_graph, d):
	return HOPE(d=d, beta=0.01).learn_embedding(nx_graph)[0]

def lap_lambda(graphName, nx_graph, d):
	# return LaplacianEigenmaps(d=d).learn_embedding(graph)[0]
	return SpectralEmbedding(n_components=d,
							 affinity="precomputed",
							 n_jobs=-1,
							 random_state=random_state)\
			.fit_transform(nx.linalg.graphmatrix.adjacency_matrix(nx_graph))

def sdne_lambda(graphName, nx_graph, d):
	model = SDNE(nx_graph, hidden_size=[256, d], alpha=0.1, beta=10)
	model.train(epochs=200)
	emb_dict = model.get_embeddings()
	return np.array([emb_dict[node_name] for node_name in nx_graph.nodes()])

def svd_lambda(graphName, nx_graph, d):
	return TruncatedSVD(n_components=d, random_state=random_state)\
			.fit_transform(nx.linalg.graphmatrix.adjacency_matrix(nx_graph))

def read_embedding():
	with open("temp.emb", "r") as embeddingFile:
		embeddings = {}
		for line in embeddingFile:
			line_entries = line.split(" ")
			if len(line_entries) == 2:
				continue #skip header
			embeddings[line_entries[0]] =\
				np.array([float(coord) for coord in line_entries[1:]])
	return embeddings

def n2v_lambda(graphName, nx_graph, d): 
	nx.write_edgelist(nx_graph, "temp.edgelist")
	subprocess.run(["python",\
			"node2vec/src/main.py",\
			"--input", "temp.edgelist",\
			"--output", "temp.emb",\
			"--dimension", str(d),\
			"--workers", "24"])
	embeddings = read_embedding()
	return np.array([embeddings[node] for node in nx_graph.nodes()])

def hgcn_lambda(graphName, nx_graph, d):
	subprocess.run(["rm", "-rf", "hgcn/data/{}/{}.edgelist".format(graphName, graphName)])
	subprocess.run(["rm", "-rf", "hgcn/logs/{}/embeddings.npy".format(graphName)])
	
	nx.write_edgelist(nx_graph, "hgcn/data/{}/{}.edgelist".format(graphName, graphName))
	result = subprocess.run(["python",
				   "hgcn/train.py",
				   "--task", "lp",
					"--dataset", graphName,
					"--save", "1",
					"--save-dir", "hgcn/logs/" + graphName,
					"--model", "Shallow",
					"--manifold", "PoincareBall",
					"--lr", "0.01",
					"--weight-decay", "0.0005",
					"--dim", str(d),
					"--num-layers", "0",
					"--use-feats", "0",
					"--dropout", "0.2",
					"--act", "None",
					"--bias", "0",
					"--optimizer", "RiemannianAdam",
					"--cuda", "0",
					"--log-freq", "1000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	print(result.stdout)
	return np.load("hgcn/logs/{}/embeddings.npy".format(graphName))

def get_embedding_algorithms(algNames):
	algLambdas = []
	if "SDNE" in algNames:
		algLambdas.append({"name": "SDNE", "method": sdne_lambda})
	if "SVD" in algNames:
		algLambdas.append({"name": "SVD", "method": svd_lambda})
	if "HOPE" in algNames:
		algLambdas.append({"name": "HOPE", "method": hope_lambda})
	if "LaplacianEigenmap" in algNames:
		algLambdas.append({"name": "LaplacianEigenmap", "method": lap_lambda})
	if "Node2Vec" in algNames:
		algLambdas.append({"name": "Node2Vec", "method": n2v_lambda})
	if "HGCN" in algNames:
		algLambdas.append({"name": "HGCN", "method": hgcn_lambda})
	return algLambdas

###########################################################################
#HELPERS

def embed_graph(graphName, nx_graph, d, algDict): 
	outputFileName = "embeddings/{}/{}/{}_{}_{}_embedding.npy".format(
		graphName,
		algDict["name"],
		graphName,
		algDict["name"],
		d)
	try:
		#check if embeddings already exist
		embeddings = np.load(outputFileName)
		assert(embeddings.shape == (len(nx_graph), d))
		return
	except:
		pass

	attempt = 1
	method = algDict["method"]
	embeddingRC = False
	embeddings = np.array([])
	print("Embedding {} {} {}".format(graphName, algDict["name"], d))
	while attempt <= MAX_TRIES:
		try: 
			embeddings = method(graphName, nx_graph, d)
			embeddingRC = True
			break;
		except:
			print("Failed")
			print(sys.exc_info()[1])
			traceback.print_tb(sys.exc_info()[2])
			attempt += 1
			continue

	if (embeddingRC):
		assert(embeddings.shape == (len(nx_graph), d))
		np.save(outputFileName, embeddings)

def main(graphMetaData, algNames, d_s):
	for graphMetaDatum in graphMetaData:
		graphName = graphMetaDatum["name"]
		edgelistFilename = graphMetaDatum["edgelistFilename"]

		nx_graph = nx.read_edgelist(edgelistFilename)
		for d in d_s:
			embedding_algorithms = get_embedding_algorithms(algNames)
			for algDict in embedding_algorithms:
				embed_graph(graphName, nx_graph, d, algDict)

###########################################################################
#MAIN

if __name__ == "__main__":
	#Read Config File
	assert len(sys.argv) > 1
	print(sys.argv)
	with open(sys.argv[1], "r") as configFile:
		config = json.load(configFile)
	main(**config)