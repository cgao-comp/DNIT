import networkx as nx
from IC_model import IC
from inverse_sigmod import run_torch_version
from MCEM import inference as em_inference
import numpy as np
from utils import calculate_MI, modified_kmeans
import warnings

warnings.filterwarnings('ignore')

def generate_infections(A, num_sim = 100):

    N = A.shape[0]
    S = np.zeros([num_sim, N])
    nx_graph = nx.from_numpy_array(A)
    trees = []
    while len(trees) < num_sim:
        seed = np.random.choice(np.arange(0, N), size=1)
        cascade, tree = IC(Networkx_Graph=nx_graph, Seed_Set=seed, Probability=A)
        if len(tree.nodes) >= 3:
            S[len(trees), cascade] = 1
            trees.append(tree)
    average_paths = 0
    for tree in trees:
        average_paths += len(tree.nodes())

    print("average length of infections: ", average_paths / len(trees))
    return S

if __name__ == '__main__':

    np.random.seed(2023)
    n = 200
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community = 10, seed = 2023)
    print(len(G.nodes),len(G.edges))

    A = nx.to_numpy_array(G)
    P = np.random.uniform(0.1,0.3,[n,n])
    A = A * P
    S = generate_infections(A, num_sim=100)

    mi_matrix, p_matrix = calculate_MI(S.T)
    cluster, fixed_cluster = modified_kmeans(mi_matrix)
    threshold = max(fixed_cluster.values())
    prune_network = np.zeros([n, n])
    prune_network[mi_matrix > threshold] = 1.0
    prune_network[mi_matrix <= threshold] = 0.0

    #-------------------MCEM---------------------------
    em_inference(S, A, sample_size = 10, prune_network = prune_network, iterations = 400)

    # -------------------inverse sigmod--------------------
    run_torch_version(A, S, iterations=1000, prune_network=prune_network)






