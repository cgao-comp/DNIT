import networkx as nx
import numpy as np

def not_infected_matrix(S):

    num_sim, num_node = S.shape

    not_infected = np.zeros([num_node, num_node])

    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                continue

            s_i_j = S[:,i] - S[:,j]

            not_infected[i,j] = len(np.where(s_i_j == 1)[0])

    return not_infected

def calculate_MI(x):
    '''
    x: [n,beta]
    '''
    [n,m] = x.shape

    mi_matrix = np.zeros([n,n])
    p = np.zeros([n,n])

    for i in range(n):

        v_i = x[i, :]
        v_i_1 = np.where(v_i == 1)[0]
        v_i_0 =  np.where(v_i == 0)[0]
        p_i1 = len(v_i_1) / m * 1.0
        p_i0 = len(v_i_0) / m * 1.0

        for j in range(i+1,n):
            v_j = x[j,:]
            v_j_1 = np.where(v_j == 1)[0]
            v_j_0 = np.where(v_j == 0)[0]

            p_j1 = len(v_j_1) / m * 1.0
            p_j0 = len(v_j_0) / m * 1.0

            p_i1_j1 = len(np.intersect1d(v_j_1, v_i_1)) / m * 1.0
            p_i1_j0 = len(np.intersect1d(v_j_0, v_i_1)) / m * 1.0
            p_i0_j1 = len(np.intersect1d(v_j_1, v_i_0)) / m * 1.0
            p_i0_j0 = len(np.intersect1d(v_j_0, v_i_0)) / m * 1.0

            if p_i1 > 0:
                p[i,j] = p_i1_j1 / p_i1

            if p_j1 > 0:
                p[j,i] = p_i1_j1 / p_j1

            if  p_i1_j1 > 0:
                mi_matrix[i,j] += p_i1_j1 * np.log(p_i1_j1 / (p_i1 * p_j1))
                mi_matrix[j,i] += p_i1_j1 * np.log(p_i1_j1 / (p_i1 * p_j1))

            if  p_i1_j0 > 0:
                mi_matrix[i, j] += p_i1_j0 * np.log(p_i1_j0 / (p_i1 * p_j0))
                mi_matrix[j, i] += p_i1_j0 * np.log(p_i1_j0 / (p_i1 * p_j0))

            if  p_i0_j1 > 0:
                mi_matrix[i, j] += p_i0_j1 * np.log(p_i0_j1 / (p_i0 * p_j1))
                mi_matrix[j, i] += p_i0_j1 * np.log(p_i0_j1 / (p_i0 * p_j1))

            if  p_i0_j0 > 0:
                mi_matrix[i, j] += p_i0_j0 * np.log(p_i0_j0 / (p_i0 * p_j0))
                mi_matrix[j, i] += p_i0_j0 * np.log(p_i0_j0 / (p_i0 * p_j0))

    return mi_matrix, p


def calculate_F1(IG,RG):

    ig_edges = IG.edges
    rg_edges = RG.edges

    TP = 0.0
    FP = 0.0
    FN = 0.0

    for (i,j) in ig_edges:
        if (i,j) in rg_edges or (j,i) in rg_edges:
            TP += 1.0
        else:
            FP += 1.0

    for (i,j) in rg_edges:
        if (i,j) not in ig_edges and (j,i) not in ig_edges:
            FN += 1.0

    P = TP / (TP+FP)
    R = TP / (TP+FN)

    return round(P,3),round(R,3),round(2*P*R / (P+R),3)


def modified_kmeans(mi_matrix):

    fixed_centroid = 0.0
    centroid = np.max(mi_matrix)

    fixed_cluster = {}
    cluster = {}

    [n,_] = mi_matrix.shape

    for i in range(n):
        for j in range(i+1,n):
            if mi_matrix[i,j] <= 0:
                continue

            if (mi_matrix[i,j]) - fixed_centroid <= (mi_matrix[i,j] - centroid):
                fixed_cluster[(i,j)] = mi_matrix[i,j]
            else:
                cluster[(i, j)] = mi_matrix[i, j]


    is_stable = False

    while is_stable == False:

        centroid = np.mean([cluster[key] for key in cluster.keys()])
        modified_count = 0

        for i in range(n):
            for j in range(i + 1, n):

                if (mi_matrix[i, j] - fixed_centroid) <= abs(mi_matrix[i, j] - centroid):
                    if (i,j) not in fixed_cluster.keys():
                        modified_count += 1
                        fixed_cluster[(i, j)] = mi_matrix[i, j]

                        if (i,j) in cluster.keys():
                            cluster.pop((i, j))
                else:
                    if (i, j) not in cluster.keys():
                        modified_count += 1
                        cluster[(i, j)] = mi_matrix[i, j]

                        if (i, j) in fixed_cluster.keys():
                            fixed_cluster.pop((i, j))

                    cluster[(i, j)] = mi_matrix[i, j]

        if modified_count > 0:
            is_stable = False

        else:
            is_stable = True

    return cluster, fixed_cluster

def post_processing(estimated_A):
    #predicted BEP
    '''
    Ensemble method
    '''

    thresholds = np.linspace(start = 1e-5, stop=5e-1, num=10000)

    FP_FN_diff = np.zeros([len(thresholds)])

    for i, t in enumerate(thresholds):

        predicted_FN = np.sum(estimated_A[estimated_A < t])
        predicted_FP = np.sum(1.0 - estimated_A[estimated_A >= t])

        FP_FN_diff[i] = np.abs(predicted_FP - predicted_FN)

    best_t = thresholds[np.argmin(FP_FN_diff)]
    IG = np.zeros_like(estimated_A)
    IG[estimated_A >= best_t] = 1
    IG[estimated_A < best_t] = 0

    IG = nx.from_numpy_matrix(IG)

    return best_t, IG