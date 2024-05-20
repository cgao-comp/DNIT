import time

import networkx as nx
import numpy as np
import tqdm
from utils import not_infected_matrix, post_processing, calculate_F1


class WilsonSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.W += 1e-2 #avoid stuck
        self.n = len(W)
        #self.W[:, 0] = 0
        np.fill_diagonal(self.W, 0.0)
        for i in range(self.n):
            self.W[:, i] /= np.sum(self.W[:, i]) #normalized

    def _sample(self):
        inTree = np.zeros(self.n).astype(bool)
        next = - np.ones(self.n).astype(int)
        root = np.random.choice(np.arange(self.n))
        inTree[root] = True
        for i in range(self.n):
            if i == root:
                continue
            u = i
            while not inTree[u]:
                next[u] = np.random.choice(np.arange(self.n), p=self.W[:, u])
                u = next[u]
            u = i
            while not inTree[u]:
                inTree[u] = True
                u = next[u]
        return next

    def sample(self):
        while True:
            yield self._sample()

def E_step(S, A, sample_size = 100):

    N = A.shape[0]

    infected_matrix = np.zeros([N,N])

    for s_i in range(S.shape[0]):

        s_i_1 = np.where(S[s_i] == 1)[0]

        sub_G = A[np.ix_(s_i_1, s_i_1)]

        np.fill_diagonal(sub_G, 0)
        tree_sampler = WilsonSample(sub_G)

        for i, sample in enumerate(tree_sampler.sample()):

            for node, parent in enumerate(sample):
                if parent == -1:
                    continue
                infected_matrix[s_i_1[parent], s_i_1[node]] += 1

            if i == sample_size:
                break

        del tree_sampler

    return infected_matrix

def M_step(infected_matrix, uninfected_matrix):

    A_esimate = infected_matrix / (uninfected_matrix + infected_matrix + 1e-10)

    return A_esimate

def inference(S, A, sample_size = 10, prune_network = None, iterations = 10):

    G = nx.from_numpy_matrix(A)
    pred_A = np.ones_like(A) * 0.2

    if prune_network is not None:
        prune_network[prune_network == 0] = 1e-5
        pred_A = pred_A * prune_network

    np.fill_diagonal(pred_A, 0)

    st = time.time()
    uninfected_matrix = not_infected_matrix(S)

    for _ in tqdm.tqdm(range(iterations)):

        infected_matrix = E_step(S, pred_A, sample_size=sample_size)
        pred_A = M_step(infected_matrix, sample_size * uninfected_matrix)

    end_time = time.time() - st
    pred_A = pred_A * prune_network
    pred_A[pred_A <= 1e-5] = 0.0
    best_t, IG = post_processing(pred_A)
    print("RUN TIME : ", end_time, "BEP point : ", best_t, "P , R,  F1 : ", calculate_F1(IG, G))