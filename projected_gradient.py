import networkx as nx
import numpy as np
import tqdm
import time
from utils import not_infected_matrix, post_processing, calculate_F1


def create_W(A):

    W = -A #非对角线都是负的A

    for i in range(A.shape[0]):
        W[i,i] = np.sum(A[:,i])

    return W

def compute_grad(S, A, not_infected):

    grad_A = np.zeros_like(A)

    for i in range(S.shape[0]):

        s_i = S[i]
        s_i_1 = np.where(s_i == 1)[0]
        sub_A = A[np.ix_(s_i_1, s_i_1)]
        W = create_W(sub_A)

        W[0,:] = 1.0 #augmented

        try:
            inv_arg_W = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            print("Error")

        inv_arg_W_T = inv_arg_W.T

        J = -inv_arg_W_T

        for j_i in range(len(s_i_1)):

            J[:,j_i] += inv_arg_W[j_i,j_i]

        grad_A[np.ix_(s_i_1,s_i_1)] += J

    grad_A -= not_infected * np.nan_to_num(np.divide(1.0, 1.0 - A))

    return -grad_A


def inference(S, A, step_size= 1e-3, iterations = 500, prune_network = None):

    n = A.shape[0]
    pred_A = np.ones_like(A) * 0.2
    if prune_network is not None:
        prune_network[prune_network==0] = 1e-5
        pred_A = pred_A * prune_network

    G = nx.from_numpy_matrix(A)

    st = time.time()
    NI_matrix = not_infected_matrix(S)
    for _ in tqdm.tqdm(range(iterations)):

        grad = compute_grad(S, pred_A, NI_matrix)
        pred_A = pred_A - step_size * grad
        pred_A[pred_A >= 1] = 1.0 - 1e-5 #projected
        pred_A[pred_A <= 0] = 1e-5

    end_time = time.time() - st

    pred_A = pred_A * prune_network
    pred_A[pred_A <= 1e-5] = 0.0
    best_t, IG = post_processing(pred_A)
    print("RUN TIME : ", end_time, "BEP point : ", best_t, "P , R, F1 : ", calculate_F1(IG, G))





