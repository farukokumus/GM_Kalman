import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=1000)
from ime_fgs.messages import GaussianMixtureMeanCovMessage

def reduction_algorithm(msg: GaussianMixtureMeanCovMessage, num_weights: int):

    weights = list(msg.weights)
    mean = list(msg.mean)
    cov = list(msg.cov)
    current_size = len(weights)

    while current_size > num_weights and np.sum(weights) > 0.995:
        index = np.argmin(weights)
        weights.pop(index), mean.pop(index), cov.pop(index)
        current_size = len(weights)

    dist_matrix = np.zeros((current_size, current_size))
    for i in range(current_size):
        for j in range(current_size):
            if i < j:
                gausi = (weights[i], mean[i], cov[i])
                gausj = (weights[j], mean[j], cov[j])
                dist_matrix[i,j] = distance_measure(gausi, gausj)

    while current_size > num_weights:
        index = np.argwhere(dist_matrix==dist_matrix.max())[0]

        gausj = weights.pop(index[1]), mean.pop(index[1]), cov.pop(index[1])
        gausi = weights.pop(index[0]), mean.pop(index[0]), cov.pop(index[0])

        wij, mij, pij = momentum_preserving_merge(gausi, gausj)
        weights += [wij]
        mean += [mij]
        cov += [pij]

        current_size = len(weights)
        new_dist_matrix = np.zeros((current_size, current_size))
        new_dist_matrix[:-1,:-1] = np.delete(np.delete(dist_matrix, index, 0), index, 1)
        for i in range(current_size):
            gausi = (weights[i], mean[i], cov[i])
            new_dist_matrix[i,-1] = distance_measure(gausi, (wij, mij, pij))

        dist_matrix = new_dist_matrix
    weights /= np.sum(weights)

    sorted_index = np.argsort(-weights, 0).reshape(-1)
    return GaussianMixtureMeanCovMessage(weights[sorted_index, :], np.array(mean)[sorted_index, :, :], np.array(cov)[sorted_index, :, :])

def distance_measure(gausi, gausj):
    wi, _, pi = gausi
    wj, _, pj = gausj
    wij, _, pij = momentum_preserving_merge(gausi, gausj)

    return 1/2 *(wij * np.log(np.linalg.det(pij)) - wi * np.log(np.linalg.det(pi)) - wj * np.log(np.linalg.det(pj)))

def momentum_preserving_merge(gausi, gausj):
    wi, mi, pi = gausi
    wj, mj, pj = gausj

    wij = wi + wj

    wi_ij = wi / wij
    wj_ij = wj / wij

    mij = wi_ij * mi + wj_ij * mj

    pij = wi_ij * pi + wj_ij * pj + wi_ij * wj_ij * (mi - mj) * (mi - mj).T

    return wij, mij, pij