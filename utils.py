import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_networkx


def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i])
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    return matrix


class DataGraph():
    def __init__(self, data, lengh):
        if lengh == 519:
            graphdata = np.asarray(data[1], dtype=object)
            data_index = np.asarray(data[0], dtype=object)
        else:
            graphdata = np.asarray(data[0], dtype=object)
            data_index = np.asarray(data[1], dtype=object)
        graph = np.zeros((lengh, lengh))
        for raw, index in zip(graphdata, data_index):
            for i in raw:
                graph[index][i] = 1
        self.graph = graph


class get_kalma():
    def __init__(self, stock_data):
        transition_matrix = []
        transition_covariance = []
        for n in range(len(stock_data)):
            data = stock_data[-100:, n:n + 4, :]
            transition_matrix_item = np.zeros((9, 9))
            transition_covariance_item = np.zeros((9, 9))
            for i in range(3):
                for j in range(9):
                    for k in range(9):
                        transition_matrix_item[j][k] += np.corrcoef(data[:, i, j], data[:, i + 1, k])[0, 1]
                        xyz = data[:, i, j] - data[:, i + 1, k]
                        transition_covariance_item[j][k] += np.var(xyz)
            transition_matrix_item /= 3
            transition_covariance_item /= 3
            transition_matrix.append(transition_matrix_item)
            transition_covariance.append(transition_covariance_item)
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        initial_state_mean = []
        initial_cov = []
        for n in range(len(stock_data)):

            data = stock_data[-100:, n:n + 4, :]
            stds = np.std(data, axis=(0, 1))
            initial_cov.append(np.diag(stds ** 2))
            initial_state = 0
            for i in range(4):
                initial_state += np.mean(data[:, i, :], axis=0)
            initial_state_mean.append(initial_state / 4)
        self.initial_state_mean = initial_state_mean
        self.initial_cov = initial_cov


def byol_loss_fn(x, y):
    return 2 - 2 * (x * y).sum(dim=-1)


def transpose_node_to_graph(feature, batch_size, hidden_dim):
    """transpose [B * N, F] to [B * F, N]"""
    feature = feature.view(batch_size, -1, hidden_dim).transpose(1, 2)
    feature = feature.reshape(batch_size * hidden_dim, -1)
    return feature


def get_graph_MS(MS):
    graph = nx.from_numpy_array(MS)
    data = from_networkx(graph)
    edge_index = data.edge_index
    return edge_index


class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0], dtype=object)
        H_T = data_masks(self.raw, n_node)

        b_test = 1.0 / H_T.sum(axis=1).reshape(1, -1)
        b_test2 = H_T.sum(axis=1).reshape(1, -1)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        k = H.sum(axis=1)
        data_weight1 = np.asarray(H.sum(axis=1).reshape(1, -1))[0]
        data_weight2 = np.asarray(H.sum(axis=1).reshape(1, -1))
        for i in range(len(data_weight1)):
            if (data_weight1[i] == 0):
                data_weight2[0][i] = 1
        DH = H.T.multiply(1.0 / data_weight2)
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency = DHBH_T.tocoo()

        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i + 1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def generate_batch(self, indicators):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / indicators)
        if self.length % indicators != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * indicators), n_batch)
        slices[-1] = np.arange(self.length - indicators, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask
