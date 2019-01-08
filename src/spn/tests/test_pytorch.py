import unittest
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.structure.leaves.parametric.parametric import Gaussian
from spn.structure.StatisticalTypes import MetaType
import numpy as np
from spn.structure.Base import Context
import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque

# class TestPytorch(unittest.TestCase):


def test_eval_gaussian():
    np.random.seed(17)
    data = np.random.normal(10, 0.01, size=2000).tolist() + \
        np.random.normal(30, 10, size=2000).tolist()
    data = np.array(data).reshape((-1, 10))
    data = data.astype(np.float32)

    ds_context = Context(meta_types=[MetaType.REAL] * data.shape[1],
                         parametric_types=[Gaussian] * data.shape[1])
    spn = learn_parametric(data, ds_context)
    # ll = log_likelihood(spn, data)

    # tf_ll = eval_pytorch(spn, data)
    return spn
    # self.assertTrue(np.all(np.isclose(ll, tf_ll)))


spn = test_eval_gaussian()

sum_type = "sum"
prd_type = "prd"


class CVMetaData(object):
    def __init__(self, spn):
        self.depth = 0
        self.masks_by_level = []
        self.type_by_level = []
        self.level_label_by_node = {}
        self.num_nodes_by_level = []
        self.nodes_by_level = []
        # The input index of the i-th leaf.
        self.leaves_input_indices = []
        self.get_cv_metadata(spn)

    def get_cv_metadata(self, spn):
        '''
        Returns a node labelling scheme.
        Perform a per level labelling to each node.
        Defines the order in which each node appears in the matrix.
        '''
        level = 0
        q = deque([spn])
        visited = {}
        # Perform a per level traversal
        while q:
            pdb.set_trace()
            level_size = len(q)
            curr_level = []  # nodes at the current level
            level_type = None
            for i in range(level_size):
                node = q.popleft()
                curr_level.append(node)
                if isinstance(node, Leaf):
                    continue
                else:
                    node_type = sum_type if isinstance(node, Sum) else prd_type
                    if level_type is None:
                        level_type = node_type
                    elif node_type != level_type:
                        error = "Level type mismatch: Expects " + level_type + " gets " + node_type
                        raise Exception(error)
                for child in node.children:
                    if child in visited:
                        continue
                    visited[child] = True
                    q.append(child)
            if level_type is None:  # HACK: Level type is always none on leaf layer. Fix this!
                curr_level = cv.get_ordered_leaves(curr_level)
            self.type_by_level.append(level_type)
            self.nodes_by_level.append(curr_level)
            self.num_nodes_by_level.append(len(curr_level))
            for (label, node) in enumerate(curr_level):
                self.level_label_by_node[node] = label
            level += 1
        self.depth = level
        self.masks_by_level = self.get_masks_by_level(cv)
        self.leaves_network_id = self.get_leaves_network_id(cv)
        self.leaves_input_indices = self.get_leaves_input_indices(cv)
        self.num_leaves_per_network = cv.x_size * cv.y_size

    def get_masks_by_level(self, cv):
        '''
        Returns a TorchSPN style layer information corresponding to ConvSPN cv
        '''
        masks_by_level = []
        for cur_level in range(self.depth - 1):
            next_level = cur_level + 1
            cur_level_count = self.num_nodes_by_level[cur_level]
            next_level_count = self.num_nodes_by_level[next_level]
            level_mask = np.zeros((cur_level_count, next_level_count)).astype('float32')
            cur_level_nodes = self.nodes_by_level[cur_level]
            for cur_node in cur_level_nodes:
                cur_label = self.level_label_by_node[cur_node]
                for child_node in cur_node.children:
                    child_label = self.level_label_by_node[child_node]
                    level_mask[cur_label][child_label] = 1
            level_mask = level_mask.T
            masks_by_level.append(level_mask)
        return masks_by_level

    def get_leaves_input_indices(self, cv):
        leaves = self.nodes_by_level[-1]
        leaves_input_indices = []
        input_size = cv.x_size * cv.y_size
        for leaf in leaves:
            index = int(leaf.y) * cv.x_size + int(leaf.x) + leaf.network_id * input_size
            leaves_input_indices.append(index)
        return leaves_input_indices

    def get_leaves_network_id(self, cv):
        leaves = self.nodes_by_level[-1]
        leaves_network_id = []
        for leaf in leaves:
            leaves_network_id.append(int(leaf.network_id))
        return leaves_network_id


def get_edge_count_from_layers(cv_layers):
    '''
    Was used to test the correctness of edge count in the conversion
    '''
    edge_count = []
    for layer in cv_layers:
        edge_count.append(np.count_nonzero(layer))
    return edge_count


cvm = CVMetaData(spn)
