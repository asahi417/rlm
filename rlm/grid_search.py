""" Grid Searcher """
import pickle
import logging
from itertools import product
from typing import List
from tqdm import tqdm

__all__ = 'GridSearch'
AGGREGATOR = {
    'mean': lambda x: sum(x)/len(x), 'max': lambda x: max(x), 'min': lambda x: min(x),
    'index_0': lambda x: x[0], 'index_1': lambda x: x[1], 'index_2': lambda x: x[2], 'index_3': lambda x: x[3],
    'index_4': lambda x: x[4], 'index_5': lambda x: x[5], 'index_6': lambda x: x[6], 'index_7': lambda x: x[7],
    'index_8': lambda x: x[8], 'index_9': lambda x: x[9], 'index_10': lambda x: x[10], 'index_11': lambda x: x[11],
    'index_12': lambda x: x[12], 'index_13': lambda x: x[13], 'index_14': lambda x: x[14], 'index_15': lambda x: x[15],
    'none': lambda x: 0
}
AGGREGATOR_POSITIVE = ['mean', 'max', 'min'] + ['index_{}'.format(i) for i in range(4)]
AGGREGATOR_NEGATIVE = ['mean', 'max', 'min'] + ['index_{}'.format(i) for i in range(8)]


def euc_distance(a: List, b: List):
    assert len(a) == len(b)
    return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b))) ** 0.5


def cos_similarity(a: List, b: List):
    assert len(a) == len(b)
    inner_prod = sum(map(lambda x: x[0] * x[1], zip(a, b)))
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    return inner_prod / (norm_a * norm_b)


class GridSearch:
    """ Grid Searcher """

    def __init__(self,
                 shared_config,
                 word_pairs_flatten,
                 mask_positions,
                 path_hidden_state,
                 path_attention,
                 queries,
                 answers,
                 negative_permutation_weight,
                 method,
                 hidden_layer,
                 export_prediction: bool = False):
        """ Grid Searcher """
        # global variables
        self.shared_config = shared_config
        self.word_pairs_flatten = word_pairs_flatten
        self.mask_positions = mask_positions
        self.path_hidden_state = path_hidden_state
        self.path_attention = path_attention
        self.queries = queries
        self.answers = answers
        self.export_prediction = export_prediction
        # local parameters for grid search
        if type(negative_permutation_weight) is not list:
            negative_permutation_weight = [negative_permutation_weight]
            if type(method) is not list:
                method = [method]
        self.hidden_layer = hidden_layer
        self.h_dict, self.a_dict = self.load_statistics(hidden_layer)
        self.all_config = list(product(
            method, negative_permutation_weight, AGGREGATOR_POSITIVE, AGGREGATOR_NEGATIVE))
        self.index = list(range(len(self.all_config)))

    def load_statistics(self, layer):
        a_list, h_list = [], []
        logging.info('\t\t - loading hidden state: layer {}'.format(layer))
        for p in tqdm(self.path_hidden_state):
            with open(p, "rb") as fp:
                h_list += map(lambda x: x[layer], pickle.load(fp))
        try:
            logging.info('\t\t - loading attention: layer {}'.format(layer))
            for p in tqdm(self.path_attention):
                with open(p, "rb") as fp:
                    a_list += map(lambda x: x[layer], pickle.load(fp))
            assert len(self.mask_positions) == len(h_list) == len(a_list) == len(self.word_pairs_flatten), \
                str([len(self.mask_positions), len(h_list), len(a_list), len(self.word_pairs_flatten)])
            h_dict = {'||'.join(w): m for w, m in zip(self.word_pairs_flatten, h_list)}
            a_dict = {'||'.join(w): m for w, m in zip(self.word_pairs_flatten, a_list)}
            return h_dict, a_dict
        except IndexError:
            assert len(self.mask_positions) == len(h_list) == len(self.word_pairs_flatten), \
                str([len(self.mask_positions), len(h_list), len(self.word_pairs_flatten)])
            h_dict = {'||'.join(w): m for w, m in zip(self.word_pairs_flatten, h_list)}
            return h_dict, None

    def single_run(self, config_index: int):
        method, np_weight, ppa, npa = self.all_config[config_index]
        if method in ['embedding_cos', 'embedding_euc']:

            def get_similarity(word_list):
                assert len(word_list) == 4, len(word_list)
                a, b, c, d = word_list
                q_embedding = self.h_dict['||'.join([a, b])]
                c_embedding = self.h_dict['||'.join([c, d])]
                if method == 'embedding_cos':
                    return cos_similarity(q_embedding, c_embedding)
                else:
                    return - euc_distance(q_embedding, c_embedding)

            similarity = list(map(
                lambda q: [
                    list(map(lambda c: AGGREGATOR[ppa](list(map(get_similarity, c[0]))), q)),
                    list(map(lambda c: AGGREGATOR[npa](list(map(get_similarity, c[1]))), q))
                ], self.queries))
            similarity_merged = list(map(lambda o: list(map(lambda s: s[0] - np_weight * s[1], o)), similarity))
            prediction = list(map(lambda x: x.index(max(x)), similarity_merged))
        else:
            raise ValueError('unknown method: {}'.format(method))

        assert len(prediction) == len(self.answers)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(prediction, self.answers))) / len(self.answers)
        tmp_config = {
            'method': method,
            'layer': self.hidden_layer,
            'positive_permutation_aggregation': ppa,
            'negative_permutation_aggregation': npa,
            'negative_permutation_weight': np_weight,
            'accuracy': accuracy}
        tmp_config.update(self.shared_config)
        if self.export_prediction:
            tmp_config['prediction'] = prediction
        return tmp_config

    def __len__(self):
        return len(self.all_config)
