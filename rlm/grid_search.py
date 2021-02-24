import tqdm
from itertools import product
from typing import List

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
PBAR = tqdm.tqdm()


def cos_similarity(a: List, b: List):
    assert len(a) == len(b)
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    inner_prod = sum(map(lambda x: x[0] * x[1], zip(a, b)))
    return inner_prod / (norm_a * norm_b)


class GridSearch:
    """ Grid Searcher """

    def __init__(self,
                 shared_config,
                 mask_position_dict,
                 hidden_state_dict,
                 attention_dict,
                 queries,
                 answers,
                 negative_permutation_weight,
                 method,
                 export_prediction: bool = False):
        """ Grid Searcher """
        # global variables
        self.shared_config = shared_config
        self.mask_position_dict = mask_position_dict
        self.hidden_state_dict = hidden_state_dict
        self.attention_dict = attention_dict
        self.queries = queries
        self.answers = answers
        self.export_prediction = export_prediction
        # local parameters for grid search
        if type(negative_permutation_weight) is not list:
            negative_permutation_weight = [negative_permutation_weight]
            if type(method) is not list:
                method = [method]
        self.all_config = list(product(method, negative_permutation_weight, AGGREGATOR_POSITIVE, AGGREGATOR_NEGATIVE))
        self.index = list(range(len(self.all_config)))

    def single_run(self, config_index: int):
        PBAR.update(1)
        method, np_weight, ppa, npa = self.all_config[config_index]

        def get_similarity(word_list):
            assert len(word_list) == 4, len(word_list)
            a, b, c, d = word_list
            q_embedding = self.hidden_state_dict['||'.join([a, b])]
            c_embedding = self.hidden_state_dict['||'.join([c, d])]
            return cos_similarity(q_embedding, c_embedding)

        if method == 'embedding_similarity':
            similarity = list(map(
                lambda q: [
                    AGGREGATOR[ppa](list(map(lambda c: list(map(get_similarity, c[0])), q))),
                    AGGREGATOR[npa](list(map(lambda c: list(map(get_similarity, c[1])), q)))
                ], self.queries))
            similarity_merged = list(map(lambda o: list(map(lambda s: s[0] - np_weight * s[1], o)), similarity))
            prediction = list(map(lambda x: x.index(max(x)), similarity_merged))
        else:
            raise ValueError('unknown method: {}'.format(method))

        assert len(prediction) == len(self.answers)
        accuracy = sum(map(lambda x: int(x[0] == x[1]), zip(prediction, self.answers))) / len(self.answers)
        tmp_config = {
            'method': method,
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
