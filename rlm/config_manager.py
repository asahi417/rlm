import os
import random
import json
import string
import logging
import pickle
from glob import glob
from typing import List

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_random_string(length: int = 6, exclude: List = None):
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


def safe_open(_file):
    with open(_file, 'r') as f:
        return json.load(f)


class ConfigManager:
    """ configuration manager for `scoring_function.RelationScorer` """

    def __init__(self, export_dir: str, test: bool, **kwargs):
        """ configuration manager for `scoring_function.RelationScorer` """
        self.config = kwargs
        self.prefix = 'test' if test else 'valid'
        logging.info(' * configuration\n' +
                     '\n'.join(list(map(lambda x: '{} : {}'.format(x[0], x[1]), self.config.items()))))
        cache_dir = os.path.join(export_dir, kwargs['data'], kwargs['model'])
        self.flatten_score = None

        ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(cache_dir))}
        same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
        if len(same_config) != 0:
            assert len(same_config) == 1, 'duplicated config found {}'.format(same_config)
            self.cache_dir = same_config[0][0].replace('config.json', '')

            # load intermediate score
            logging.debug('load model logit')
            with open('{}/mask_position.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
                mask_position = pickle.load(fp)
            with open('{}/hidden_state.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
                hidden_state = pickle.load(fp)
            with open('{}/attention.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
                attention = pickle.load(fp)
            self.flatten_score = [mask_position, hidden_state, attention]
        else:
            self.cache_dir = os.path.join(cache_dir, get_random_string())

    def __cache_init(self):
        assert self.cache_dir is not None
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def cache_scores(self, mask_positions: List, hidden_states: List, attentions: List):
        """ cache scores """
        self.__cache_init()
        with open('{}/mask_position.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(mask_positions, fp)
        with open('{}/hidden_state.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(hidden_states, fp)
        with open('{}/attention.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(attentions, fp)
