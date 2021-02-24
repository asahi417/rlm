import os
import json
import logging
import pickle
from typing import List


class ConfigManager:
    """ configuration manager for `scoring_function.RelationScorer` """

    def __init__(self, export_dir: str, test: bool, **kwargs):
        """ configuration manager for `scoring_function.RelationScorer` """
        self.config = kwargs
        self.prefix = 'test' if test else 'valid'
        logging.info('configuration')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, v))
        self.cache_dir = os.path.join(export_dir, kwargs['data'], kwargs['model'])
        self.flatten_score = None
        try:
            with open('{}/word_pairs_flatten.{}.pkl'.format(self.cache_dir, self.prefix), "rb") as fp:
                word_pairs_flatten = pickle.load(fp)
            with open('{}/mask_position.{}.pkl'.format(self.cache_dir, self.prefix), "rb") as fp:
                mask_position = pickle.load(fp)
            with open('{}/hidden_state.{}.pkl'.format(self.cache_dir, self.prefix), "rb") as fp:
                hidden_state = pickle.load(fp)
            with open('{}/attention.{}.pkl'.format(self.cache_dir, self.prefix), "rb") as fp:
                attention = pickle.load(fp)
            self.flatten_score = [word_pairs_flatten, mask_position, hidden_state, attention]
            logging.info('load cached logit: skip inference')
        except FileNotFoundError:
            logging.info('no cache found: run inference')

    def __cache_init(self):
        assert self.cache_dir is not None
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def cache_scores(self, word_pairs_flatten, mask_positions: List, hidden_states: List, attentions: List):
        """ cache scores """
        self.__cache_init()
        with open('{}/word_pairs_flatten.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(word_pairs_flatten, fp)
        with open('{}/mask_position.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(mask_positions, fp)
        with open('{}/hidden_state.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(hidden_states, fp)
        with open('{}/attention.{}.pkl'.format(self.cache_dir, self.prefix), "wb") as fp:
            pickle.dump(attentions, fp)
