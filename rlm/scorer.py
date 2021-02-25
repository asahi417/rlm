import logging
import os
import json
import pickle
from typing import List
from multiprocessing import Pool
from itertools import chain

import pandas as pd
from .lm import RelationEmbedding
from .data import get_dataset, get_dataset_raw
from .grid_search import GridSearch

__all__ = ('export_csv_summary', 'Scorer')
default_export_dir = './output'


def export_csv_summary(export_dir=default_export_dir):

    def _export_csv_summary(_test):
        prefix = 'test' if _test else 'valid'
        path_jl = '{}/.cache.{}.jsonl'.format(export_dir, prefix)
        assert os.path.exists(path_jl), 'file not found: {}'.format(path_jl)
        # save as a csv
        with open(path_jl, 'r') as f:
            json_line = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
        logging.debug('jsonline with {} lines'.format(len(json_line)))
        df = pd.DataFrame(json_line)
        df = df.drop_duplicates()
        df = df.sort_values(by='accuracy', ascending=False)
        return df

    df_val = _export_csv_summary(True)
    df_test = _export_csv_summary(False)

    accuracy_val = df_val.pop('accuracy').to_numpy()
    accuracy_test = df_test.pop('accuracy').to_numpy()
    assert df_val.shape == df_test.shape

    df_test['accuracy_validation'] = accuracy_val
    df_test['accuracy_test'] = accuracy_test

    df_test['accuracy'] = (accuracy_val * 37 + accuracy_test * 337) / (37 + 337)
    df_test = df_test.sort_values(by=['accuracy'], ascending=False)
    df_test.to_csv('{}/summary.full.csv'.format(export_dir))


def pool_map(f, arg):
    _pool = Pool()
    out = _pool.map(f, arg)
    _pool.close()
    return out


def get_partition(_list):
    """ Get partition in multiprocess """
    p = Partition(_list)
    return pool_map(p, range(len(_list)))


class Partition:
    """ Get the partition information of a nested list for restoring the original structure """

    def __init__(self, _list):
        self.length = pool_map(len, _list)

    def __call__(self, x):
        return [sum(self.length[:x]), sum(self.length[:x + 1])]


class Scorer:
    """ Scoring relations with language models """

    def __init__(self,
                 model: str = 'roberta-base',
                 max_length: int = 32,
                 cache_dir: str = None,
                 num_worker: int = 1):
        """ Scoring relations with language models

        :param model: LM parameter
        :param max_length: LM parameter
        :param cache_dir: LM parameter
        :param num_worker: LM parameter
        """
        logging.debug('*** setting up a scorer ***')
        # language model setup
        self.max_length = max_length
        self.lm = RelationEmbedding(model=model, max_length=max_length, cache_dir=cache_dir, num_worker=num_worker)
        self.model_name = model

    def analogy_test(self,
                     data: str,
                     export_dir: str = default_export_dir,
                     method: (str, List) = 'embedding',
                     test: bool = False,
                     batch_size: int = 32,
                     no_inference: bool = False,
                     skip_scoring_prediction: bool = False,
                     export_prediction: bool = False,
                     negative_permutation_weight: (float, List) = 1.0):
        """ Test Analogy with Relation embedding """
        logging.info('## ANALOGY TEST ##')
        logging.info('Model inference')
        answers, word_pairs, queries = get_dataset(data=data, test_set=test)
        word_pairs_flatten = list(set(list(chain(*word_pairs))))
        logging.info('\t * dataset `{}`: {} relations'.format(data, len(word_pairs_flatten)))
        logging.info('\t * model       : {}'.format(self.model_name))

        logging.info('Configuration manager')
        prefix = 'test' if test else 'valid'
        cache_dir = os.path.join(export_dir, data, self.model_name)
        try:
            with open('{}/word_pairs_flatten.{}.pkl'.format(cache_dir, prefix), "rb") as fp:
                word_pairs_flatten = pickle.load(fp)
            with open('{}/mask_position.{}.pkl'.format(cache_dir, prefix), "rb") as fp:
                mask_positions = pickle.load(fp)
            with open('{}/hidden_state.{}.pkl'.format(cache_dir, prefix), "rb") as fp:
                h_list = pickle.load(fp)
            with open('{}/attention.{}.pkl'.format(cache_dir, prefix), "rb") as fp:
                a_list = pickle.load(fp)
            logging.info('load cached logit: skip inference')
        except FileNotFoundError:
            logging.info('no cache found: run inference')
            assert not no_inference, '"no_inference==True" but no cache found'
            mask_positions, h_list, a_list = self.lm.get_embedding(word_pairs_flatten, batch_size=batch_size)

            logging.info('save logits')
            os.makedirs(cache_dir, exist_ok=True)
            if not os.path.exists('{}/config.json'.format(cache_dir)):
                with open('{}/config.json'.format(cache_dir), 'w') as f:
                    json.dump({'model': self.model_name, 'data': data}, f)
            with open('{}/word_pairs_flatten.{}.pkl'.format(cache_dir, prefix), "wb") as fp:
                pickle.dump(word_pairs_flatten, fp)
            with open('{}/mask_position.{}.pkl'.format(cache_dir, prefix), "wb") as fp:
                pickle.dump(mask_positions, fp)
            with open('{}/hidden_state.{}.pkl'.format(cache_dir, prefix), "wb") as fp:
                pickle.dump(h_list, fp)
            with open('{}/attention.{}.pkl'.format(cache_dir, prefix), "wb") as fp:
                pickle.dump(a_list, fp)

        if skip_scoring_prediction:
            return

        logging.info('Get prediction')
        assert len(mask_positions) == len(h_list) == len(a_list) == len(word_pairs_flatten),\
            str([len(mask_positions), len(h_list), len(a_list), len(word_pairs_flatten)])
        mask_position_dict = {'||'.join(w): m for w, m in zip(word_pairs_flatten, mask_positions)}
        h_dict = {'||'.join(w): m for w, m in zip(word_pairs_flatten, h_list)}
        a_dict = {'||'.join(w): m for w, m in zip(word_pairs_flatten, a_list)}

        pool = Pool()
        searcher = GridSearch(
            shared_config={'model': self.model_name, 'data': data},
            method=method,
            mask_position_dict=mask_position_dict,
            hidden_state_dict=h_dict,
            attention_dict=a_dict,
            queries=queries,
            answers=answers,
            num_hidden_layers=self.lm.num_hidden_layers,
            negative_permutation_weight=negative_permutation_weight,
            export_prediction=export_prediction)
        logging.info('\t * start grid search: {} combinations'.format(len(searcher)))
        logging.info('\t * multiprocessing  : {} cpus'.format(os.cpu_count()))
        json_line = pool.map(searcher.single_run, searcher.index)
        pool.close()

        logging.info('\t * export to {}'.format(export_dir))

        if export_prediction:
            NotImplementedError('TBA')
            # logging.debug('export prediction mode')
            # assert len(json_line) == 1, 'more than one config found: {}'.format(len(searcher))
            # json_line = json_line[0]
            # val_set, test_set = get_dataset_raw(data)
            # data_raw = test_set if test else val_set
            # prediction = json_line.pop('prediction')
            # assert len(prediction) == len(data_raw), '{} != {}'.format(len(prediction), len(data_raw))
            # for d, p in zip(data_raw, prediction):
            #     d['prediction'] = p
            # os.makedirs('{}/prediction'.format(export_dir), exist_ok=True)
            # _file = '{}/prediction/{}.{}.{}.csv'.format(
            #     export_dir, data, self.model_name, config.prefix)
            # pd.DataFrame(data_raw).to_csv(_file)
            # logging.debug("prediction exported: {}".format(_file))
            # return
        # save as a json line
        path_jl = '{}/.cache.{}.jsonl'.format(export_dir, prefix)
        if os.path.exists(path_jl):
            with open(path_jl, 'a') as writer:
                writer.write('\n')
                writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))
        else:
            with open(path_jl, 'w') as writer:
                writer.write('\n'.join(list(map(lambda x: json.dumps(x), json_line))))
