import os
import logging
import requests
import zipfile
import json
from itertools import combinations, chain

__all__ = 'get_dataset'
default_cache_dir = './data'
root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/raw/master'


def sampling_permutation(a, b, c, d):
    positive = [(a, b, c, d), (a, c, b, d), (b, a, d, c), (c, a, d, b)]
    negative = [(a, b, d, c), (a, c, d, b), (a, d, b, c), (a, d, c, b),
                (b, a, c, d), (b, c, d, a), (b, d, c, a), (c, b, d, a)]
    return positive, negative


def wget(url, cache_dir):
    logging.debug('downloading zip file from {}'.format(url))
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    with zipfile.ZipFile('{}/{}'.format(cache_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    os.remove('{}/{}'.format(cache_dir, filename))


def get_dataset_raw(data_name: str, cache_dir: str = default_cache_dir):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats', 'sample'], 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        url = '{}/{}.zip'.format(root_url_analogy, data_name)
        wget(url, cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val, test


def get_dataset(data: str, test_set: bool = True):
    """ Get prompted SAT-type dataset """

    def single_entry(dictionary):
        a, b = dictionary['stem']

        def comb(c, d):
            c = list(combinations([a, b, c, d], 2))
            c.pop(c.index((a, b)))
            return c

        _list = [(a, b)]
        _list += list(chain(*list(map(lambda x: comb(*x), dictionary['choice']))))
        perm = list(map(lambda x: sampling_permutation(a, b, x[0], x[1]), dictionary['choice']))
        return dictionary['answer'], _list, perm

    val, test = get_dataset_raw(data)
    if test_set:
        data = list(map(lambda x: single_entry(x), test))
    else:
        data = list(map(lambda x: single_entry(x), val))
    list_answer = list(list(zip(*data))[0])
    list_word_pair = list(list(zip(*data))[1])
    list_query = list(list(zip(*data))[2])
    return list_answer, list_word_pair, list_query
