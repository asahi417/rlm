""" Relation Embedding with pretrained Masked Language Models """
import os
import logging
from typing import List
from tqdm import tqdm
from multiprocessing import Pool

import transformers
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
__all__ = 'RelationEmbedding'


class EncodePlus:
    """ Wrapper of encode_plus for multiprocessing """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        if max_length:
            assert self.max_length >= max_length, '{} < {}'.format(self.max_length, max_length)
            self.max_length = max_length

    def __call__(self, word_pair):
        """ Encoding a word pair
        :param word_pair: two words
        :return: an output from tokenizer.encode_plus
        """
        assert len(word_pair) == 2, 'word_pair contains wrong number of tokens: {}'.format(len(word_pair))
        h, t = word_pair
        sentence = ' '.join([h] + [self.tokenizer.mask_token] + [t])
        param = {'max_length': self.max_length, 'truncation': True, 'padding': 'max_length'}
        encode = self.tokenizer.encode_plus(sentence, **param)
        assert encode['input_ids'][-1] == self.tokenizer.pad_token_id, 'exceeded max_length'
        encode['mask_position'] = encode['input_ids'].index(self.tokenizer.mask_token_id)
        return encode


class Dataset(torch.utils.data.Dataset):
    """ `torch.utils.data.Dataset` """
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data  # a list of dictionaries

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class RelationEmbedding:
    """ Get embedding representing the relation in between two words by pre-trained LM """

    def __init__(self, model: str, max_length: int = 32, cache_dir: str = None, num_worker: int = 0):
        """ Get embedding representing the relation in between two words by pre-trained LM
        :param model: a model name corresponding to a model card in `transformers`
        :param max_length: a model max length if specified, else use model_max_length
        """
        logging.debug('Initialize `Prompter`')
        assert 'bert' in model, '{} is not BERT'.format(model)
        self.num_worker = num_worker
        self.model_name = model
        self.cache_dir = cache_dir
        self.device = None
        self.model = None
        self.max_length = max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.config = transformers.AutoConfig.from_pretrained(
            model, cache_dir=cache_dir, output_attentions=True, output_hidden_states=True)
        self.num_hidden_layers = self.config.num_hidden_layers

    def __load_model(self):
        """ Load pretrained language model """
        if self.model:
            return
        logging.debug('loading language model')
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            self.model_name, config=self.config, cache_dir=self.cache_dir)
        self.model.eval()
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.model.to(self.device)
        logging.debug('running on {} GPU'.format(torch.cuda.device_count()))

    def get_embedding(self, word_pairs: List, batch_size: int = 4):
        """ Get relation embedding in two word
        :param word_pairs: a list of two words
        :param batch_size: batch size
        :return:
            - mask_positions: position of masked token (len(word_pairs), )
            - h_list: embedding corresponding to masked token (len(word_pairs), layer_n + 1, n_hidden)
            - a_list: weight attending on masked from context (len(word_pairs), layer_n, head_n, max_sequence)
        """
        if type(word_pairs[0]) is str:
            word_pairs = [word_pairs]
        logging.debug('Get relation embedding on {} pairs'.format(len(word_pairs)))

        # build dataset in parallel
        logging.debug('\t* preprocess dataset')
        pool = Pool()
        data = pool.map(EncodePlus(self.tokenizer, self.max_length), word_pairs)
        pool.close()
        data_loader = torch.utils.data.DataLoader(
            Dataset(data), num_workers=self.num_worker, batch_size=batch_size, shuffle=False, drop_last=False)

        logging.debug('\t* run LM inference')
        self.__load_model()
        mask_positions, h_list, a_list = [], [], []
        with torch.no_grad():
            for encode in tqdm(data_loader):
                encode = {k: v.to(self.device) for k, v in encode.items()}
                mask_position = encode.pop('mask_position').cpu().tolist()
                output = self.model(**encode, return_dict=True)
                assert 'hidden_states' in output.keys() and 'attentions' in output.keys(), str(output.keys())
                # hidden state of masked token: layer x batch x h_n
                hidden_states = list(map(
                        lambda h: list(map(lambda x: x[0][x[1]].cpu().tolist(), zip(h, mask_position))),
                        output['hidden_states']))
                # batch x layer x head x h_n
                h_list += list(map(list, zip(*hidden_states)))
                # attention weight attending to masked token attended from context: layer x batch x head x max_len
                attentions = list(map(
                    lambda a: list(map(lambda a_: a_[0][:, :, a_[1]].cpu().tolist(), zip(a, mask_position))),
                    output['attentions']))

                # batch x layer x head x h_n
                a_list += list(map(list, zip(*attentions)))

                mask_positions += mask_position
        return mask_positions, h_list, a_list

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
