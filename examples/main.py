import logging
import rlm

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
batch = 512
length = 16
negative_permutation_weight = [0, 0.2, 0.4, 0.6]


def main(lm):
    scorer = rlm.Scorer(lm, max_length=length)
    scorer.analogy_test(test=True, data=data, batch_size=batch, method=['embedding_cos', 'embedding_euc'],
                        negative_permutation_weight=negative_permutation_weight)
    scorer.analogy_test(test=False, data=data, batch_size=batch, method=['embedding_cos', 'embedding_euc'],
                        negative_permutation_weight=negative_permutation_weight)


if __name__ == '__main__':
    # bats is large as it gets to be ~ 50GB in total
    for data in ['sat', 'google', 'u2', 'u4']:
        for model in ['roberta-large', 'bert-large-cased']:
            main(model)
    rlm.export_csv_summary()
