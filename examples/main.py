import logging
import rlm

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main(lm):
    scorer = rlm.Scorer(lm, max_length=length)
    scorer.analogy_test(test=True, data=data, batch_size=batch, method=['embedding_cos', 'embedding_euc'])
    scorer.analogy_test(test=False, data=data, batch_size=batch, method=['embedding_cos', 'embedding_euc'])


if __name__ == '__main__':
    batch = 512
    length = 16
    # bats is large as it gets to be ~ 50GB in total
    for data in ['sat', 'google', 'u2', 'u4']:
        for model in ['roberta-large', 'bert-large-cased']:
            main(model)
    rlm.export_csv_summary()
