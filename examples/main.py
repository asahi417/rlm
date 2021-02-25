import logging
import rlm

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main(lm):
    scorer = rlm.Scorer(lm, max_length=length)
    scorer.analogy_test(test=True, data=data, batch_size=batch)
    scorer.analogy_test(test=False, data=data, batch_size=batch)


if __name__ == '__main__':
    data = 'sat'
    batch = 512
    length = 16
    models = ['roberta-large', 'bert-large-cased']
    for m in models:
        main(m)
    rlm.export_csv_summary()
