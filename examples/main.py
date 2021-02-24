import rlm


def main(lm):
    scorer = rlm.Scorer(lm, max_length=length)
    scorer.analogy_test(test=True, data=data, batch_size=batch)
    scorer.analogy_test(test=False, data=data, batch_size=batch)


data = 'sat'
batch = 512
length = 16
models = ['roberta-large', 'bert-large-cased']
for m in models:
    main(m)
rlm.export_csv_summary()
