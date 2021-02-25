""" UnitTest for scorer """
import unittest
import logging
import shutil
import os
import rlm

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
scorer = rlm.Scorer('albert-base-v1', max_length=32)


class Test(unittest.TestCase):
    """Test"""

    def test_scorer(self):
        if os.path.exists('./tests/output'):
            shutil.rmtree('./tests/output')
        scorer.analogy_test(
            test=True,
            export_dir='./tests/output',
            data='debug',
            batch_size=4)
        scorer.analogy_test(
            test=False,
            export_dir='./tests/output',
            data='debug',
            batch_size=4)
        rlm.export_csv_summary(export_dir='./tests/output')


if __name__ == "__main__":
    unittest.main()
