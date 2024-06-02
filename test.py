import unittest
from utils.train_nc import main

class TestRun(unittest.TestCase):
    def test_working(self):
        # dummy values for testing purposes
        seed = 3
        dataset_name = 'cora'
        model_name = 'gcn'
        gamma = explore = alpha = beta = gamma = 1

        original_acc = main(seed=seed, dataset_name=dataset_name, extend_metric='None', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
        self.assertIsNotNone(original_acc)

        test_run = main(seed=seed, dataset_name=dataset_name, extend_metric='degree', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
        self.assertIsNotNone(test_run)

        test_run = main(seed=seed, dataset_name=dataset_name, extend_metric='adamic_adar', model_name=model_name, alpha=alpha, beta=beta, gamma=gamma, explore=explore)
        self.assertIsNotNone(test_run)

if __name__ == '__main__':
    unittest.main()