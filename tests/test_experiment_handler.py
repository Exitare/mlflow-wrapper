import unittest
from parameterized import parameterized

from src.mlflow_wrapper.experiment_handler import ExperimentHandler


class TestExperimentHandler(unittest.TestCase):

    def test_get_experiment_id_by_name(self):
        exp_handler: ExperimentHandler = ExperimentHandler()
        experiment_id: str = exp_handler.get_experiment_id_by_name("Library Test Experiment")
        self.assertIsNotNone(experiment_id)


if __name__ == '__main__':
    unittest.main()
