import unittest
import mlflow
import pandas as pd
import time
from src.mlflow_wrapper.experiment_handler import ExperimentHandler
from src.mlflow_wrapper.upload_handler import UploadHandler
from src.mlflow_wrapper.run_handler import RunHandler


class TestUploadHandler(unittest.TestCase):

    def test_upload_dataframe(self):
        exp_handler: ExperimentHandler = ExperimentHandler()
        experiment_id: str = exp_handler.get_experiment_id_by_name("Library Test Experiment")
        run_handler: RunHandler = RunHandler()
        self.assertIsNotNone(experiment_id)

        upload_handler: UploadHandler = UploadHandler(save_path="test_data")

        with mlflow.start_run(experiment_id=experiment_id, run_name="Upload Test") as run:
            upload_handler.upload_dataframe(data=pd.DataFrame(columns=['A', 'B']), file_name="Test Upload.csv")

        time.sleep(1)
        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name="Upload Test")

    def test_upload_file(self):
        exp_handler: ExperimentHandler = ExperimentHandler()
        experiment_id: str = exp_handler.get_experiment_id_by_name("Library Test Experiment")
        run_handler: RunHandler = RunHandler()
        self.assertIsNotNone(experiment_id)

        run_name: str = "Upload test run"

        upload_handler: UploadHandler = UploadHandler(save_path="test")

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            test_pd = pd.DataFrame(columns=['A', 'B'])
            test_pd.to_csv("test/test.csv")

            upload_handler.upload_file(file_name="test.csv")

        time.sleep(1)
        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)


if __name__ == '__main__':
    unittest.main()
