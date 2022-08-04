import unittest
import mlflow
import pandas as pd
from parameterized import parameterized
from src.mlflow_wrapper.experiment_handler import ExperimentHandler
from src.mlflow_wrapper.run_handler import RunHandler
from src.mlflow_wrapper.upload_handler import UploadHandler
from mlflow.entities import Run
import time
from typing import Dict, List
import shutil
from pathlib import Path


class TestRunHandler(unittest.TestCase):

    def test_get_run_by_name(self):
        run_handler: RunHandler = RunHandler()
        experiment_handler: ExperimentHandler = ExperimentHandler()

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name="Library Test Experiment")
        run_name: str = "Test run"

        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_param("TestRun", 1)

        run: Run = run_handler.get_run_by_name(experiment_id=experiment_id, run_name=run_name)
        self.assertIsNotNone(run)

        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)

    def test_get_run_by_id(self):
        run_handler: RunHandler = RunHandler()
        experiment_handler: ExperimentHandler = ExperimentHandler()

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name="Library Test Experiment")
        run_name: str = "Test run " + str(time.time())

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_param("TestRun", 1)

        run: Run = run_handler.get_run_by_id(experiment_id=experiment_id, run_id=run.info.run_id)
        self.assertIsNotNone(run)

        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)

    def test_get_run_id_by_name(self):
        run_handler: RunHandler = RunHandler()
        experiment_handler: ExperimentHandler = ExperimentHandler()

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name="Library Test Experiment")
        run_name: str = "Test run " + str(time.time())

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_param("TestRun", 1)

        run_id: str = run_handler.get_run_id_by_name(experiment_id=experiment_id, run_name=run_name)
        self.assertIsNotNone(run_id)

        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)

    def test_get_run_and_child_runs(self):
        run_handler: RunHandler = RunHandler()
        experiment_handler: ExperimentHandler = ExperimentHandler()

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name="Library Test Experiment")
        run_name: str = "Parent Run"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_param("TestRun", 1)

            with mlflow.start_run(experiment_id=experiment_id, run_name="child_run", nested=True) as child_run:
                mlflow.log_param("Child Run", 1)

        runs: List = run_handler.get_run_and_child_runs(experiment_id=experiment_id, run_name=run_name)
        self.assertEqual(2, len(runs))

        run_handler.delete_runs_and_child_runs(experiment_id=experiment_id, run_name=run_name)

    def test_download_artifacts(self):
        run_handler: RunHandler = RunHandler()
        experiment_handler: ExperimentHandler = ExperimentHandler()

        experiment_id: str = experiment_handler.get_experiment_id_by_name(experiment_name="Library Test Experiment")
        run_name: str = "Download Test Run"

        store_folder = Path("test_data")
        save_path = Path("test")

        if store_folder.exists():
            shutil.rmtree(store_folder)
        if save_path.exists():
            shutil.rmtree(save_path)

        upload_handler: UploadHandler = UploadHandler(save_path=store_folder)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_param("TestRun", 1)

            df = pd.DataFrame(columns=["A", "B"])

            upload_handler.upload_dataframe(data=df, file_name="uploaded_file.csv")

        run: Run = run_handler.get_run_by_name(experiment_id=experiment_id, run_name=run_name)

        run_handler.download_artifacts(run=run, save_path=save_path)

        df = pd.read_csv(f"{save_path}/{run.info.run_id}/uploaded_file.csv")
        self.assertIsNotNone(df)

        shutil.rmtree(store_folder)
        shutil.rmtree(save_path)


if __name__ == '__main__':
    unittest.main()
