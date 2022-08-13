from mlflow.entities import Run, RunInfo
from typing import Optional, Dict, List, Union
from pathlib import Path
from mlflow_wrapper.folder_management import FolderManagement
import mlflow


class RunHandler:
    # Cache for already downloaded runs
    __runs: dict = {}

    def __init__(self, client=None, tracking_url: str = "http://127.0.0.1:5000"):

        if client is None:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_url)

        self._client = client

    @property
    def client(self):
        return self._client

    @staticmethod
    def get_run_name_by_run_id(run_id: str, runs: []) -> Optional[str]:
        run: Run
        for run in runs:
            if run.info.run_id == run_id:
                return run.data.tags.get('mlflow.runName')

        return None

    def get_run_by_id(self, experiment_id: str, run_id: str) -> Optional[Run]:
        # Get all cached runs
        cached_runs: List = self.__runs.get(experiment_id)

        if cached_runs is not None and len(cached_runs) != 0:
            for cached_run in cached_runs:
                if cached_run.info.run_id == run_id:
                    return cached_run

        # Find run from mlflow
        all_run_infos: [] = reversed(self._client.list_run_infos(experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self._client.get_run(run_info.run_id)

            if full_run.info.run_id == run_id:
                self.__add_run_to_cache(runs=cached_runs, experiment_id=experiment_id, run=full_run)
                return full_run

        return None

    def get_run_id_by_name(self, experiment_id: str, run_name: str, parent_run_id: str = None) -> Optional[str]:
        """
        Returns a run id for a given name in a given experiment
        @param experiment_id: The experiment id in which the run is located
        @param run_name:  The run name to search for
        @param parent_run_id:  The run name to search for
        @return: A run or None if not found
        """
        run: Run

        # Check cache
        runs: List = self.__runs.get(experiment_id)

        if runs is not None and len(runs) != 0:
            for run in runs:
                if run.data.tags.get('mlflow.runName') == run_name:
                    return run.info.run_id

        # Run not cached
        all_run_infos: reversed = reversed(self._client.list_run_infos(experiment_id=experiment_id))

        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self._client.get_run(run_info.run_id)
            if full_run.data.tags.get('mlflow.runName') == run_name:
                if parent_run_id is not None and full_run.data.tags.get('mlflow.parentRunId') != parent_run_id:
                    continue

                # Add to cache
                self.__add_run_to_cache(runs=runs, experiment_id=experiment_id, run=full_run)
                return full_run.info.run_id

        # Run not found
        return None

    def get_run_by_name(self, experiment_id: str, run_name: str, parent_run_id: str = None) -> Optional[Run]:
        """
        Returns a run for a given name in a given experiment
        @param experiment_id: The experiment id in which the run is located
        @param run_name:  The run name to search for
        @param parent_run_id:  The parent run id for the run to search for
        @return: A run or None if not found
        """
        run: Run
        run_name = run_name.strip()

        # Check cache
        runs: List = self.__runs.get(experiment_id)

        if runs is not None and len(runs) != 0:
            for run in runs:
                if run.data.tags.get('mlflow.runName') == run_name:
                    return run

        # Run not cached
        all_run_infos: [] = reversed(self._client.list_run_infos(experiment_id=experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self._client.get_run(run_info.run_id)

            if full_run.data.tags.get('mlflow.runName') == run_name:
                if parent_run_id is not None and full_run.data.tags.get('mlflow.parentRunId') != parent_run_id:
                    continue
                # Add to cache
                self.__add_run_to_cache(runs=runs, experiment_id=experiment_id, run=full_run)

                return full_run

        # Run not found
        return None

    def get_run(self, experiment_id: str, run_name: str, include_children: bool = True) -> List:
        """
        Get all runs for a specific experiment id and run name.
        Key is the parent run, values are the children runs
        @param experiment_id:
        @param run_name:
        @param include_children:
        @return: Returns a list of runs. The first run is the parent run
        Values returns are run objects
        """

        runs: List = []

        parent_run_id: str = self.get_run_id_by_name(experiment_id=experiment_id, run_name=run_name)
        if parent_run_id is None:
            return runs

        parent_run: Run = self._client.get_run(parent_run_id)

        if parent_run.info.lifecycle_stage == 'active':
            runs.append(parent_run)

        if not include_children:
            return runs

        # Run not cached
        all_run_infos: [] = reversed(self._client.list_run_infos(experiment_id))
        run_info: RunInfo
        for run_info in all_run_infos:
            full_run: Run
            full_run = self._client.get_run(run_info.run_id)

            if full_run.data.tags.get('mlflow.parentRunId') == parent_run_id:
                runs.append(full_run)

        return runs

    def delete_run(self, experiment_id: str, run_name: str, delete_children: bool = True):
        """
        Deletes the run with the given name. If multiple runs share the same name, only the first one is being deleted
        :param experiment_id: The experiment id where the run located
        :param run_name: The run name to be deleted
        :param delete_children: Should the children runs also be deleted?
        :return:
        """
        try:
            if delete_children:
                runs: List = self.get_run(experiment_id=experiment_id, run_name=run_name, include_children=True)
            else:
                runs: List = [self.get_run_by_name(experiment_id=experiment_id, run_name=run_name)]

            if len(runs) != 0:
                for run in runs:
                    run: Run
                    # Remove run from local cache
                    self.__runs.pop(run.info.run_id, None)
                    # Delete run from mlflow
                    if run.info.lifecycle_stage == 'active':
                        self._client.delete_run(run.info.run_id)
        except:
            raise

    def download_artifacts(self, save_path: Union[Path, str], run: Run = None, runs: [] = None,
                           mlflow_folder: str = None) -> dict:
        """
         Downloads all artifacts of the found runs. Creates download folder for each run
        @param save_path:  The path where the artifacts should be saved
        @param runs: Runs which should be considered
        @param run: The run which should be considered
        @param mlflow_folder: The specific folder to be downloaded for the given runs
        @return: Returns a dictionary with the run name as key and the directory as value
        """

        print("Started download of files...")

        if run is None and runs is None:
            raise ValueError("Please provide either a run to download or a list of runs")

        created_directories: dict = {}

        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Download single run
        if run is not None:
            run_save_path = Path(save_path, run.info.run_id)
            run_save_path = FolderManagement.create_directory(run_save_path, remove_if_exists=False)
            created_directories[run.info.run_id] = run_save_path
            if mlflow_folder is not None:
                self._client.download_artifacts(run_id=run.info.run_id, path=mlflow_folder,
                                                dst_path=str(run_save_path))
            else:
                self._client.download_artifacts(run_id=run.info.run_id, path="", dst_path=str(run_save_path))

            return created_directories

        # Download multiple runs
        for run in runs:
            try:
                run_path = Path(save_path, run.info.run_id)
                run_path = FolderManagement.create_directory(run_path, remove_if_exists=False)
                created_directories[run.info.run_id] = run_path
                if mlflow_folder is not None:
                    self._client.download_artifacts(run_id=run.info.run_id, path=mlflow_folder,
                                                    dst_path=str(run_path))
                else:
                    self._client.download_artifacts(run_id=run.info.run_id, path="", dst_path=str(run_path))

            except BaseException as ex:
                print(ex)
                continue
        print("Download complete.")
        return created_directories

    def __add_run_to_cache(self, runs: List, experiment_id: str, run: Run):
        # Add to cache
        if runs is None or len(runs) == 0:
            self.__runs[experiment_id] = [run]
        else:
            self.__runs[experiment_id].append(run)
