from pathlib import Path
import mlflow
from typing import Union
import pandas as pd


class UploadHandler:

    def __init__(self, save_path: Union[str, Path]):
        self._save_path: Path = save_path if isinstance(save_path, Path) else Path(save_path)

        # Create folder if it does not exist
        if not self._save_path.exists():
            self._save_path.mkdir(parents=True, exist_ok=True)

    def upload_dataframe(self, data: Union[pd.DataFrame, pd.Series], file_name: str, mlflow_folder: str = None,
                         remove_index: bool = True):
        """
        Uploads a dataframe or pandas series from memory
        :param data: The data to be uploaded
        :param file_name: The file name of the file to be uploaded
        :param mlflow_folder: Optional: A mlflow folder which will be generated if not existing
        :param remove_index: Removes the index of the pandas dataframe/series if provided. Default = True
        :return:
        """

        try:
            save_path = Path(self._save_path, file_name)

            if remove_index:
                data.to_csv(save_path, index=False)
            else:
                data.to_csv(save_path, index=True)

            if mlflow_folder is not None:
                mlflow.log_artifact(str(save_path), mlflow_folder)
            else:
                mlflow.log_artifact(str(save_path))
        except:
            raise

    def upload_file(self, file_name: str, mlflow_folder: str = None):
        """
        Uploads a file from the file system
        :param file_name: The file name to upload
        :param mlflow_folder: Optional: A mlflow folder which will be generated if not existing
        :return:
        """

        try:
            save_path = Path(self._save_path, file_name)
            if mlflow_folder is not None:
                mlflow.log_artifact(str(save_path), mlflow_folder)
            else:
                mlflow.log_artifact(str(save_path))

        except:
            raise
