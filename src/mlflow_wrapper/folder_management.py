from pathlib import Path
import shutil
from typing import Union


class FolderManagement:

    @staticmethod
    def create_directory(path_to_create: Union[Path, str], remove_if_exists: bool = True) -> Path:
        try:
            if isinstance(path_to_create, str):
                path_to_create: Path = Path(path_to_create)

            if path_to_create.exists() and remove_if_exists:
                shutil.rmtree(path_to_create)
            path_to_create.mkdir(parents=True, exist_ok=True)
            return path_to_create

        except BaseException as ex:
            raise

    @staticmethod
    def delete_directory(path_to_delete: Union[Path, str]):
        if isinstance(path_to_delete, str):
            path_to_delete: Path = Path(path_to_delete)

        if path_to_delete.exists():
            try:
                shutil.rmtree(path_to_delete)
            except:
                return
