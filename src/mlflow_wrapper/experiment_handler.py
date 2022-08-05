import mlflow.exceptions
from typing import Optional
from mlflow.exceptions import ErrorCode
from mlflow.entities import ViewType


class ExperimentHandler:

    def __init__(self, client=None, tracking_url: str = "http://127.0.0.1:5000"):
        if client is None and tracking_url is None:
            raise ValueError("Please provide either a client object or a tracking url")

        if client is None:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_url)

        self._client = client

    @property
    def client(self):
        return self._client

    def get_experiment_id_by_name(self, experiment_name: str, experiment_description: str = None,
                                  create_experiment: bool = True) -> Optional[str]:
        """
        Gets the experiment id associated with the given experiment name.
        If no experiment is found by default a new experiment will be created
        @param experiment_name: The experiment name
        @param experiment_description: The description for a new experiment
        @param create_experiment: Should the experiment be created if it does not exist
        @return: The experiment id
        """

        # The experiment id
        found_experiment_id = None

        experiments = self._client.list_experiments(
            view_type=ViewType.ACTIVE_ONLY)  # returns a list of mlflow.entities.Experiment
        for experiment in experiments:
            if experiment.name == experiment_name:
                found_experiment_id = experiment.experiment_id

        if found_experiment_id is None and create_experiment:
            found_experiment_id = self.create_experiment(name=experiment_name, description=experiment_description)
        elif found_experiment_id is None and not create_experiment:
            raise ValueError(
                "Could not find experiment! Please provide a valid experiment name, or set create_experiment to True")
        return found_experiment_id

    def create_experiment(self, name: str, description: str = "") -> str:
        """
        Creates a new experiment with the given name
        @param name: The name of the experiment
        @param description: The description for the experiment
        @return: The string of the newly created experiment
        """
        try:
            experiment_id: str = self._client.create_experiment(name=name)
            self._client.set_experiment_tag(experiment_id, "description", description)
            return experiment_id

        except mlflow.exceptions.RestException as ex:
            raise

        except BaseException as ex:
            raise
