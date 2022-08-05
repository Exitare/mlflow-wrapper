# MLFlow Wrapper

Mlflow Wrapper is a python library intended to abstract some functionality away from the developer
when interacting with the mlflow library.
The library supports/ improves the handling of experiments and offers helper functions which are not available in mlflow
by default.
Furthermore, the library offers functions to interact with Runs itself, and uploading/downloading files.

# Installation

```pip install mlflow-wrapper```

# Quick start

Some quick start examples. Please refer to the [wiki](https://github.com/Exitare/mlflow-wrapper/wiki)
for a complete overview of all functions.


## Experiment Handler

```
from mlflow_wrapper.experiment_handler import ExperimentHandler

# Connect to a local mlflow server
exp_handler:ExperimentHandler = ExperimentHandler()

# Get the id of a new experiment which does no exist so far
exp_id = exp_handler.get_experiment_by_name(experiment_name="New Experiment")


```

## Run Handler

```
from mlflow_wrapper.run_handler import RunHandler

run_handler: RunHandler = RunHandler()

# Delete a parent run and all associated children run. 
# Does only delete the first occurence of the given run name. If multiple runs do have the same name,
# this command needs to be executed multiple times

run_handler.delete_runs_and_child_runs(experiment_id=exp_id, run_name="My Run")


```

# Bugs & Issues

Please use the GitHub issue tracker for issues. I will try to get to them asap.

# Feedback

Feedback is most welcomed and I will respond asap.

