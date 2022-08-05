# MLFlow Wrapper

Mlflow Wrapper is a python library intended to abstract some functionality away from the developer
when interacting with the mlflow library. 
The library supports/ improves the handling of experiments and offers helper functions which are not available in mlflow by default.
Furthermore, the library offers functions to interact with Runs itself, and uploading/downloading files.



# Installation

```pip install mlflow-wrapper```


# Quick start

## Experiment Handler


```
from mlflow_wrapper.experiment_handler import ExperimentHandler

exp_handler: ExperimentHandler = ExperimentHandler()
```

This example will connect to your local mlflow instance listening on 127.0.0.1:5000.  
However, the constructor provides option for either a mlflow client (that is already setup properly) 
or another url to connect to. Providing a client will always override the provided url.

### Get experiment by name
Experiments can be queried by name. This function is natively supported by Mlflow. 
However, if an experiment does not exist, it just returns None. 
This function however, accepts a setting to create an experiment if it does not exist on the fly.  

```
experiment_id: str = exp_handler.get_experiment_by_name("Your experiment name", create_experiment=True)
```

### Create an experiment

Creates a new experiment with the given name. Raises an exception if the experiment already exists.

```
id: str = create_experiment(self, name: str) 
```


## Run Handler

The run handler is designed to extend to already existing mlflow api with convenience functions.
Like the experiment handler, the constructor supports parameters for an already created client or an
url to connect to a mlflow server instance.

```
from mlflow_wrapper.run_handler import RunHandler

run_handler: RunHandler = RunHandler()
```

### Get Run by Id
This function is included for completeness. Mlflows api offers this function as well.

```
get_run_by_id(self, experiment_id: str, run_id: str) -> Optional[Run]:

run:Run = run_handler.get_run_by_id(expiremnt_id, run_id)

```

Returns a run if found, else returns None

### Get Run Id By Name

Returns the run id for a run with the given name.  
As multiple runs can have the same names, the first run with this name will be returned.  
If the parent run id is provided, only a run that matches the name inside the parent run will be returned.


```
run_id: str = get_run_id_by_name(self, experiment_id: str, run_name: str, parent_run_id: str = None)

```


### Get Run by Name

Returns a run by name. Similar to other functions this returns only the first occurrence of the run with the name.
If a parent run id is provided, only children runs will be checked and if found returned

```
run: Run = run_handler.get_run_by_name(self, experiment_id: str, run_name: str, parent_run_id: str = None)
```


### Get parent and child run
Returns a list of runs, containing both parent run and all related child runs.
If no run is found, returns an empty list

```
runs: List = run_handler.get_run_and_child_runs(self, experiment_id: str, run_name: str) -> List:
```

### Delete parent and child runs

Mlflow allows to delete runs. However, if the parent run is deleted, the children runs are not deleted.
This function offers the ability to delete all runs for a given parent run.

```
run_handler.delete_runs_and_child_runs(self, experiment_id: str, run_name: str)
```


### Download artifacts

Mlflow allows to download artifacts. However, this can be sometimes tedious when a lot of files should be downloaded.
The function can download all artifacts by either one run or a list of runs.

```
run_handler.download_artifacts(self, save_path: Union[Path, str], run: Run = None, runs: [] = None,
                           mlflow_folder: str = None)
```


## Upload Handler

To create an instance of an upload handler:

```
upload_handler: UploadHandler = UploadHandler(save_path)

```

The save path is required, because mlflow cannot upload files directly from memory.
If the provided path does not exist, the handler will create the path for you.


### Upload dataframe

Uploads a pandas dataframe or pandas series. It does all the lifting, e.g. writing the files 
to the hard drive, uploading it to mlflow.

Use remove index, to remove the index from the dataframe, when writing it to the disk.
Specify the mlflow folder to create a folder structure inside of your mlflow run.

```
upload_handler.upload_dataframe(self, data: Union[pd.DataFrame, pd.Series], file_name: str, mlflow_folder: str = None,
                         remove_index: bool = True)
```

### Upload file

If you just want to upload a file from your hard drive, use this function.
The file name refers to the file which will be uploaded.  
mlflow_folder refers again to a folder inside the mlflow artifact store.

```
upload_handler.upload_file(self, file_name: str, mlflow_folder: str = None):
```




# Bugs & Issues

Please use the github issue tracker for issues. I will try to get to them asap.

# Feedback

Feedback is most welcomed and I will respond asap.

