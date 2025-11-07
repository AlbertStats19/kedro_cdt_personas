# data-bbog-integration-fabrica-personas

## Overview

This is your new Kedro project, which was generated using `kedro 0.18.14`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project
You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```
> *Note:* The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)


## ADL Kedro command RUN


### Ejecución pipeline models_output

#### Opciones del comando de ejecución
- conf-source: Ruta en S3 donde se encuentra los archivos de configuración (catalog.yml y parameters*.yml)
- pipeline: Nombre del pipeline a ejecutar

#### Parámetros requeridos
- product: Sigla del producto, ej: FID, AH, LD etc.
- fecha_ejecucion: Fecha en formato ISO (YYYY-MM-DD), define el periodo y la fuente de datos a procesar. 
- variable_apertura: Según producto.
- target: Según el producto.


**Ejemplo de comando de ejecución**

```bash
kedro run \
--conf-source=s3://data-bbog-dev-resources/casos-uso/fabrica/configs/env=test \
--pipeline=model_output \
--params=product=FID,fecha_ejecucion=2025-04-01,variable_apertura=fid_cant_aper_mes,target=fid_cant_ap_group3
```

### Ejecucion de Backtesting al corte del 202505
kedro run --conf-source=./my_conf/ --pipeline=backtesting --params=product=CDT,fecha_ejecucion=2025-05-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_ap_group3

### Ejecucion de Backtesting al corte del 202507
kedro run --conf-source=./conf_mlops/ --pipeline=backtesting --params=product=CDT,fecha_ejecucion=2025-07-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_ap_group3


kedro run --conf-source=./conf_mlops/ --pipeline=backtesting --params=product=FID,fecha_ejecucion=2025-07-10,variable_apertura=fid_cant_aper_mes,target=fid_cant_ap_group3


kedro run --conf-source=./conf_mlops/ --pipeline=monitoreo --params=product=FID,fecha_ejecucion=2025-07-10,variable_apertura=fid_cant_aper_mes,target=fid_cant_ap_group3

## Modo True

C:\Users\Alber\anaconda3\envs\my_env312\Scripts\python.exe -W "default:Kedro is not yet fully compatible" -m kedro run --conf-source=./conf/ --pipeline=backtesting --params=product=CDT,fecha_ejecucion=2025-07-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_ap_group3

C:\Users\Alber\anaconda3\envs\my_env312\Scripts\python.exe -W "default:Kedro is not yet fully compatible" -m kedro run --conf-source=./conf/ --pipeline=monitoreo --params=product=CDT,fecha_ejecucion=2025-07-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_ap_group3

## Modo False

C:\Users\Alber\anaconda3\envs\my_env312\Scripts\python.exe -W "default:Kedro is not yet fully compatible" -m kedro run --conf-source=./conf/ --pipeline=backtesting --params=product=CDT,fecha_ejecucion=2025-07-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_aper_mes

C:\Users\Alber\anaconda3\envs\my_env312\Scripts\python.exe -W "default:Kedro is not yet fully compatible" -m kedro run --conf-source=./conf/ --pipeline=monitoreo --params=product=CDT,fecha_ejecucion=2025-07-10,variable_apertura=cdt_cant_aper_mes,target=cdt_cant_ap_group3