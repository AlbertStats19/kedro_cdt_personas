"""
Nodos de la capa models
"""

import numpy as np
import math

# import tkinter as tk
# Cambiar el backend
# Qt5Agg
# matplotlib.use('WebAgg') #            #matplotlib.use('TkAgg') TkAgg, Qt5Agg, GTK3Agg --> pip install pyqt5  --> sudo apt-get install python3-tk
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import logging
from typing import Dict, Any
from sklearn.metrics import classification_report
import gc


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 0.0 Funcion Auxiliar para tratar la Y de interes
def tratamiento_y(X_i, y_i, params):
    """
    Funcion auxiliar para tratar cuando la variable de interes es nula
    """
    metodo = params["y_method"]
    valor = params["y_method_value"]
    if int(y_i[[params["target"]]].isnull().sum().iloc[0]) == 0:
        X_i2 = X_i.copy()
        y_i2 = y_i.copy()
    else:
        if metodo == "drop":
            logger.info(f"Ajuste de y: {metodo}")
            X_i2 = X_i[y_i.isnull() is False]
            y_i2 = y_i[y_i.isnull() is False]
        elif metodo == "fillna":
            logger.info(f"Ajuste de y: {metodo}")
            y_i2 = y_i.fillna(valor)
            X_i2 = X_i.copy()
        else:
            X_i2 = X_i.copy()
            y_i2 = y_i.copy()
    gc.collect()
    return X_i2, y_i2


# 0.0.1 Funcion Axuliar para clinear los inputs:
def clean_x(dff: pd.DataFrame, params: Dict[str, Any]):
    """
    Funcion auxiliar para quitar la variable id y la variable objetivo
    """
    target = [params["target"]]
    ids = [params["id"]]
    for col in ids + target:
        try:
            dff = dff.drop(col, axis=1)
        except Exception:
            pass
    return dff


def run_clean_x_tratamiento_y(
    X_train: pd.DataFrame, y_train: pd.DataFrame, params: Dict[str, Any]
):
    """
    Funcion auxiliar para limpiar los inputs del output y procesar los nulos de los outputs
    """
    target = [params["target"]]
    X_clean = clean_x(X_train, params)
    X_i2, y_i2 = tratamiento_y(X_clean, y_train[target], params)
    return X_i2, y_i2


# 0.1.0 Funcion Auxiliar para balanceos
def balance_osc(
    XX: pd.DataFrame, yy: pd.DataFrame, params: Dict[str, Any], group: [float, str, Any]
) -> pd.DataFrame:
    """
    Balancea la variable objetivo utilizando el método Synthetic Minority Over-sampling Technique (SMOTE) en dos DataFrames.

    Parameters
    ----------
    XX : pd.DataFrame
        Primer DataFrame de Pandas que se balanceará.
    yy : pd.DataFrame
        Primer DataFrame de Pandas que se busca balancear.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    group: [float,str, Any]
        Grupo del diccionario de parametros model input que se ejecutara para balancear

    Returns
    -------
        pd.DataFrame
            DataFrame con la target balanceada.
    """
    target = [params["target"]]
    ids = [params["id"]]
    random_state = params["balance_target_variable"]["random_state"]

    sampling_strategy = params["balance_target_variable"]["Muestreo"][group][
        "sampling_strategy_osc"
    ]
    sampling_strategy2 = params["balance_target_variable"]["Muestreo"][group][
        "sampling_strategy_osc2"
    ]
    last_balance = params["balance_target_variable"]["Muestreo"][group]["Use_auto"]
    Type = params["balance_target_variable"]["Muestreo"][group]["Type"]
    df = pd.concat([XX, yy], axis=1)
    try:
        y = df[target]
        X = df.drop(columns=target + ids, axis=1)
    except Exception:
        y = df[target]
        X = df.copy()
        for col in target + ids:
            try:
                X.drop(col, axis=1, inplace=True)
            except Exception:
                pass
    counts_before = yy.value_counts().to_frame()
    counts_before["%"] = counts_before.values / yy.shape[0]
    logger.info(f"Conteo de clases antes del balanceo: {counts_before}")

    if Type == "Smote":
        if last_balance is True:
            sampling_strategy = "auto"
        smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(f"Conteo de clases después del balanceo Smote: {sampling_strategy}")
        logger.info(f"{counts_after}")

    elif Type == "Undersampling":
        if last_balance is True:
            sampling_strategy = "auto"
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Subsample: {sampling_strategy}"
        )
        logger.info(f"{counts_after}")

    elif Type == "Oversampling":
        if last_balance is True:
            sampling_strategy = "auto"
        ros = RandomOverSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = ros.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Oversampling: {sampling_strategy}"
        )
        logger.info(f"{counts_after}")

    elif Type == "ADASYN":
        if last_balance is True:
            sampling_strategy = "auto"
        adasyn = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Oversampling (ADASYN): {sampling_strategy}"
        )
        logger.info(f"{counts_after}")
    elif Type == "Undersampling-ADASYN":
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Subsample: {sampling_strategy}"
        )
        logger.info(f"{counts_after}")
        X1 = df_resampled.drop(columns=target, axis=1)
        y1 = df_resampled[target]
        if last_balance is True:
            sampling_strategy2 = "auto"
        adasyn = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy2)
        X_resampled1, y_resampled1 = adasyn.fit_resample(X1, y1)
        df_resampled1 = pd.DataFrame(X_resampled1, columns=X1.columns)
        df_resampled1[target] = y_resampled1
        counts_after = y_resampled1.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Oversampling (ADASYN): {sampling_strategy2}"
        )
        logger.info(f"{counts_after}")
        df_resampled = df_resampled1.copy()

    elif Type == "Undersampling-Smote":
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Subsample: {sampling_strategy}"
        )
        logger.info(f"{counts_after}")
        X1 = df_resampled.drop(columns=target, axis=1)
        y1 = df_resampled[target]
        if last_balance is True:
            sampling_strategy2 = "auto"
        smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy2)
        X_resampled1, y_resampled1 = smote.fit_resample(X1, y1)
        df_resampled1 = pd.DataFrame(X_resampled1, columns=X1.columns)
        df_resampled1[target] = y_resampled1
        counts_after = y_resampled1.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Smote: {sampling_strategy2}"
        )
        logger.info(f"{counts_after}")
        df_resampled = df_resampled1.copy()

    elif Type == "Undersampling-Oversampling":
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = rus.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Subsample: {sampling_strategy}"
        )
        logger.info(f"{counts_after}")
        X1 = df_resampled.drop(columns=target, axis=1)
        y1 = df_resampled[target]
        if last_balance is True:
            sampling_strategy2 = "auto"
        over = RandomOverSampler(
            random_state=random_state, sampling_strategy=sampling_strategy2
        )
        X_resampled1, y_resampled1 = over.fit_resample(X1, y1)
        df_resampled1 = pd.DataFrame(X_resampled1, columns=X1.columns)
        df_resampled1[target] = y_resampled1
        counts_after = y_resampled1.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(
            f"Conteo de clases después del balanceo Oversampling: {sampling_strategy2}"
        )
        logger.info(f"{counts_after}")
        df_resampled = df_resampled1.copy()

    elif Type == "Smote-Undersampling":
        smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target] = y_resampled
        counts_after = y_resampled.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()
        logger.info(f"Conteo de clases después del balanceo Smote: {sampling_strategy}")
        logger.info(f"{counts_after}")

        #
        X1 = df_resampled.drop(columns=target, axis=1)
        y1 = df_resampled[target]
        if last_balance is True:
            sampling_strategy2 = "auto"
        rus = RandomUnderSampler(
            random_state=random_state, sampling_strategy=sampling_strategy2
        )
        X_resampled1, y_resampled1 = rus.fit_resample(X1, y1)
        df_resampled1 = pd.DataFrame(X_resampled1, columns=X1.columns)
        df_resampled1[target] = y_resampled1
        counts_after = y_resampled1.value_counts().to_frame()
        counts_after["%"] = counts_after.values / counts_after.values.sum()

        logger.info(
            f"Conteo de clases después del balanceo Subsample: {sampling_strategy2}"
        )
        logger.info(f"{counts_after}")

        df_resampled = df_resampled1.copy()
    else:
        logger.info("MAL PARAMETRIZADO")
    return df_resampled


# 0.1.1 Funcion Auxiliar para efectuar el balanceo
def balance_target_variable_pd_oscar(
    X_entrada: pd.DataFrame,
    y_objetivo: pd.DataFrame,
    group: Any,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Balancea la variable objetivo utilizando el método Synthetic Minority Over-sampling Technique (SMOTE) en dos DataFrames.

    Parameters
    ----------
    X_entrada : pd.DataFrame
        Primer DataFrame de Pandas que se balanceará o Inputs.

    y_objetivo : pd.DataFrame
        Segundo DataFrame de Pandas que contiene la variable de interes y/o los ids.
    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        pd.DataFrame
        Tupla con los DataFrames con la target balanceada.
    """
    logger.info("Iniciando el balanceo de la variable objetivo...")
    logger.info(f"Muestreo de Balanceo: {group}")
    # target = [params["target"]]
    target = (
        [params["target"]] if isinstance(params["target"], str) else params["target"]
    )
    try:
        df1_balanced = balance_osc(X_entrada, y_objetivo, params, group)
    except Exception:
        logger.info("FALLO")
        logger.info("Nulos:")
        logger.info(X_entrada.isnull().sum().replace(0, np.nan).dropna())
        logger.info(y_objetivo.isnull().sum().replace(0, np.nan).dropna())
        raise ValueError(
            "Procesamiento de datos nulos en variables numericas no parametrizado..."
        )
    finally:
        logger.info("Balanceo de la variable objetivo completado!")
        X_balance = df1_balanced.drop(target, axis=1)
        y_balance = df1_balanced[target]
        return X_balance, y_balance


# 1. Generacion de balanceos o Bucket Code Temporal
def Experimentacion_balanceos(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    scaler,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construye un pickle en formato diccionario que contiene un espacio o key por cada modelo que se ejeuctara junto con su base de datos balanceada.

    Parameters
    ----------
    X_train : pd.DataFrame
        Primer DataFrame de Pandas que se balanceará o Inputs.
    y_train : pd.DataFrame
        Segundo DataFrame de Pandas que contiene la variable de interes y/o los ids.
    X_test : pd.DataFrame
        Primer DataFrame de Pandas que se balanceará o Inputs.
    y_test : pd.DataFrame
        Segundo DataFrame de Pandas que contiene la variable de interes y/o los ids.
    scaler:
        Pickle con el procesamiento de datos
    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        Dict
            Espacio creado de experimentacion de modelos.
    """
    # parametros
    n_ciclos = params["numero_nodos"]
    lista_muestreos = params["muestreos"]
    names = params["names"]
    if n_ciclos > len(lista_muestreos):
        raise ValueError(
            f"Posible error despues del nodo {len(lista_muestreos)} por no parametrizar todos los muestreos requeridos al momento de balancear la data..."
        )
    else:
        info_save = {}
        # nodo en el que va entrenando le codigo
        info_save["nodo_run"] = 0

        # 0.0 Funcion Auxiliar para tratar la Y de interes y limpiar los
        logger.info("Procesando la data de Testeo..")
        X_i2, y_i2 = run_clean_x_tratamiento_y(X_train, y_train, params)
        X_i3, y_i3 = scaler.transform(X_test)
        X_i3, y_i3 = run_clean_x_tratamiento_y(X_i3, y_test, params)

        for i in range(int(n_ciclos)):
            group = lista_muestreos[i]
            logger.info(f"Algoritmo: {names[i]}")
            info_save[i] = {}
            # 0.1.1 Funcion Auxiliar para efectuar el balanceo
            X_balance, y_balance = balance_target_variable_pd_oscar(
                X_i2, y_i2, group, params
            )
            info_save[i]["Muestreo_tipo"] = {
                "Type": params["balance_target_variable"]["Muestreo"][group]["Type"],
                "sampling_strategy1": params["balance_target_variable"]["Muestreo"][
                    group
                ]["sampling_strategy_osc"],
                "sampling_strategy2": params["balance_target_variable"]["Muestreo"][
                    group
                ]["sampling_strategy_osc2"],
                "Use_auto": params["balance_target_variable"]["Muestreo"][group][
                    "Use_auto"
                ],
            }
            info_save[i]["X_balance"] = X_balance
            info_save[i]["y_balance"] = y_balance
            info_save[i]["X_train"] = X_i2
            info_save[i]["y_train"] = y_i2
            info_save[i]["X_test"] = X_i3
            info_save[i]["y_test"] = y_i3
            info_save[i]["Scaler"] = scaler
            info_save[i]["name_model"] = names[i]
            print("------------------------------")
    return info_save


def train_xgboost_with_cv(info_save: Dict[str, Any], params: Dict[str, Any]):
    """
    Train an XGBoost classifier using cross-validation and grid search with early stopping.

    Args:
        info_save (Dict): Contiene todo la data de entrenamiento balanceada y sin balancear.

        params (Dict[str, Any]): Parameters from Kedro's params.yml file.

    Returns:
        Dict with model training
    """
    logger.info("INICIANDO ENTRENAMIENTO DE XGBOOTS ...")
    i = info_save["nodo_run"]
    names_tag = info_save[i]["name_model"]

    nombre = "xgboost"
    info_save[i]["model_name"] = nombre + "_" + str(i)
    logger.info(
        f"Ejecutando el modelo: {i}. Nombre: {info_save[i]['model_name']}. NameTag: {names_tag}"
    )

    # Extract param_grid and cv_params from the params dictionary
    param_grid = params[nombre]["param_grid"]
    cv_params = params[nombre]["cv_params"]
    class_weight = params[nombre]["class_weight"]

    # extrayendo los datos
    X_balance = info_save[i]["X_balance"]
    y_balance = info_save[i]["y_balance"]
    X_i2 = info_save[i]["X_train"]
    y_i2 = info_save[i]["y_train"]

    if class_weight is True:
        aux = y_balance.value_counts().to_frame()
        w = float(aux.loc[0].iloc[0, 0])
        w = math.ceil(w / float(aux.loc[0].iloc[0, 0]))
        logger.info(f"W: {w}")
        param_grid["scale_pos_weight"] = list(set(param_grid["scale_pos_weight"] + [w]))

    logger.info(cv_params)
    logger.info(param_grid)

    # Convertir X_train y y_train balanceados a arrays de NumPy
    X_train = X_balance.values
    y_train = y_balance.values.ravel()

    # Convertir X_train y y_train sin balancear a arrays de NumPy
    X_i = X_i2.values
    y_i = y_i2.values.ravel()

    # Initialize the XGBClassifier with early stopping rounds
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=cv_params["n_jobs"],
        random_state=42,
        #        scale_pos_weight = class_weight["1.0"],
        early_stopping_rounds=cv_params.get("early_stopping_rounds", 10),
    )

    # Implement stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=cv_params["n_splits"], shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skf,
        scoring=cv_params["scoring"],
        n_jobs=cv_params["n_jobs"],
    )

    # Fit the model
    grid_search.fit(X_train, y_train, eval_set=[(X_i, y_i)], verbose=False)

    best_model = grid_search.best_estimator_

    # guardando modelo
    info_save[i]["best_model"] = best_model
    info_save[i]["grid_search"] = grid_search

    # nodo en el que va entrenando le codigo
    info_save["nodo_run"] = info_save["nodo_run"] + 1

    gc.collect()
    return info_save


def train_random_forest_with_cv(info_save: Dict[str, Any], params: Dict[str, Any]):
    """
    Train a Random Forest classifier using cross-validation and randomized search with Kedro parameters.

    Args:
        info_save (Dict): Contiene todo la data de entrenamiento balanceada y sin balancear.
        params (Dict[str, Any]): Parameters from Kedro's params.yml file.

    Returns:
        Dict with model training
    """
    logger.info("INICIANDO ENTRENAMIENTO DE RANDOM FOREST ...")
    i = info_save["nodo_run"]
    names_tag = info_save[i]["name_model"]

    nombre = "random_forest"
    info_save[i]["model_name"] = nombre + "_" + str(i)
    logger.info(
        f"Ejecutando el modelo: {i}. Nombre: {info_save[i]['model_name']}. NameTag: {names_tag}"
    )

    # Extract param_distributions and cv_params from the params dictionary
    param_distributions = params[nombre]["param_grid"]
    cv_params = params[nombre]["cv_params"]
    class_weight = params[nombre]["class_weight"]

    # extrayendo los datos
    X_balance = info_save[i]["X_balance"]
    y_balance = info_save[i]["y_balance"]
    # X_i2 = info_save[i]["X_train"]
    # y_i2 = info_save[i]["y_train"]

    if class_weight is True:
        aux = y_balance.value_counts().to_frame()
        w = float(aux.loc[0].iloc[0, 0])
        w = math.ceil(w / float(aux.loc[0].iloc[0, 0]))
        logger.info(f"W: {w}")
        if {0.0: 1, 1.0: w} not in param_distributions["class_weight"]:
            param_distributions["class_weight"].append({0.0: 1, 1.0: w})

    logger.info(cv_params)
    logger.info(param_distributions)

    # Convertir X_train y y_train a arrays de NumPy
    X_train = X_balance.values
    y_train = y_balance.values.ravel()

    # Convertir X_train y y_train sin balancear a arrays de NumPy
    # X_i = X_i2.values
    # y_i = y_i2.values.ravel()

    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(
        random_state=42, n_jobs=cv_params["n_jobs"], verbose=1
    )

    # Implement stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=cv_params["n_splits"], shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=cv_params.get("n_iter", 100),
        cv=skf,
        scoring=cv_params["scoring"],
        n_jobs=cv_params["n_jobs"],
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # guardando modelo
    info_save[i]["best_model"] = best_model
    info_save[i]["grid_search"] = random_search

    # nodo en el que va entrenando le codigo
    info_save["nodo_run"] = info_save["nodo_run"] + 1
    gc.collect()
    return info_save


# red neuronal
def red_neuronal(info_save, params):
    """
    Train a MLP classifier using cross-validation and randomized search with Kedro parameters.

    Args:
        info_save (Dict): Contiene todo la data de entrenamiento balanceada y sin balancear.
        params (Dict[str, Any]): Parameters from Kedro's params.yml file.

    Returns:
        Dict with model training
    """
    logger.info("INICIANDO ENTRENAMIENTO DE MLP CLASSIFIER ...")
    i = info_save["nodo_run"]
    names_tag = info_save[i]["name_model"]

    nombre = "MLP"
    info_save[i]["model_name"] = nombre + "_" + str(i)
    logger.info(
        f"Ejecutando el modelo: {i}. Nombre: {info_save[i]['model_name']}. NameTag: {names_tag}"
    )

    # Extract param_distributions and cv_params from the params dictionary
    # ajustes de parametros
    param_grid = params[nombre]["param_grid"]
    val_all = []
    remove = []
    for b in param_grid.keys():
        llave = "_".join(b.split("_")[:-1])
        if (llave == "hidden_layer_sizes") | (b == "hidden_layer_sizes"):
            remove.append(b)
            val = param_grid[b]
            # val = [if len(j) != 0 tuple(j) for j in val]
            val = [tuple(j) for j in val if len(j) != 0]
            val_all = list(set(val_all + val))

    for b in remove:
        del param_grid[b]
    param_grid["hidden_layer_sizes"] = val_all
    logger.info(param_grid)

    # otros parametros
    cv_params = params[nombre]["cv_params"]
    class_weight = params[nombre]["class_weight"]
    max_iter = params[nombre]["max_iter"]

    # extrayendo los datos
    X_balance = info_save[i]["X_balance"]
    y_balance = info_save[i]["y_balance"]
    # X_i2 = info_save[i]["X_train"]
    # y_i2 = info_save[i]["y_train"]

    # Datos
    X_balance_adj = X_balance.values
    y_balance_adj = y_balance.values.ravel()

    if class_weight is True:
        aux = y_balance.value_counts().to_frame()
        w = float(aux.loc[0].iloc[0, 0])
        w = math.ceil(w / float(aux.loc[0].iloc[0, 0]))
        logger.info(f"W: {w}")
        param_grid["alpha"] = list(set(param_grid["alpha"] + [w]))

    logger.info(cv_params)

    mlp = MLPClassifier(max_iter=max_iter, random_state=42)
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=cv_params["n_splits"],
        n_jobs=cv_params["n_jobs"],
        verbose=1,
        scoring=cv_params["scoring"],
    )
    grid_search.fit(X_balance_adj, y_balance_adj)
    best_model = grid_search.best_estimator_

    # guardando modelo
    info_save[i]["best_model"] = best_model
    info_save[i]["grid_search"] = grid_search

    # nodo en el que va entrenando le codigo
    info_save["nodo_run"] = info_save["nodo_run"] + 1
    gc.collect()
    return info_save


# Funcion auxiliar para grafico de metricas:
def plot_cv(results_df, model_name_list, params):
    """
    It seeks to graphically show the result of the behavior of 1 algorithm trained with a grid

    Args:
        results_df: Paquete completo de resultados de un Dataframe proveniente de GridSearch

        model_name_list: Lista con el nombre para identiricar el modelo

        params: Catalogo

    Returns:
        Plots.
    """
    if_run = params["plot"]["desea_plot"]
    want_ls = params["plot"]["grilla"]
    if if_run is True:
        model_name, apodo = model_name_list
        model_name = "_".join(model_name.split("_")[:-1])
        apodo
        x_labels = []
        y_splits = []
        yyy = ["mean_test_score"]
        for col in results_df.columns:
            filt = col.split("_")
            if filt[0] == "param":
                if "_".join(filt[1:]) in want_ls:
                    x_labels.append(col)
            if len(filt) > 1:
                if (filt[0][:5] == "split") & (filt[1] == "test"):
                    y_splits.append(col)
        for i in x_labels:
            dont_run = if_run
            print(i)
            # print(plt.get_backend())
            plt.figure(figsize=(10, 2))
            try:
                plt.scatter(
                    results_df[i], results_df[yyy], label="mean", marker="*", s=100
                )
                for ii in y_splits:
                    plt.scatter(
                        results_df[i], results_df[ii], label=ii, marker="D", s=5
                    )

                plt.title(
                    "Grid Search CV test score during learning: '"
                    + apodo
                    + "' o '"
                    + model_name_list[0]
                    + "' on balanced data"
                )
                plt.xlabel(i)
                plt.ylabel(params[model_name]["cv_params"]["scoring"] + " en CV")
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                plt.tight_layout()
                plt.show()
                plt.close("all")
            except Exception:
                if all([isinstance(loop, dict) for loop in results_df[i]]) is True:
                    if model_name == "random_forest":
                        results_df[i] = pd.DataFrame(results_df[i].tolist())[1.0]
                    else:
                        logger.info(
                            f"Revisar cual key() de {i} se desea plotear entre los parametros de la grilla para el algoritmo '{apodo}' o '{model_name_list[0]}'"
                        )
                        logger.info("Luego programarlo en la funcion 'plot_cv' ...")
                        dont_run = False
                if i == "param_hidden_layer_sizes":
                    # i = "param_hidden_layer_sizes"
                    one_capa = np.array(
                        [isinstance(capa, (int, float)) for capa in results_df[i]]
                    )
                    deep_capa = np.array(
                        [isinstance(capa, (tuple)) for capa in results_df[i]]
                    )
                    index_pos = results_df[i][one_capa].sort_values().index.tolist()
                    index_pos2 = results_df[i][deep_capa].sort_values().index.tolist()
                    index_pos = index_pos + index_pos2
                    # results_df2 = results_df.copy()
                    results_df = results_df.loc[index_pos]
                    results_df[i] = results_df[i].astype(str)
                if dont_run is True:
                    print(i)
                    plt.scatter(
                        results_df[i], results_df[yyy], label="mean", marker="*", s=100
                    )
                    for ii in y_splits:
                        plt.scatter(
                            results_df[i], results_df[ii], label=ii, marker="D", s=5
                        )
                    plt.title(
                        "Grid Search CV test score during learning: '"
                        + apodo
                        + "' o '"
                        + model_name_list[0]
                        + "' on balanced data"
                    )
                    plt.xlabel(i)
                    plt.ylabel(
                        params[model_name]["cv_params"]["scoring"] + " : Grid Search CV"
                    )
                    plt.legend(
                        loc="upper left", bbox_to_anchor=(1, 1), title="balanced data"
                    )
                    plt.tight_layout()
                    plt.show()
                    plt.close("all")


# Funcion auxiliar para calculo de metricas:
def calc_metrics(Y_, Y_pred_, model_name, dataset_name, params):
    """
    Funcion Auxiliar para calcular las metricas de desempeño del modelo

    params
    -----------------------
        y__: observaciones reales
        y_pred_ : Predicciones del modelo
        model_name : Nombre del modelo
        dataset_name: Nombre de la data
        params:
            Dict

    return
    -----------------------
        pd.DataFrame con las metricas tecnicas del modelo
    """
    # Compute confusion matrix components
    y_pred, y = tratamiento_y(Y_pred_, Y_, params)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate overall metrics
    # Calcular las métricas generales
    overall_metrics = {
        "accuracy": 100 * ((tp + tn) / (tp + tn + fp + fn)),
        "precision": 100 * precision_score(y, y_pred),
        "recall": 100 * recall_score(y, y_pred),
        "f1-score": 100 * f1_score(y, y_pred),
        "True positives": tp,
        "False negatives": fn,
        "True negatives": tn,
        "False positives": fp,
        "roc_auc": 100 * roc_auc_score(y, y_pred),
        "cohen_kappa": 100 * cohen_kappa_score(y, y_pred),
        "matthews_corrcoef": 100 * matthews_corrcoef(y, y_pred),
    }
    # Formatear los valores numéricos a 4 decimales sin convertir a string
    overall_metrics = {
        k: round(v, 4) if isinstance(v, (float, np.float64)) else v
        for k, v in overall_metrics.items()
    }

    # Convert overall metrics to DataFrame
    overall_metrics_df = pd.DataFrame(
        list(overall_metrics.items()), columns=["metric_name", "value"]
    )
    overall_metrics_df["model_name"] = model_name
    overall_metrics_df["dataset_name"] = dataset_name
    overall_metrics_df["metric_type"] = "overall metric"
    overall_metrics_df["class_name"] = "overall metric"
    # Generate the classification report and convert it to a DataFrame
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Filter the report to keep only per-class metrics
    report_df = report_df.drop(
        ["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )

    # Reshape the report DataFrame
    report_df = report_df.reset_index().melt(
        id_vars=["index"], var_name="metric_name", value_name="value"
    )
    report_df = report_df.rename(columns={"index": "class_name"})

    # Add the model name, dataset name, and metric type to the report DataFrame
    report_df["model_name"] = model_name
    report_df["dataset_name"] = dataset_name
    report_df["metric_type"] = "class metric"
    # Valores de 'metric_name' que requieren transformación
    metric_names_to_transform = {
        "accuracy",
        "precision",
        "recall",
        "f1-score",
        "roc_auc",
        "cohen_kappa",
        "matthews_corrcoef",
    }

    # Aplicar la transformación
    report_df.loc[report_df["metric_name"].isin(metric_names_to_transform), "value"] = (
        report_df.loc[
            report_df["metric_name"].isin(metric_names_to_transform), "value"
        ].astype(float)
        * 100
    ).round(4)

    # Combine overall metrics and class metrics into a single DataFrame
    combined_df = pd.concat(
        [
            overall_metrics_df,
            report_df[
                [
                    "model_name",
                    "dataset_name",
                    "metric_name",
                    "metric_type",
                    "class_name",
                    "value",
                ]
            ],
        ],
        ignore_index=True,
    )
    gc.collect()
    return combined_df


# Penultimo nodo:
def evaluate_models_for_all(info_save, params):  # -> pd.DataFrame
    """
    Evaluate multiple models on a balanced data set and their grid behavior produces a single DataFrame with metrics.

    Args:
        info_save: Paquete completo de modelos y bases de datos

        params: Catalogo

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for all models on both datasets.
    """
    logger.info("Iniciando la evaluacion de modelos...")
    all_results = pd.DataFrame()
    for i in range(info_save["nodo_run"]):
        model_name = info_save[i]["model_name"]
        name_tag = info_save[i]["name_model"]
        model = info_save[i]["best_model"]
        grid_search = info_save[i]["grid_search"]

        results_df = pd.DataFrame(grid_search.cv_results_)
        logger.info(f"Iniciando la evaluacion de '{model_name}' o '{name_tag}' ...")
        plot_cv(results_df, [model_name, name_tag], params)
        for dataset_name in ["balance", "train", "test"]:
            try:
                X = info_save[i]["X_" + dataset_name]
                y = info_save[i]["y_" + dataset_name]
                logger.info(
                    f"Evaluacion de metricas para '{model_name}' o '{name_tag}' en {dataset_name}..."
                )
                y_pred = model.predict(X.values)  # la red neuronal recibe un array
                y_pred = pd.DataFrame(y_pred, index=y.index, columns=y.columns)
                combined_df = calc_metrics(
                    y, y_pred, name_tag + "," + model_name, dataset_name, params
                )
                combined_df = combined_df[
                    [
                        "model_name",
                        "dataset_name",
                        "metric_name",
                        "metric_type",
                        "class_name",
                        "value",
                    ]
                ]
                # display(combined_df[combined_df["class_name"].isin(params["compare_metrics_models"]["class_name"])])
                all_results = pd.concat([all_results, combined_df], axis=0)
                info_save["all_results"] = all_results
            except Exception:
                pass

    gc.collect()
    return info_save


# ultimo nodo
def compare_metrics_models(info_save, params: dict):  # -> pd.DataFrame
    """
    Evaluate multiple models on a balanced data set and their grid behavior produces a single DataFrame with metrics.

    Args:
        all_results: pd.DataFrame que contiene las metricas de data balanceada de entrenamiento, entrenamiento, etc...

        params: Catalogo

    Returns:
        pd.DataFrame: DataFrame que contiene los resultados de los mejores algoritmos...
    """
    label_target = params["compare_metrics_models"]["class_name"]
    dataset_target = params["compare_metrics_models"]["dataset_name"]
    metric_target = params["compare_metrics_models"]["metric_name"]
    all_results = info_save["all_results"]
    all_results["value"] = all_results["value"].astype(float)
    logger.info(
        "Iniciando la comparacion de metricas sobre las datas generadas en Model_input: "
    )
    metrics = ["cohen_kappa", "roc_auc"]
    df_filtered = all_results[all_results["metric_name"].isin(metrics)]
    msg = list(df_filtered["metric_type"].unique())
    logger.info(f"Graficos de desempeños en {metrics} calculadas tipo: {msg}")
    df_pivot = df_filtered.pivot(
        index=("model_name", "dataset_name"), columns="metric_name", values="value"
    ).reset_index()
    # Crear el scatter plot
    plt.figure(figsize=(15, 10))
    sns.scatterplot(
        data=df_pivot,
        x="cohen_kappa",
        y="roc_auc",
        hue="model_name",
        palette="viridis",
        style="dataset_name",
        s=50,
    )

    # ejes
    plt.xlim(0, 100)
    plt.xticks(
        ticks=[10 * i for i in range(11)], labels=[f"{10 * i:.1f}" for i in range(11)]
    )
    plt.ylim(0, 100)
    plt.yticks(
        ticks=[10 * i for i in range(11)], labels=[f"{10 * i:.1f}" for i in range(11)]
    )

    # Ajustar el título y etiquetas de los ejes
    plt.title("Scatter Plot of Kappa vs Roc_auc")
    plt.xlabel("cohen_kappa")
    plt.ylabel("roc_auc")

    # Ajustar el diseño del gráfico
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Model Name")
    plt.tight_layout()
    plt.show()
    plt.close("all")

    # grafico sobre la metrica de interes en la clase de interes:
    logger.info(
        f"Graficos de desempeños en {label_target} calculadas sobre y {metric_target}"
    )

    filt = all_results[
        all_results["metric_name"].isin(
            [
                "accuracy",
                "cohen_kappa",
                "precision",
                "recall",
                "f1-score",
                "roc_auc",
                "matthews_corrcoef",
            ]
        )
    ]
    filt = filt[filt["class_name"].isin(label_target)]

    drop_model = list(filt[filt["value"] < 1]["model_name"].unique())
    if len(drop_model) >= 1:
        logger.info(f"Algoritmos con Metricas = 0: {drop_model}")
    # quitando 0:
    filt = filt[filt["value"] != 0]

    # Pivotar los datos para el gráfico de barras apiladas
    df_pivot = filt.pivot(
        index=("dataset_name", "model_name"), columns="metric_name", values="value"
    )
    # Crear el gráfico de barras apiladas
    ax = df_pivot.plot(kind="barh", stacked=False, figsize=(15, 15), colormap="viridis")

    plt.title(f"Class Metrics:{metric_target} .'y={label_target}'.")
    plt.ylabel("Metric Name")
    plt.xlabel("Value")
    # Establecer los límites del eje X desde 0 a 1
    plt.xlim(0, 100)

    # Establecer los ticks del eje X en intervalos de 0.1
    plt.xticks(
        ticks=[10 * i for i in range(11)], labels=[f"{10 * i:.1f}" for i in range(11)]
    )
    plt.yticks(rotation=45, fontsize=8)

    plt.legend(loc="best", bbox_to_anchor=(1, 1), title="Model Name")
    # Agregar etiquetas de valores sobre las barras
    for p in ax.patches:
        width = p.get_width()
        plt.text(
            width + 1,
            p.get_y() + p.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )
    plt.tight_layout()
    plt.show()
    plt.close("all")

    # reodena los indices
    filt.index = list(range(filt.shape[0]))
    # Agrupar por 'metric_name' y 'dataset_name' y encontrar el valor máximo en cada grupo
    max_values = filt.loc[
        filt.groupby(["dataset_name", "metric_name"])["value"].idxmax()
    ]
    max_values1 = max_values[max_values["dataset_name"] != "balance"]
    logger.info("Mejores algoritmos por metrica y dataset: ")
    display(max_values1)
    selec_model = max_values1[max_values1["dataset_name"] == dataset_target]
    selec_model = selec_model[selec_model["metric_name"] == metric_target]
    selec_model.index = ["best_model_name"]
    selec_model = selec_model.T
    selec_model.loc[dataset_target] = selec_model.loc["value", "best_model_name"]
    selec_model = selec_model.drop(["value", "dataset_name"], axis=0)
    llave1 = filt[filt["metric_name"] == metric_target]
    llave1 = llave1.pivot(index="dataset_name", columns="model_name", values="value")
    bias = (
        (llave1.loc["train"] - llave1.loc["balance"])
        .sort_index()
        .to_frame()
        .rename(columns={0: "bias"})
    )
    bias["real_bias"] = bias["bias"] < 0

    selec_model["best_unbiased_model"] = None
    selec_model.loc["model_name", "best_unbiased_model"] = bias["bias"].idxmax()

    updatting = ["best_model_name", "best_unbiased_model"]

    test_exist = False
    if any(filt["dataset_name"] == "test") is True:
        updatting = updatting + ["best_fitting_model"]
        test_exist = True

        overfitting = (
            (llave1.loc["test"] - llave1.loc["train"])
            .sort_index()
            .to_frame()
            .rename(columns={0: "fitting"})
        )
        overfitting["type"] = (
            (overfitting["fitting"] < 0)
            .replace(True, "overfitting")
            .replace(False, "Underfitting")
        )

        selec_model["best_fitting_model"] = None
        selec_model.loc["model_name", "best_fitting_model"] = (
            overfitting["fitting"].abs().idxmin()
        )

    for key in updatting:
        if key != "best_model_name":
            selec_model.loc["metric_name", key] = metric_target
            selec_model.loc["metric_type", key] = selec_model.loc[
                "metric_type", "best_model_name"
            ]
            selec_model.loc["class_name", key] = selec_model.loc[
                "class_name", "best_model_name"
            ]

        llave = filt[filt["model_name"] == selec_model.loc["model_name", key]].copy()
        llave2 = llave[
            llave["metric_name"] == selec_model.loc["metric_name", key]
        ].copy()
        for i in llave2["dataset_name"].unique():
            selec_model.loc[i, key] = llave2[llave2["dataset_name"] == i]["value"].iloc[
                0
            ]

        selec_model.loc["bias", key] = bias.loc[
            selec_model.loc["model_name", key], "bias"
        ]
        selec_model.loc["real_bias", key] = bias.loc[
            selec_model.loc["model_name", key], "real_bias"
        ]

        if test_exist is True:
            selec_model.loc["test-train", key] = overfitting.loc[
                selec_model.loc["model_name", key], "fitting"
            ]
            selec_model.loc["fitting", key] = overfitting.loc[
                selec_model.loc["model_name", key], "type"
            ]
    logger.info("Mejores algoritmos: ")
    display(selec_model)
    info_save["best_model_metrics_dataset"] = max_values1
    info_save["select_model"] = selec_model

    # Ajusatr las cifras para que sean strings
    all_results["value"] = all_results["value"].apply(lambda x: f"{x:.4f}")
    info_save["all_results"] = all_results
    return info_save
