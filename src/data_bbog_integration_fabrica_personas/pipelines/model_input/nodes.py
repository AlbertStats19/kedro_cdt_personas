"""
Nodos de la capa model_input
"""
from typing import Dict, Any, Tuple, Union

import pandas as pd
import numpy as np
import logging
import gc

# from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# # 0. filtrando las cataracteristicas de interes:
def feature_selec_pd(
    df1: pd.DataFrame, df2: [Any], params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Toma la base de datos completa y filtra las variables que se desean del DataFrame

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se filtrara.

    df2: [Any]
        Contiene las columans que se desean en el modelo.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
         DataFrame con las columnas deseadas.
    """
    logger.info("Iniciando la filtracion de feature relevantes ...")
    target = [params["target"]]
    keys = [params["id"]]
    filtro = list(set(list(df2) + target + keys))
    df1_filt = df1[df1.columns[df1.columns.isin(filtro)]]
    return df1_filt


# 1. separacion de data
def split(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = [params["target"]]
    ids = [params["id"]]
    test_size = params["train_test_split"]["test_size"]
    random_state = params["train_test_split"]["random_state"]
    shuffle = params["train_test_split"]["shuffle"]
    X = df.drop(columns=target, axis=1)
    y = df[target]
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=shuffle,
        )
        logger.info("Ok!!")
    except Exception:
        try:
            logger.info("Imputando Nulos temporalmente")
            X1 = X.copy()
            yy = y.copy()
            dropping = pd.DataFrame(index=X.index)
            for col in target + ids:
                try:
                    X1.drop(col, axis=1, inplace=True)
                    dropping = pd.concat([dropping, X[col]], axis=1)
                except Exception:
                    pass
            df_numeric = X.select_dtypes(include=["number"])
            # df_categorical = X.select_dtypes(include=["object", "category", "bool"])

            imputer = SimpleImputer(strategy="mean")
            X2 = imputer.fit_transform(df_numeric)

            imputer2 = SimpleImputer(strategy="most_frequent")

            # X3 = imputer2.fit_transform(df_categorical)
            X4 = pd.concat([X1, X2], axis=1)

            y1 = imputer2.fit_transform(y)
            X_train_val1, X_test1, y_train_val1, y_test1 = train_test_split(
                X4,
                y1,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
                shuffle=shuffle,
            )
            X_train_val = X.loc[X_train_val1.index.tolist()]
            X_test = X.drop(X_train_val1.index.tolist(), axis=0)

            y_train_val = yy.loc[y_train_val1.index.tolist()]
            y_test = yy.drop(y_train_val1.index.tolist(), axis=0)
            logger.info("Ok!!")
            gc.collect()
        except Exception:
            logger.info("Split Manual")
            # Fijar la semilla
            np.random.seed(random_state)

            lista = X.index.tolist()
            replace = False  # Valores unicos
            high = int(len(lista) * (1 - test_size))  # Tamaño del array
            # array de números enteros aleatorios unicos
            random_array = np.random.choice(lista, high, replace)
            X_train_val = X.loc[random_array]
            y_train_val = y.loc[random_array]
            X_test = X.drop(random_array, axis=0)
            y_test = y.drop(random_array, axis=0)
        finally:
            pass
    finally:
        logger.info("Data completa")
        logger.info(y.value_counts())
        logger.info("Data de entrenamiento")
        logger.info(y_train_val.value_counts())
        logger.info("Data de testeo")
        logger.info(y_test.value_counts())

    # train_data = pd.concat([X_train_val, y_train_val], axis=1)
    # test_data = pd.concat([X_test, y_test], axis=1)
    return X_train_val, y_train_val, X_test, y_test


def train_test_split_pd(
    df1: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide dos DataFrames en tres subconjuntos: entrenamiento, validación y
    prueba.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se dividirá.
    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tupla con los DataFrames de entrenamiento-validacion y prueba para X y Y
        con ambos DataFrames de entrada.
    """

    logger.info("Iniciando la división de datos en entrenamiento y prueba...")

    X_train_val, y_train_val, X_test, y_test = split(df1, params)

    logger.info("División de datos en entrenamiento-validación y prueba completada!")

    return X_train_val, y_train_val, X_test, y_test


# 2. Tratamiento de datos nulos en numeros
def treatment_null_numbers(
    df1: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Union[SimpleImputer, IterativeImputer], Any]:
    """
    Procesa los datos numericos que son nulos.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para entrenar y fitear modelo
        para tratar nulos de variables numericas. (Data Entrenamiento)

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    object
        El modelo que contiene el procesamiento de nulos numericos.
    """

    logger.info("Iniciando el fitting sobre los datos numericos nulos")
    process_calc = params["null_adj"]["numerical"]
    process = params["null_adj"]["method_numerical"]
    random_state = params["null_adj"]["random_state"]

    logger.info("Iniciando el tratamiento de datos nulos en variables numericas...")
    target = [params["target"]]
    ids = [params["id"]]
    try:
        X = df1.drop(target + ids, axis=1)
    except Exception:
        X = df1.copy()
        for col in target + ids:
            try:
                X.drop(col, axis=1, inplace=True)
            except Exception:
                pass
    finally:
        df_numeric = X.select_dtypes(include=["number"])

    if process == "Simple":
        imputer = SimpleImputer(
            strategy=process_calc
        )  # 'median', 'most_frequent', y 'constant'
    elif process == "Interactive":
        imputer = IterativeImputer(
            random_state=random_state, initial_strategy=process_calc
        )  # mean
    else:
        logger.info(
            "Procesamiento de datos nulos en variables numericas no parametrizado..."
        )
        raise ValueError(
            "Procesamiento de datos nulos en variables numericas no parametrizado..."
        )
    order_col = list(df_numeric.columns)
    imputer.fit(df_numeric)
    # Forzar la recolección de basura después de cada iteración para liberar memoria
    gc.collect()
    return imputer, order_col


# 3. Identificacion de Outliers
def treatment_outliers(df1: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Procesa e identifica la densidad de datos Outliers para solucionar su
    transformacion.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para identificar variables
        numericas Atipicas. (Data Entrenamiento)

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    pd.DataFrame
        DataFrame que contendra la variable numerica en cada fila y su metodo
        con el que se transformaria la variable para hacer correcion de outliers.
    """

    logger.info(
        "Iniciando la identificacion de outliers y "
        "definiendo como se transformaran los datos"
    )

    Hacer_transformacion = params["Outliers"]["Hacer_transformacion"]
    Sugerir_procesamiento = params["Outliers"]["Sugerir_procesamiento"]
    method = params["Outliers"]["method"]
    umbral = params["Outliers"]["umbral"]

    ids = [params["id"]]
    target = [params["target"]]
    try:
        X = df1.drop(target + ids, axis=1)
    except Exception:
        X = df1.copy()
        for col in target + ids:
            try:
                X.drop(col, axis=1, inplace=True)
            except Exception:
                pass
    finally:
        X = X.select_dtypes(include=["number"])

    logger.info("Detectando variables altamente dispersas ...")
    method_calc = pd.DataFrame(columns=["Method"])
    if Hacer_transformacion is True:
        if method == "IQR":
            X1 = X.quantile(
                q=[0.25, 0.50, 0.75], axis=0, numeric_only=True, interpolation="linear"
            )
            X1.loc["IQR"] = X1.loc[0.75] - X1.loc[0.25]
            X1.loc["Up"] = X1.loc[0.75] + 1.5 * X1.loc["IQR"]
            X1.loc["Down"] = X1.loc[0.25] - 1.5 * X1.loc["IQR"]
        elif method == "Z-Score":
            X1 = X.mean().to_frame().T
            X2 = X.std().to_frame().T
            X1.index = ["Mean"]
            X2.index = ["Sd"]
            X1 = pd.concat([X1, X2], axis=0)
            X1.loc["Up"] = X1.loc["Mean"] + X1.loc["Sd"] * 3
            X1.loc["Down"] = X1.loc["Mean"] - X1.loc["Sd"] * 3
        else:
            logger.info("Metodo de deteccion de Outliers no parametrizado...")
            raise ValueError("Metodo de deteccion de Outliers no parametrizado...")

        for col in X.columns:
            n = (X[col] >= X1.loc["Up", col]).sum()
            n = n + (X[col] <= X1.loc["Down", col]).sum()
            n = n / X.shape[0]
            if n >= umbral and Sugerir_procesamiento is True:
                if X[col].min() > 0:
                    method_calc.loc[col, "Method"] = "log"
                elif X[col].min() >= 0:
                    method_calc.loc[col, "Method"] = "sqrt"
                else:
                    method_calc.loc[col, "Method"] = "sqrt3"
            elif n >= umbral:
                method_calc.loc[col, "Method"] = "sqrt3"
            else:
                pass
    method_calc.index.name = "columns"
    logger.info("Transformacion de nulos: ")
    logger.info(method_calc["Method"].unique())
    gc.collect()
    return method_calc


# 3.1 Transformacion de Outliers
def adj_outliers(
    df1: pd.DataFrame, metodologia: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Procesa o transforma los datos segun una metodologia dada para tratar
    Outliers.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para identificar variables
        numericas Atipicas. (Data Entrenamiento)

    metologia: pd.DatFrame
        DataFrame en fomrato CSV que contiene las columnas que deben ser
        transformadas y el metodo de transformacion

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    pd.DataFrame
        El DataFrame con la transformacion de las variables que lo requieran.
    """

    logger.info("Comenzado a transformar los datos")
    target = [params["target"]]
    ids = [params["id"]]
    try:
        X = df1.drop(target + ids, axis=1)
    except Exception:
        X = df1.copy()
        for col in target + ids:
            try:
                X.drop(col, axis=1, inplace=True)
            except Exception:
                pass
    finally:
        X = X.select_dtypes(include=["number"])

    try:
        metodologia.set_index("columns", inplace=True)
    except Exception:
        pass

    if metodologia.shape[0] == 0:
        pass
    else:
        for meth in list(metodologia["Method"].unique()):
            col_list = list(metodologia[metodologia["Method"] == meth].index)
            if meth == "log":
                X[col_list] = X[col_list].apply(np.log).replace([np.nan, -np.inf], 0)
            elif meth == "sqrt3":
                X[col_list] = X[col_list].apply(np.cbrt)
            elif meth == "sqrt":
                X[col_list] = X[col_list].apply(np.sqrt).replace([np.nan, -np.inf], 0)
            else:
                logger.info(
                    "Metodo de transformacion de Datos Outliers no parametrizado..."
                )
                raise ValueError(
                    "Metodo de transformacion de Datos Outliers no parametrizado..."
                )
    gc.collect()
    return X


# 4. Limpiar la Data Numerica antes de procesar la data categorica, string o boolerana
def run_numeric_values(
    df1: pd.DataFrame,
    order_col_numeric: Any,
    imputer_model_numeric,
    metodologia,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Procesa o transforma los datos segun los pipelines anteriores.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame de Pandas que se usara para modelar.

    imputer_model_numeric:
        Modelo que contiene la forma de imputacion de datos nulos sobre las
        variables numericas.

    metologia: pd.DatFrame
        DataFrame en fomrato CSV que contiene las columnas que deben ser
        transformadas y el metodo de transformacion.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    pd.DataFrame
        Un DataFrame en formato parquet que contiene todas las variables
        originales y procesa las variables numericas (Nulos y Outliers).
    """

    logger.info(
        "Iniciando con el tratamiento de datos nulos "
        "en variables numericas y tratando los outliers"
    )
    target = [params["target"]]
    ids = [params["id"]]
    # process = params["null_adj"]["method_numerical"]
    try:
        # guardando la variable de interes
        y_feature_num = df1[target + ids]
        # sacamos la variable de interes del procesamiento de datos
        X_all = df1.drop(target + ids, axis=1)
    except Exception:
        X_all = df1.copy()
        y_feature_num = pd.DataFrame(index=df1.index)
        for col in target + ids:
            try:
                X_all.drop(col, axis=1, inplace=True)
                y_feature_num = pd.concat([y_feature_num, df1[col]], axis=1)
            except Exception:
                pass

    # filtramos las variables categoricas porque no la vamos a ejecutar
    X_categorical = X_all.select_dtypes(include=["object", "category", "bool"])
    # iniciamos el procesamiento de datos numericos
    X_numeric = X_all.select_dtypes(include=["number"])
    X_numeric = X_numeric[order_col_numeric]
    # llenamos los datos nulos
    logger.info("Comenzando a llenar los datos nulos")
    master_feature_num = imputer_model_numeric.transform(X_numeric)
    logger.info("Datos Nulos rellenados.")

    master_feature_num = pd.DataFrame(
        master_feature_num, columns=X_numeric.columns.tolist(), index=X_numeric.index
    )
    # trabajamos los datos nulos
    logger.info("Comenzando a transformar las variables con outliers")
    master_feature_num_O = adj_outliers(master_feature_num, metodologia, params)
    try:
        all_data_proce = pd.concat(
            [master_feature_num_O, X_categorical, y_feature_num], axis=1
        )
    except Exception:
        all_data_proce = pd.concat([master_feature_num_O, X_categorical], axis=1)
    finally:
        all_data_proce = all_data_proce[df1.columns]
    logger.info("Actualizando los datos nulos y outliers en variables numericas")
    return all_data_proce


# 5. Tratamiento de datos nulos en categorias o strings o booleanos
class aux_imputer_categ:
    """
    Clase que guarda un modelos por variable.

    param:
        df: DataFrame que contiene todas las variables de los datos con que se
            entrenara el modelo.
        params: Parametrizacion de la infraestructura.
    """

    def __init__(self, df, params):
        self.df = df.copy()  # evita la eliminacion de columnas
        self.models = {}
        self.metrics = {}
        self.target_column = [params["target"]]
        self.keys = [params["id"]]
        self.random_state = params["null_adj"]["random_state"]
        self.metric_min = params["null_adj"]["min_mectric"]
        self.type_model = params["null_adj"]["model_interactive_categorical"]

    def _identify_categorical_variables(self, data):
        """
        Identifica las variables categóricas y booleanas de un
        DataFrame dado que son diferentes a la variable de interes.
        """
        # identificar as varibales inputs para
        numeric_inputs = data.select_dtypes(include=["number"]).columns.tolist()
        for col in self.keys + self.target_column:
            if col in numeric_inputs:
                print("Removing inputs training in ", col)
                numeric_inputs.remove(col)

        # indentificando las variables sobre las cuales
        # se realizara el procesamiento de trtar nulos
        categorical_targets = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        for col in self.keys + self.target_column:
            if col in categorical_targets:
                print("Removing output training in ", col)
                categorical_targets.remove(col)
        return numeric_inputs, categorical_targets

    def save_identify_categorical_variables(self):
        """
        Guarda la identificacion las variables categóricas
        y booleanas del DataFrame usado para entrenar
        """
        # identificar as varibales inputs para

        (
            self.numeric_inputs,
            self.categorical_columns,
        ) = self._identify_categorical_variables(self.df)

    def fit(self, names_col):
        """
        Entrena un modelo para cada columna categórica.
        """
        # actualizamos cuales son las variables numericas
        # (inputs) y cada variable categoricas
        self.save_identify_categorical_variables()
        X = self.df[self.numeric_inputs]
        feature_names_in_ = []
        for col in names_col:
            if col in self.categorical_columns:
                print(f"Iniciando Modelo para {col}")
                if self.type_model == "RandomForest":
                    model = RandomForestClassifier(
                        random_state=self.random_state, n_jobs=os.cpu_count() - 1
                    )
                    model1 = LogisticRegression(random_state=self.random_state)
                elif self.type_model == "LogisticRegression":
                    model = LogisticRegression(random_state=self.random_state)
                    model1 = RandomForestClassifier(
                        random_state=self.random_state, n_jobs=os.cpu_count() - 1
                    )
                else:
                    ValueError(
                        "Procesamiento de datos nulos en categorias con el "
                        "metodo interactivo no parametrizado..."
                    )
                index_not_null = self.df[col].notna()
                Y = self.df.loc[index_not_null, col]
                model.fit(X.loc[index_not_null], Y)
                y_pred = model.predict(X.loc[index_not_null])
                self.models[col] = model
                self.metrics[col] = accuracy_score(Y, y_pred)
                feature_names_in_.append(col)
                if self.metrics[col] < self.metric_min:
                    try:
                        model1.fit(X.loc[index_not_null], Y)
                        y_pred = model1.predict(X.loc[index_not_null])
                        acc = accuracy_score(Y, y_pred)
                        if acc > self.metrics[col]:
                            self.metrics[col] = acc
                            self.models[col] = model1
                    except Exception:
                        print("Ajuste no funciono")
                print(
                    f"Modelo para {col} - Accuracy: {self.metrics[col]:.2f}."
                    f" Metrica Min Requerida: {self.metric_min}"
                )
                gc.collect()
        self.feature_names_in_ = feature_names_in_

    def transform(self, X):
        """
        Predice los nulos usando los modelos entrenados.
        :param X: DataFrame con toda la data.
        :return: Diccionario con predicciones para cada modelo.
        """
        (
            self.numeric_inputs_pred,
            self.categorical_columns_pred,
        ) = self._identify_categorical_variables(X)
        # predictions = {}
        X_encoded = X[self.numeric_inputs].copy()
        Y_encoded = X[self.categorical_columns].copy()
        X_return = X.copy()
        for col, model in self.models.items():
            if col in Y_encoded.columns:
                index_null = (
                    Y_encoded[col]
                    .isnull()
                    .replace(False, np.nan)
                    .dropna()
                    .index.tolist()
                )
                if len(index_null) >= 1:
                    X_return.loc[index_null, col] = model.predict(
                        X_encoded.loc[index_null]
                    )
            else:
                raise ValueError(
                    f"La columna {col} no estuvo en los datos de entrada durante el "
                    "Fitting para hacer predicción o puede que si pero no se guardo "
                    "porque no cumplia la metrica adecuada...."
                )
        gc.collect()
        return X_return


def treatment_null_categorical(df1: pd.DataFrame, params: Dict[str, Any]):
    """
    Procesa los datos categoricos que son nulos.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para entrenar y fitear modelo
        para tratar nulos de variables categoricas. (Data Entrenamiento)

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    object
        El modelo que contiene el procesamiento de nulos categoricos y boleanos.
    """

    logger.info(
        "Iniciando el Fitting de datos nulos sobre " "strings/booleanos y categorias"
    )
    process_calc = params["null_adj"]["categorical"]
    process = params["null_adj"]["method_categorical"]
    # random_state = params["null_adj"]["random_state"]
    target = [params["target"]]
    ids = [params["id"]]
    llave_no_run = "No Categoricas"

    df_categorical = df1.copy()
    df_categorical = df_categorical.select_dtypes(
        include=["object", "category", "bool"]
    )
    for col in target + ids:
        try:
            df_categorical = df_categorical.drop(col, axis=1)
        except Exception:
            pass

    names_col_null = df_categorical.columns.tolist()

    logger.info("Variables Categoricas:")
    logger.info(names_col_null)

    if df_categorical.shape[1] >= 1:
        if process == "Simple":
            imputer = SimpleImputer(
                strategy=process_calc
            )  # 'median', 'most_frequent', y 'constant'
            imputer.fit(df1[names_col_null])
            return imputer
        elif process == "Interactive":
            # names_col_null = df_categorical.isnull().sum()
            # .replace(0,np.nan).dropna().index.tolist()
            # if len(names_col_null) >= 1:
            imputer = aux_imputer_categ(df1, params)
            imputer.fit(names_col_null)
            # else:
            #    imputer = llave_no_run
            return imputer
        else:
            logger.info(
                "Procesamiento de datos nulos en variables "
                "categoricas no parametrizado..."
            )
            raise ValueError(
                "Procesamiento de datos nulos en variables "
                "categoricas no parametrizado..."
            )
    else:
        imputer = llave_no_run
        return imputer


# 6. Ejecutar de datos nulos en categorias o strings o booleanos
def run_categorical_values(
    df1: pd.DataFrame, model_imputer_model_categorical, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Procesa los datos categoricos que son nulos y los rellena.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para entrenar y fitear modelo
        para tratar nulos de variables categoricas. (Data Entrenamiento)

    model_imputer_model_categorical:
        Objeto que contiene el metodo de procesar los datos nulos fiteado

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    pd.DataFrame
        El Dataframe que limpia los datos nulos tipo: categoricas, string y
        boleanos junto con las demas variables numericas
    """

    # logger.info(f"Iniciando con el tratamiento de datos
    # nulos en variables categoricas")
    target = [params["target"]]
    ids = [params["id"]]
    process = params["null_adj"]["method_categorical"]
    llave_no_run = "No Categoricas"
    # aseguro quitar los None
    df1 = df1.fillna(np.nan)
    if model_imputer_model_categorical == llave_no_run:
        df2 = df1.copy()
    else:
        try:
            df_categorical = df1.drop(target + ids, axis=1).select_dtypes(
                include=["object", "category", "bool"]
            )
        except Exception:
            df_categorical = df1.copy()
            for col in target + ids:
                try:
                    df_categorical.drop(col, axis=1, inplace=True)
                except Exception:
                    pass
            df_categorical = df1.select_dtypes(include=["object", "category", "bool"])
        finally:
            df_categorical = df_categorical.fillna(np.nan)

        names_col_null = df_categorical.columns.tolist()
        for col_fit in names_col_null:
            if col_fit not in model_imputer_model_categorical.feature_names_in_:
                logger.info(
                    "Es posible que algun valor numerico le "
                    f"quede como string!!!! Variable: '{col_fit}' "
                )
        for col_fit in model_imputer_model_categorical.feature_names_in_:
            if col_fit not in names_col_null:
                logger.info(
                    f"Faltan variables strings!!!  Variable: '{col_fit}'. "
                    "Se ejecutara el codigo a la fuerza"
                )
        names_col_null = model_imputer_model_categorical.feature_names_in_
        logger.info(names_col_null)
        if process == "Simple":
            df2 = model_imputer_model_categorical.transform(df1[names_col_null])
            df2 = pd.DataFrame(df2, columns=names_col_null, index=df1.index)
            df2 = pd.concat(
                [df2, df1[df1.columns[~df1.columns.isin(names_col_null)]]], axis=1
            )
            df2 = df2[df1.columns]
        elif process == "Interactive":
            df2 = model_imputer_model_categorical.transform(df1)
        else:
            raise ValueError(
                f"La columna {col} no está en los datos de entrada para predicción."
            )
    # logger.info(f"Actualizando los datos nulos en variables categoricas")
    return df2


# 7. Como hacer el One Hot Encoding
def One_Hot_encoding_keys(df1: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifica los Feature de las variables categoricas y como hace el
    OneHotEncoding.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se usara para entrenar y fitear modelo
        para tratar nulos de variables categoricas. (Data Entrenamiento)

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    pd.DataFrame
        Retorna un DataFrame que comprende la caracteristica omitida en la fila
        'DROP_FIRST' de la variable eliminada y las caracteristicas de
        entrenamiento.
    """

    logger.info("Iniciando Comprensión del One Hot Encoding...")
    target = [params["target"]]
    ids = [params["id"]]

    X = df1.copy()
    X = X.select_dtypes(include=["object", "category", "bool"])
    for col in target + ids:
        try:
            X = X.drop(col, axis=1)
        except Exception:
            pass

    feature_imp = pd.DataFrame()
    if X.shape[1] == 0:
        feature_imp["REINDEX"] = np.nan
    else:
        X_train_encoded_pd = pd.get_dummies(X, drop_first=True)

        feature_imp["REINDEX"] = X_train_encoded_pd.columns
        for col in X.columns:
            items = list(X[col].unique())
            k = 0
            for i in items:
                if str(col) + "_" + str(i) not in X_train_encoded_pd.columns:
                    feature_imp.loc["DROP_FIRST", col] = i
                else:
                    feature_imp.loc[k, col] = i
                    k = k + 1
    feature_imp.index.name = "llave"
    return feature_imp


# 8. Efectuando el One Hot Encoding
def One_Hot_encoding_func(
    df1: pd.DataFrame, df_keys: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Procesa o efectual el OneHotEncoding

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame de Pandas sobre el cual se hara el One Hot Encoding.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        Retorna un archivo  con el OneHotEncoding Realizado
        y con todas las demas variables
    """
    logger.info("Iniciando a actualizar el One Hot Encoding...")

    target = [params["target"]]
    ids = [params["id"]]

    try:
        df_keys.set_index("llave", inplace=True)
    except Exception:
        pass

    # lista de columnas de entrenamiento
    variables_OHE = list(df_keys["REINDEX"].dropna().values)
    variables_before_ = df_keys.drop(["REINDEX"], axis=1).columns.tolist()

    index_list = df1.index
    if len(variables_OHE) == 0:
        X_adj = df1.copy()
    else:
        try:
            X = df1.drop(target + ids, axis=1)
            y = df1[target + ids]
        except Exception:
            X = df1.copy()
            y = pd.DataFrame(index=df1.index)
            for col in target + ids:
                try:
                    X.drop(col, axis=1, inplace=True)
                    y = pd.concat([y, df1[col]], axis=1)
                except Exception:
                    pass

        X1 = X.select_dtypes(include=["number"])
        X = X.select_dtypes(include=["object", "category", "bool"])

        X_encoded_pdd = pd.get_dummies(X[variables_before_])
        X_encoded_pd = X_encoded_pdd.reindex(columns=variables_OHE, fill_value=0)
        X_encoded_pd.index = index_list
        X_adj = pd.concat([X1, X_encoded_pd, y], axis=1)

        for col in variables_before_:
            items = list(X[col].unique())
            for i in items:
                llave = str(col) + "_" + str(i)
                # for llave in X_encoded_pdd.columns.tolist():
                if llave not in variables_OHE:
                    #            col = "_".join(llave.split("_")[:-1])
                    #            i = llave.split("_")[1]
                    strr = df_keys.loc["DROP_FIRST", col]
                    if strr == i:
                        pass
                    else:
                        logger.info(
                            f"La Feature: {i} de la variable: {col}. "
                            f"Se procesa como {strr}"
                        )
    return X_adj


# 9. Escalado de variables numéricos
def get_scaler(params: Dict[str, Any]) -> Tuple:
    """
    Devuelve el scaler adecuado basado en el método de
    estandarización especificado en params.

    Parameters
    ----------
    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    Tuple
        Objeto de scaler a usar.
    """
    method = params["Estandarizacion"]
    if method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Normalized":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Method '{method}' is not supported.")
    return scaler


def scale(
    df1: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Union[StandardScaler, MinMaxScaler], Any]:
    """
    Fitear el metodo de Estandarizacion de todas las variables.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se estandarizará.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        Objeto o modelo que servira para estandarizar
    """
    logger.info("Fiteando la estandarizacion de datos")
    target = [params["target"]]
    ids = [params["id"]]
    scaler = get_scaler(params)
    # Ajustar el scaler con los datos de entrenamiento
    try:
        X_train = df1.drop(target + ids, axis=1)
    except Exception:
        X_train = df1.copy()
        for col in target + ids:
            try:
                X_train.drop(col, axis=1, inplace=True)
            except Exception:
                pass
    scaler.fit(X_train)
    col_names_all = list(X_train.columns)
    return scaler, col_names_all


# 10. ejecutando la estandarizacion
def min_max_scaler_pd(
    df1: pd.DataFrame,
    params: Dict[str, Any],
    scaler_min_max: [StandardScaler, MinMaxScaler],
    col_names_all: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estandariza las columnas numéricas (excluyendo binarias)
    de dos DataFrames utilizando el método Min-Max Scaler.

    Parameters
    ----------
    df1 : pd.DataFrame
        Primer DataFrame de Pandas que se estandarizará.

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    scaler:
        Modelo Fiteado para transformar

    Returns
    -------
    pd.DataFrame
        Tupla [DataFrame estandarizado con los inputs solamente,
        Otro DataFrame con ids y variable de interes].
    """
    logger.info("Iniciando la estandarización con Min-Max Scaler...")

    target = [params["target"]]
    ids = [params["id"]]
    try:
        X = df1.drop(target + ids, axis=1)
        otras_variables = df1[target + ids]
    except Exception:
        X = df1.copy()
        otras_variables = pd.DataFrame(index=df1.index)

        for col in target + ids:
            try:
                X.drop(col, axis=1, inplace=True)
                otras_variables = pd.concat([otras_variables, df1[col]], axis=1)
            except Exception:
                pass
    df1_scaled = scaler_min_max.transform(X[col_names_all])
    df1_scaled = pd.DataFrame(df1_scaled, columns=col_names_all, index=X.index)
    logger.info("Estandarización con Min-Max Scaler completada!")
    return df1_scaled


# 11. save_transformer: Guardar y compilar el artefacto de preprocesamiento
class process_forecast_data:
    """
    Consolida el artefacto para procesar los datos en una Clase

    Parameters
    ----------
    order_col_numeric:
        Orden de fiteo de datos numericos con los que se trato todo.

    imputer_model_numeric :
        Clase u objeto funcional para predecir y transformar los datos nulos numericos

    model_imputer_model_categorical:
        Clase u objeto funcional para predecir y transformar los datos nulos categoricos

    outliers_adj:
        DataFrame que contiene las variables que aplican una transfomracion de outliers

    reindex_OneHotEncoding:
        DataFrame que contiene la parametrizacion del OneHotEncoding

    model_imputer_scale:
        Funcion que aplica el escalado de las variables

    filters:
        Lista de filtros de la variable de interes

    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        Clase que contiene todo el artefacto de procesamiento
        de data y que al momento de usar self.transform(df1)
        sobre cualquier data df1 retornara:
            Tupla pd.DataFrame que se genera para poder predecir
            los datos teniendo en cuenta el procesamiento de datos de entrenamiento.
                Primer DataFrame son los inputs del modelo
                Segundo DataFrame son la variable de interes si existe junto con los Ids
    """

    def __init__(
        self,
        col_names_all,
        order_col_numeric,
        imputer_model_numeric,
        model_imputer_model_categorical,
        outliers_adj,
        reindex_OneHotEncoding,
        model_imputer_scale,
        filters,
        params: Dict[str, Any],
    ):
        self.order_col_all = col_names_all
        self.order_col_numeric = order_col_numeric
        self.imputer_model_numeric = imputer_model_numeric
        self.model_imputer_model_categorical = model_imputer_model_categorical
        self.outliers_adj = outliers_adj
        self.reindex_OneHotEncoding = reindex_OneHotEncoding
        self.model_imputer_scale = model_imputer_scale
        self.filters = filters
        self.params = params

    def all_importance_variables(self):
        self.filters = list(
            set(self.filters + [self.params["id"]] + [self.params["target"]])
        )

    def transform(self, df1):
        self.all_importance_variables()
        df2 = df1.copy()

        y_interest = pd.DataFrame(index=df1.index.tolist())
        for col in [self.params["target"]] + [self.params["id"]]:
            if col in df1.columns:
                y_interest = pd.concat([y_interest, df2[col]], axis=1)

        df_adj = feature_selec_pd(df2, self.filters, self.params)
        # procesamiento de datos numericos
        try:
            df_adj1 = run_numeric_values(
                df_adj,
                self.order_col_numeric,
                self.imputer_model_numeric,
                self.outliers_adj,
                self.params,
            )
        except ValueError as e:
            logger.info("OCURRIO EL SIGUIENTE ERROR:")
            logger.info(f"{e}")
            control = (df_adj.abs() == np.inf).sum()
            if control.max() >= 1:
                logger.info("Identificamos datos infinitos...")
                logger.info("Volviendo estos datos como nulos...")
                df_adj_except = df_adj.replace(np.inf, np.nan)
                df_adj_except = df_adj_except.replace(-np.inf, np.nan)
            else:
                df_adj_except = df_adj.copy()
            try:
                logger.info("INTENTANDO VOLVER A PROCESAR LOS DATOS")
                df_adj1 = run_numeric_values(
                    df_adj_except,
                    self.order_col_numeric,
                    self.imputer_model_numeric,
                    self.outliers_adj,
                    self.params,
                )
            except ValueError as e:
                logger.info("OCURRIO EL SIGUIENTE ERROR:")
                logger.info(f"{e}")
                logger.info("Llenando los datos nulos numericos con la media...")
                for col in self.order_col_numeric:
                    try:
                        df_adj_except[col] = df_adj_except[col].fillna(
                            df_adj_except[col].mean()
                        )
                    except Exception:
                        logger.info(f"No encontramos la variable categorica {col}")

                logger.info(
                    "Mantenemos la calificación del "
                    f"{np.round(100*df_adj_except.shape[0]/df_adj_except.shape[0])}"
                    "% de la data"
                )
                df_adj1 = run_numeric_values(
                    df_adj_except,
                    self.order_col_numeric,
                    self.imputer_model_numeric,
                    self.outliers_adj,
                    self.params,
                )
        # procesamiento de datos categoricos
        try:
            df_adj2 = run_categorical_values(
                df_adj1, self.model_imputer_model_categorical, self.params
            )
        except ValueError as e:
            logger.info("OCURRIO EL SIGUIENTE ERROR:")
            logger.info(f"{e}")
            df_adj_except = df_adj1.copy()
            logger.info("Llenando los datos nulos numericos con la mediana...")
            for col in self.model_imputer_model_categorical.feature_names_in_:
                try:
                    df_adj_except[col] = df_adj_except[col].fillna(
                        df_adj_except[col].median()
                    )
                except Exception:
                    logger.info(f"No encontramos la variable categorica {col}")
            logger.info(
                "Mantenemos la calificación del "
                f"{np.round(100*df_adj_except.shape[0]/df_adj_except.shape[0])}"
                "% de la data"
            )
            df_adj2 = run_categorical_values(
                df_adj_except, self.model_imputer_model_categorical, self.params
            )
        # one hot encoding
        df_adj3 = One_Hot_encoding_func(
            df_adj2, self.reindex_OneHotEncoding, self.params
        )
        # estandarizacion de datos
        scaler_master = min_max_scaler_pd(
            df_adj3, self.params, self.model_imputer_scale, self.order_col_all
        )
        return scaler_master, y_interest


def save_transformer(
    df_importance,
    order_col_numeric,
    col_names_all,
    imputer_model_numeric,
    model_imputer_model_categorical,
    outliers_adj,
    reindex_OneHotEncoding,
    model_imputer_scale,
    params: Dict[str, Any],
):
    """
    Balancea la variable objetivo utilizando el método
    Synthetic Minority Over-sampling Technique (SMOTE)
    en dos DataFrames.

    Parameters
    ----------
    df_importance:
        DataFrame que contiene las caracteristicas relevantes desde Features

    order_col_numeric:
        Orden de fiteo de datos numericos con los que se trato todo.

    imputer_model_numeric :
        Clase u objeto funcional para predecir y transformar los datos nulos numericos

    model_imputer_model_categorical:
        Clase u objeto funcional para predecir y transformar los datos nulos categoricos

    outliers_adj:
        DataFrame que contiene las variables que aplican una transfomracion de outliers

    reindex_OneHotEncoding:
        DataFrame que contiene la parametrizacion del OneHotEncoding

    model_imputer_scale:
        Funcion que aplica el escalado de las variables


    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
        Clase que contiene todo el artefacto de procesamiento de data
        y que al momento de usar self.transform(df1)
        sobre cualquier data df1 retornara:
            Tupla pd.DataFrame que se genera para poder predecir
            los datos teniendo en cuenta el procesamiento de datos de entrenamiento.
                Primer DataFrame son los inputs del modelo
                Segundo DataFrame son la variable de interes si existe junto con los Ids
    """
    logger.info("Iniciando a guardar todo el procesamiento de datos")
    filters = list(df_importance.columns)
    claseee = process_forecast_data(
        col_names_all,
        order_col_numeric,
        imputer_model_numeric,
        model_imputer_model_categorical,
        outliers_adj,
        reindex_OneHotEncoding,
        model_imputer_scale,
        filters,
        params,
    )
    logger.info("Guardando")
    return claseee
