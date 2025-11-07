"""
Nodos de la capa feature
"""
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# from sklearn.impute import IterativeImputer

# Configuración básica del logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 0. Función para calcular nuevas variables
def calculate_new_variables_pd(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Calcula las variables requeridas segun la forma como esta parametrizada en los
    parametros

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas necesarias para calcular las nuevas variables.

    params: Dict
        Dict que contiene un objeto 'crear_nuevas_variables' con la parametrizacion
        de la construccion de variables nuevas

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas variables calculadas.
    """

    new_variables = list(params["crear_nuevas_variables"].keys())
    for new in new_variables:
        name_variable = params["crear_nuevas_variables"][new]["nombre"]
        insumos = params["crear_nuevas_variables"][new]["insumos"]
        metodo = params["crear_nuevas_variables"][new]["metodo"]
        for variable in list(insumos.keys()):
            if variable not in df.columns:
                logger.info(
                    f"La variable insumo: {variable} no "
                    "esta parametrizada en el DataFrame de entrada"
                )

        try:
            if metodo == "separar_string":
                variable_insumo = list(insumos.keys())[0]
                key = list(insumos.values())[0]
                pos_str = int(key[1:])
                if key[0] == ">":
                    df[name_variable] = df[variable_insumo].astype(str).str[pos_str:]
                elif key[0] == "<":
                    df[name_variable] = df[variable_insumo].astype(str).str[:pos_str]
                else:
                    # generamos error
                    logger.info(
                        f"El simbolo {key[0]} no se identica.. solamente se identifica"
                        f' "<", ">" cuando se construye: {name_variable}'
                    )
                    raise
                try:
                    df[name_variable] = df[name_variable].astype(int)
                except Exception:
                    pass
            elif metodo == "sumar":
                total_valor = 0
                for variable, key in zip(insumos.keys(), insumos.values()):
                    if key == "+":
                        total_valor = total_valor + df[variable]
                    elif key == "-":
                        total_valor = total_valor - df[variable]
                    else:
                        # generamos error
                        logger.info(
                            f"El simbolo {key} asociado a la variable insumo:"
                            f"{variable} no esta identificada dentro "
                            f"del metodo: {metodo}"
                        )
                        logger.info(f"Cuando se construye: {name_variable}")
                        raise
                df[name_variable] = total_valor.values
            else:
                # generamos error
                logger.info(
                    f"Metodo no identificado en la construccion de  {name_variable}"
                )
                raise
            logger.info(f'Se construyo la variable "{name_variable}" con exito!')

        except Exception:
            logger.info(
                f"No se pudo construir la variable '{name_variable}' "
                f"con el metodo '{metodo}' usando estos insumos:"
            )
            logger.info(insumos)
            raise
    logger.info(
        "Nuevas variables calculadas completamente. "
        f"Dimension de la data: {df.shape}"
    )
    return df


# 1 Guardar la manera como se homologa las regiones
def modelo_homologacion_regiones(parametros):
    """
    La funcion tiene como objetivo guardar la forma como se homologaran los strings
    para el procesamiento y la prediccion de los datos.

    Args:
        parametros: Metodo de como se homologara y preprocesaran los datos

    Returns:
        Dict: DataFrame con la metodologia respectiva de homologacion de strings.
    """

    logger.info("Guardando el pickle de como se homologan las regiones...")
    return parametros["homologacion_x_variable"]


# 2 Agregar regiones segun el departamento
def homologate_region(df: pd.DataFrame, modelo_homologaciones: Dict) -> pd.DataFrame:
    """
    Homologa los strings en un especio mas pequeño del DataFrame en base al
    diccionario de homologaciones.

    Args:
        df (pd.DataFrame):
            DataFrame que contiene la columna de departamentos a homologar.
        homologate_region_model: Any
            Pickle con toda la parametrizacion de como homologar los datos antes
            del preprocesamiento

    Returns:
        pd.DataFrame: DataFrame con la nueva columna de regiones homologadas.
    """

    columns_homologate = list(modelo_homologaciones.keys())
    for col in columns_homologate:
        target_col = modelo_homologaciones[col]["nombre"]
        insumo_col = modelo_homologaciones[col]["insumo"]
        nulos_value = modelo_homologaciones[col]["fillna"]
        homologations = modelo_homologaciones[col]["modo_homologacion"]
        # Normaliza las caracteristicas a strings
        # otra vez(ej. minúsculas y sin espacios)
        # esto lo hacemos asi no sea necesario por si en otra
        # capa se requiere ejecutar la data en crudo y
        # pegarle esto a los resultados finales
        df[insumo_col] = df[insumo_col].str.lower().str.replace(r"\s+", "_", regex=True)

        for character in df[insumo_col].unique():
            if character not in list(homologations.keys()):
                if character not in [None, np.nan]:
                    logger.info(
                        f"Caracter: {character} de la variable insumo: "
                        f"{insumo_col} no parametrizado en la homologacion de"
                        f" strings para crear: {target_col}"
                    )
        # Mapear los departamentos a regiones usando el diccionario de homologaciones
        if (str(nulos_value) == "None") | (nulos_value is None):
            logger.info("La nulidad se manejara en la capa de model_input")
            df[target_col] = df[insumo_col].map(homologations)
        else:
            logger.info(
                f"La nulidad se remplazara con {nulos_value} "
                "antes de la capa de model_input"
            )
            df[target_col] = df[insumo_col].map(homologations).fillna(nulos_value)

        logger.info(f"Ok la homologacion de la variable {target_col} con {insumo_col}")

    logger.info(f"Tamaño de la data: {df.shape}")
    return df


# 3 Eliminar columnas de apertura de productos
def eliminar_columnas(df: pd.DataFrame, parametros: List[str]) -> pd.DataFrame:
    """
    Elimina las columnas especificadas en cols_to_drop del DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    DataFrame que contiene los datos de entrada.
    cols_to_drop : List[str]
    Lista de columnas que se desea eliminar.

    Returns
    -------
    pd.DataFrame
    DataFrame sin las columnas especificadas.
    """

    cols_to_drop = parametros["cols_to_drop"]
    cols_to_drop1 = [i.upper() for i in cols_to_drop] + [
        i.lower() for i in cols_to_drop
    ]
    cols_to_drop = cols_to_drop + cols_to_drop1
    cols_to_drop = list(set(cols_to_drop))
    # Verificar si las columnas a eliminar existen
    # en el DataFrame antes de intentar eliminarlas
    columnas_existentes_a_eliminar = [col for col in cols_to_drop if col in df.columns]
    logger.info(
        "Segun la parametrizacion de la Fabrica se"
        f" eliminara: {columnas_existentes_a_eliminar}"
    )
    # Eliminar las columnas que existen en el DataFrame
    df.drop(columns=columnas_existentes_a_eliminar, inplace=True)
    logger.info(f"Tamaño de mi df: {df.shape}")
    return df


# 4. Función para procesar la importancia de características
def preprocesar_feature_df(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesa el DataFrame para codificar variables categóricas y realizar
    imputación de valores faltantes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las características originales.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye la columna a conservar.

    Returns
    -------
    pd.DataFrame
        DataFrame preprocesado con variables categóricas codificadas y valores
        faltantes imputados.
    """

    logger.info("Iniciando el preprocesamiento (LabelEncoder) del DataFrame...")

    # Identificar columnas categóricas
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    # target = params["target"]

    # Codificación de columnas categóricas
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Imputar valores faltantes
    logger.info("Iniciando el preprocesamiento (Completitud) del DataFrame...")
    medianas_all = df_encoded.select_dtypes(include=["number"]).median()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            mode_all = (
                df_encoded[[col]]
                .select_dtypes(include=["object", "string"])
                .mode()
                .iloc[0]
            )
            df_encoded.loc[:, col] = df_encoded[col].fillna(mode_all.loc[col])
        else:
            # df_encoded[col].fillna(df_encoded[col].median(), inplace=True)
            df_encoded.loc[:, col] = df_encoded[col].fillna(medianas_all.loc[col])

    # Convertir todos los datos a numéricos
    df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce")

    return df_encoded


# 5. separar características
def separar_características(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las características y la variable objetivo del DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preprocesado.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye la columna a conservar.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        DataFrame con características y Serie con la variable objetivo.
    """
    logger.info("Separando características y la variable objetivo...")

    # Verifica que params['target'] sea un string
    target = params.get("target")
    ids = params.get("id")

    if not isinstance(target, str):
        logger.error(f"El valor de 'target' no es una cadena: {target}")
        raise TypeError(
            "El valor de 'target' debe ser una cadena que "
            "representa el nombre de la columna."
        )

    logger.info(f"Columna objetivo: {target}")
    logger.info(f"Columna a eliminar de los inputs: {target},{ids}")

    X = df.drop(columns=[target, ids])
    y = df[[target]]
    return X, y


# 6. Calcular importancia
def calcular_importancia(X: pd.DataFrame, y: pd.Series, params: Dict) -> pd.DataFrame:
    """
    Calcula la importancia de las características utilizando un modelo de XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame con las características.
    y : pd.Series
        Serie con la variable objetivo.
    params: Dict
        Parametros del uso del modelo

    Returns
    -------
    pd.DataFrame
        DataFrame con la importancia de las características ordenadas.
    """
    logger.info("Calculando la importancia de las características usando XGBoost...")
    target = params.get("target")
    y = y[target]
    # Definir el modelo XGBoost
    # regressor = XGBRegressor(max_depth=5, random_state=42, n_estimators=60)
    objective = "binary:logitraw"  # las predicciones son los logit
    objective = "binary:logistic"  # las predicciones son probabilidades
    test_size = params["bootstrapping_feature"]["test_size"]
    ciclos = params["bootstrapping_feature"]["N_Iter"]
    by_order = params["bootstrapping_feature"]["by_order"]
    importance_df_full = pd.DataFrame()
    for i in range(ciclos):
        random_state = int(np.random.rand(1)[0] * 1000)
        X_loop, X_loop1, y_loop, y_loop1 = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
            shuffle=True,
        )

        objective = "binary:logistic"  # las predicciones son probabilidades
        regressor = XGBClassifier(
            objective=objective, n_estimators=60, max_depth=5, random_state=42
        )
        # Ajustar el modelo a los datos
        regressor.fit(X_loop, y_loop)

        # Obtener la importancia de las características
        importances = regressor.feature_importances_
        feature_names = X_loop.columns
        importance_df = pd.DataFrame(
            {"Feature": feature_names, f"Importance_{i}": importances}
        )

        # Ordenar las características por importancia
        importance_df = importance_df.set_index("Feature")
        if by_order is True:
            importance_df = importance_df.sort_values(
                by=f"Importance_{i}", ascending=False
            )
            importance_df[f"Importance_{i}"] = list(
                range(importance_df.shape[0], 0, -1)
            )
        importance_df_full = pd.concat([importance_df_full, importance_df], axis=1)
        msj = 100 * (i / ciclos)
        if msj % 10 == 0:
            logger.info(
                f"Bootstrapping en la iteración {i} de {ciclos}. Ejecución al {msj}%"
            )
    importance_df_full = (
        importance_df_full.mean(axis=1)
        .to_frame()
        .rename(columns={0: "Importance"})
        .reset_index()
    )
    importance_df_full = importance_df_full.sort_values(
        by="Importance", ascending=False
    )
    return importance_df_full


# 7. Seleccionar características
def seleccionar_características(
    importance_df: pd.DataFrame, df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Orden las características segun su importancia y añade las caracteristicas
    requeridas en el modelo.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame con la importancia de las características.
    df : pd.DataFrame
        DataFrame original con las características.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye las características requeridas.

    Returns
    -------
    pd.DataFrame
        DataFrame con la importancia de las características seleccionadas.
    """

    # Asegurar que las características requeridas estén en importance_df
    metodo = params["requered_importances"]["ignore_importance_model"]
    required_features = params["requered_importances"]["variables_select"]
    n_top_model = params["requered_importances"]["n_top_select_importance_model"]
    add_variables = params["requered_importances"]["add_variables"]
    if add_variables is None:
        add_variables = []
    if required_features is None:
        required_features = []
    add_variables = [feat for feat in add_variables if feat in df.columns]
    required_features_in_df = [feat for feat in required_features if feat in df.columns]

    for feat in required_features_in_df + add_variables:
        if feat not in importance_df["Feature"].values:
            importance_df = pd.concat(
                [
                    importance_df,
                    pd.DataFrame({"Feature": [feat], "Importance": [0.001]}),
                ],
                ignore_index=True,
            )

    # Ordenar por importancia
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    if metodo is False:
        logger.info(f"Seleccionando las  {n_top_model} características más importantes")
        selected_columns = importance_df.head(n_top_model)["Feature"].tolist()
        logger.info(
            f"Agregando {len(add_variables)} variables extras a las"
            f" {n_top_model} características más importantes"
        )
        selected_columns = list(set(selected_columns + add_variables))
    else:
        logger.info(
            "Ignorando las variables de mayor importancia y escogiendo las requeridas"
        )
        selected_columns1 = list(set(required_features_in_df))  # unicos
        selected_columns = []
        for col in selected_columns1:
            if col in importance_df["Feature"].values:
                selected_columns.append(col)
            else:
                logger.info(
                    f"La variable {col} no esta parametrizada en los feature"
                    " importance... Puede que este mal escrita en los parameters"
                )

    importance_df = importance_df[importance_df["Feature"].isin(selected_columns)]
    logger.info(f"Se seleccionaron {len(importance_df)} características importantes.")
    return importance_df


# 8. Función para filtrar columnas en un DataFrame
def filtrar_columnas_df(
    segmented_df: pd.DataFrame, importance_df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Filtra las columnas de un DataFrame basándose en la importancia de las
    características y conserva una columna adicional especificada, eliminando
    columnas no deseadas si están presentes.

    Parameters
    ----------
    segmented_df : pd.DataFrame
        DataFrame que contiene el segmento a filtrar.
    importance_df : pd.DataFrame
        DataFrame con la importancia de las características.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye la columna a conservar.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado con las columnas seleccionadas.
    """

    logger.info(
        "Filtrando columnas del DataFrame basado "
        "en la importancia de características..."
    )

    target = params["target"]
    selected_columns = importance_df["Feature"].tolist()
    logger.info(f"Inputs seleccionados: {selected_columns}")
    # Asegurar que la columna a conservar esté en la lista de columnas seleccionadas
    if target not in selected_columns:
        logger.info(f"Incorporando la variable objetivo: {target} en la lista")
        selected_columns.append(target)

    filtered_df = segmented_df[selected_columns]
    logger.info("Filtrado completado (Inputs y Output).")
    return filtered_df
