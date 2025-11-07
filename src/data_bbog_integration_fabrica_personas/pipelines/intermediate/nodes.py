"""
Nodos de la capa intermediate
"""
from typing import Dict, Any

import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Filtro de segmento
def filter_data_segment_pd(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Filtra un DataFrame basado en múltiples valores de un segmento específico de
    datos y reasigna estos valores a un nombre único.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos a filtrar.
    params: Dict[str, Any]
        Diccionario de parámetros intermediate.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado y modificado con los valores reasignados.
    """

    # Registra un mensaje de información indicando el inicio del filtrado
    logger.info(
        "Iniciando el filtrado de datos por segmento y reasignación de valores..."
    )

    # Extrae los nombres de las columnas, valores y
    # el nombre a reasignar desde el diccionario de parámetros
    condiciones = list(params["filter_segment"].keys())
    filtered_df = df.copy()
    for t_cond, condicion in enumerate(condiciones):
        logger.info(f"Condicion Nro {t_cond+1}")
        segment_col = params["filter_segment"][condicion]["column"]
        segment_values = params["filter_segment"][condicion]["value"].split(", ")
        # Filtra el DataFrame basado en los valores proporcionados
        filtered_df = filtered_df[filtered_df[segment_col].isin(segment_values)]
        segment_values_found = filtered_df[segment_col].unique().tolist()
        logger.info(f"Segmentos que se desean filtrar: {segment_values}")
        logger.info(f"Segmentos encontrados de interes: {segment_values_found}")
        # Registra un mensaje de información indicando
        # el resultado del filtrado y reasignación
        logger.info(f"{len(filtered_df)} registros retenidos con esta condición.")
    logger.info(
        f"Dimension Original {df.shape}. Dimension Filtrada: {filtered_df.shape}"
    )
    # Retorna el DataFrame filtrado y modificado
    return filtered_df


# 2. Filtro de producto
def filter_data_prod_pd(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Elimina columnas específicas de un DataFrame basadas en una lista de variables
    proporcionada en los parámetros, excluyendo la primera variable, que se
    considera la variable objetivo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos a procesar.
    params: Dict[str, Any]
        Diccionario de parámetros que contiene la lista de columnas a eliminar.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas especificadas eliminadas.
    """

    # Registra un mensaje de información indicando el
    # inicio del proceso de eliminación de columnas
    logger.info(
        f"Iniciando la eliminación de columnas de datos de producto {df.shape} ..."
    )

    # Obtiene la variable objetivo, que es el primer elemento de la lista
    target_variable = params["target"]

    # Extrae la lista de columnas a eliminar desde el diccionario de parámetros
    all_columns_to_drop1 = params["filter_product"]
    all_columns_to_drop2 = list(params["todas_variables_apertura"].keys())
    all_columns_to_drop3 = list(params["todas_variables_apertura"].values())
    all_columns_to_drop4 = list(params["future_target_window"].keys())
    all_columns_to_drop = list(
        set(
            all_columns_to_drop1
            + all_columns_to_drop2
            + all_columns_to_drop3
            + all_columns_to_drop4
        )
    )

    columns_to_drop = []
    for col in all_columns_to_drop:
        if col in df.columns:
            if col != target_variable:
                columns_to_drop.append(col)

    # Elimina las columnas especificadas
    df_filtered = df.drop(columns=columns_to_drop, errors="ignore")
    # Elimina las filas donde tenemos NA en la variable objetivo:
    # Los ultimos periodos donde finalmente no se supo si aperturo o no el producto
    df_filtered = df_filtered[~df_filtered[target_variable].isnull()]
    try:
        df_filtered[target_variable]
        logger.info("La variable objetivo esta identificada")
    except Exception:
        logger.info(f"La variable objetivo: {target_variable} no esta identificada")
        raise
    # Registra un mensaje de información indicando el resultado de la eliminación
    logger.info(
        "Eliminación de columnas completada. Variable objetivo: "
        f"{target_variable} y el dataframe resultante es {df_filtered.shape}"
    )

    # Retorna el DataFrame con las columnas eliminadas
    return df_filtered
