"""
Nodos de la capa primary
"""
from typing import Dict, Any

import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Filtros de negocio
def filter_business_data_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Filtra los datos de un DataFrame basado en las condiciones especificadas en un diccionario de parámetros.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos a filtrar.
    params: Dict[str, Any]
        Diccionario de parámetros primary.

    Returns
    -------
    pd.DataFrame: DataFrame filtrado según las condiciones especificadas.
    """
    # Registra un mensaje de inicio del proceso de filtrado
    logger.info(
        "Iniciando el filtrado de datos basado en los parámetros proporcionados..."
    )

    # Itera sobre las condiciones de filtrado en 'filter_business'
    for column, condition in params["filter_business"].items():
        if isinstance(condition, list) and len(condition) == 2:
            # Si la condición es una lista, se espera que sean valores mínimo y máximo (rango)
            min_val, max_val = condition
            df = df[(df[column] >= min_val) & (df[column] <= max_val)]
            logger.info(
                f"Filtrado por rango en {column}: {min_val} <= {column} <= {max_val}"
            )

        elif isinstance(condition, str):
            # Si la condición es una cadena, se espera que sea un operador y un valor (e.g., '>0.5')
            operator = condition[0]
            value = float(condition[1:])
            if operator == ">":
                df = df[df[column] > value]
                logger.info(f"Filtrado por {column} > {value}")
            elif operator == "<":
                df = df[df[column] < value]
                logger.info(f"Filtrado por {column} < {value}")
            elif operator == "=":
                df = df[df[column] == value]
                logger.info(f"Filtrado por {column} = {value}")
            else:
                logger.warning(
                    f"Operador no soportado: {operator} en la columna {column}"
                )
        else:
            logger.warning(
                f"Condición de filtro no reconocida para la columna {column}"
            )

    # Registra el número de registros después del filtrado
    final_records = len(df)
    logger.info(f"Filtrado completado. Número de registros finales: {final_records}")

    # Retorna el DataFrame filtrado
    return df
