"""
This is a boilerplate pipeline 'pre_raw'
generated using Kedro 0.18.14
"""

from typing import Dict, Any
from io import StringIO
import pandas as pd
import polars as pl
import logging
import gc

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create a string buffer to capture log output
log_capture_string = StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


# Funcion auxiliar para limpiar la data
def convertir_a_float64(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    gc.collect()
    try:
        # si es un parquet pasarlo a polars
        df = pl.from_pandas(df)
    except Exception:
        pass
    numeric_cols = [
        col
        for col in cols
        if df[col].dtype in [pl.Float64, pl.Int64, pl.Int32, pl.Float32]
    ]
    ret = df.select(cols).with_columns(
        [pl.col(col).cast(pl.Float64) for col in numeric_cols]
    )
    gc.collect()
    return ret


# 1. Leer la ventana de entrenamiento y filtrar variables.
def concat_dataframes_pl_pd(params: Dict[str, Any]) -> pd.DataFrame:
    """
    Funcion que unifica las bases de datos para tener informacion en una ventana de tiempo.

    Args:
        params (Dict[str, Any]): Parametros de Kedro

    Returns:
        DataFrame unificado
    """
    rutas = params["rutas_pre_raw"]
    union_tablas = params["union"]
    vars_pre_raw = params["vars_pre_raw"]
    variables = params["vars"]
    on_union = union_tablas["on"]
    how_union = union_tablas["how"]
    df_full = pl.DataFrame()
    for ruta in list(rutas.keys()):
        df_full_corte = pd.DataFrame()
        n_tables = len(rutas[ruta])
        for i, corte_i in enumerate(rutas[ruta], start=1):
            vars_table = vars_pre_raw[f"base{i}"]
            if n_tables > 1:
                vars_table = list(set(vars_table + union_tablas["on"]))

            logger.info(f"Loading data from {corte_i}")
            df = pd.read_parquet(corte_i)

            if "full" not in vars_table:
                df = df[vars_table]
            if i == 1:
                df_full_corte = df.copy()
            else:
                logger.info(f"Uniendo las bases de datos por {on_union}")
                for validate in union_tablas["on"]:
                    if validate not in df.columns:
                        logger.info("Warning!!!")
                        logger.info(f"Variable {validate} not in {corte_i}")
                df_full_corte = pd.merge(df_full_corte, df, on=on_union, how=how_union)
            gc.collect()
        logger.info("Procesando todas las columnas relevantes")
        df_full_corte = df_full_corte[variables]
        df_full_corte = convertir_a_float64(df_full_corte, variables)

        logger.info("Iniciando el proceso de concatenación de DataFrames...")
        df_list = [df_full, df_full_corte]
        logger.info(f"Añadiendo: {df_full_corte.shape[0]} registros.")
        df_full = pl.concat(df_list, how="vertical")
        gc.collect()
    logger.info(
        f"DataFrame concatenado: {df_full.shape[0]} registros y {df_full.shape[1]} columnas."
    )
    df_full_pd = df_full.to_pandas()

    if df_full_pd is None or df_full_pd.empty:
        logger.error("El DataFrame final es None o está vacío.")
        return None

    logger.info(
        "Proceso de concatenación completado y convertido a DataFrame de pandas."
    )
    return df_full_pd
