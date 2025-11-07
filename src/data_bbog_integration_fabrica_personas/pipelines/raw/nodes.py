"""
Nodos de la capa raw
"""
from typing import Dict, List, Any

import re
import pandas as pd
import numpy as np
import logging
import gc

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Leer la ventana de entrenamiento y filtrar variables.
def validar_columnas(df: pd.DataFrame, params: Dict[str, List[str]]) -> pd.DataFrame:
    """

    Valida que el DataFrame contenga las columnas definidas en los parámetros.
    Parameters
    ----------
    df : pd.DataFrame

        DataFrame a validar.

    params : Dict[str, List[str]]

        Diccionario que contiene la lista de columnas esperadas bajo la clave 'vars'.
    Returns

    -------

    pd.DataFrame

        El DataFrame original si todas las columnas están presentes.
    Raises

    ------
    NameError
        Si faltan columnas en el DataFrame.
    """

    # Obtener la lista de columnas esperadas
    logger.info("Validando columnas requeridas...")
    gc.collect()
    expected_columns = params.get("vars", [])
    # filtar las columnas espereadas
    df = df[expected_columns]

    # Identificar las columnas faltantes
    missing_cols = list(set(expected_columns) - set(df.columns))
    gc.collect()
    # Si hay columnas faltantes, registrar un error y lanzar una excepción
    if len(missing_cols) > 0:
        gc.collect()
        logger.error(f"Cantidad de columnas faltantes: {len(missing_cols)}")
        logger.error(f"Columnas faltantes: {missing_cols}")
        gc.collect()
        raise NameError(
            "La tabla no tiene las columnas necesarias para el procesamiento"
        )
    else:
        # Si no faltan columnas, registrar un mensaje de información
        gc.collect()
        logger.info("La tabla contiene las columnas necesarias")
        gc.collect()
        return df


# 2 . minuscalas
def convertir_a_minusculas(df, columna_excluida="hashvalue1"):
    if columna_excluida != "hashvalue1":
        try:
            columna_excluida = columna_excluida["id"]
        except Exception:
            logger.info(columna_excluida)
    # Convertir los nombres de las columnas a minúsculas, excepto la columna excluida
    df.columns = [col.lower() if col != columna_excluida else col for col in df.columns]
    logger.info("Convirtiendo nombres de columnas a minúsculas...")
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    logger.info(string_cols)
    gc.collect()
    # Convertir a minúsculas las columnas que sean de tipo object (string), excepto la columna excluida
    for col in string_cols:
        gc.collect()
        if col != columna_excluida:
            # codigo viejo
            # df[col] = df[col].str.lower()
            # codigo nuevo
            try:
                # prueba = df[col][~df[col].isnull()].astype(float)
                df.loc[:, col] = df.loc[:, col].astype(float)
                logger.info(f"Variable {col} es flotante")
            except Exception:
                df.loc[:, col] = df.loc[:, col].str.lower()
    return df


# 3 Valores especiales
def standardize_strings(df: pd.DataFrame, params: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Estandariza los strings para columnas de texto en un DataFrame de Pandas.
    Reemplaza caracteres con tilde por sus versiones sin tilde, y otros caracteres
    especiales los reemplaza por "_".

    Parámetros:
    df : pd.DataFrame
        DataFrame de entrada.
    param_id_col : str
        Columna que no debe ser modificada.

    Retorna:
    pd.DataFrame
        DataFrame con las columnas de texto estandarizadas.
    """
    # Obtener las columnas de tipo object (string) excepto la columna id
    logger.info("Estandarizando variables de texto: ")
    gc.collect()
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()
    param_id_col = params["id"]
    if param_id_col in string_cols:
        string_cols.remove(param_id_col)
    logger.info(string_cols)

    # Diccionario de caracteres con tildes a reemplazar
    string_to_replace = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ý": "y",
        "à": "a",
        "è": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "ä": "a",
        "ë": "e",
        "ï": "i",
        "ö": "o",
        "ü": "u",
        "ÿ": "y",
        "â": "a",
        "ê": "e",
        "î": "i",
        "ô": "o",
        "û": "u",
        "ã": "a",
        "õ": "o",
        "@": "a",
        "ñ": "n",
    }

    # Caracteres especiales a reemplazar por "_"
    special_chars = r"[()/*\s:.\-;<>?/,'']"

    # Función que reemplaza caracteres en un string
    def replace_special_characters(text):
        if pd.isnull(text):
            return text
        # Reemplazar caracteres con tilde por los equivalentes sin tilde
        text = text.translate(
            str.maketrans(string_to_replace)
        )  # Más eficiente que un bucle de replace
        # Reemplazar caracteres especiales por "_"
        text = re.sub(special_chars, "_", text)
        return text

    # Aplicar los reemplazos a cada columna de tipo string
    for col in string_cols:
        gc.collect()
        try:
            # Intentamos convertir a float (sin aplicar la función de reemplazo en este paso)
            df[col] = pd.to_numeric(df[col], errors="raise")
            logger.info(f"Variable {col} es flotante")
        except ValueError:
            logger.info(f"Ajustando la variable String: {col}")
            # Si no es convertible a float, aplicamos la función de reemplazo
            logger.info(f"{df[col].unique()}")
            df[col] = df[col].apply(replace_special_characters)
            logger.info(f"{df[col].unique()}")
    return df


# 4. definir valores nulos fill nan
def values_to_null(df: pd.DataFrame) -> pd.DataFrame:
    # (df:pd.DataFrame, param_buro_null: List[Any]) -> pd.DataFrame:
    """
    Reemplaza valores por nulos.
    """
    logger.info("Reemplazando valores específicos vacios por nulos...")
    df = df.fillna(np.nan)
    gc.collect()
    return df


# 5. se cambia el tipo de datos a los standard


def change_dtypes(df: pd.DataFrame, parametros: Dict[str, str]) -> pd.DataFrame:
    """
    Cambia el tipo de datos para cada columna del DataFrame según los parámetros.


    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a modificar.
    param_col_types : Dict[str, str]
        Diccionario que contiene los nombres de las columnas y sus tipos esperados.


    Returns
    -------
    pd.DataFrame
        DataFrame con los tipos de datos modificados.
    """
    logger.info("Cambiando la tipologia de los datos...")
    param_col_types = parametros["param_col_types"]
    new_cols = []
    for col in df.columns:
        gc.collect()
        if col in param_col_types:
            expected_dtype = param_col_types[col]
            current_dtype = df[col].dtype

            # Verificar si el tipo de dato actual es diferente del tipo de dato esperado
            if current_dtype != expected_dtype:
                if df[col].isna().any() and (
                    pd.api.types.is_integer_dtype(expected_dtype)
                    or pd.api.types.is_bool_dtype(expected_dtype)
                ):
                    if pd.api.types.is_integer_dtype(expected_dtype):
                        fill_value = 0  # Cambiar a un valor adecuado si necesario
                    elif pd.api.types.is_bool_dtype(expected_dtype):
                        fill_value = False  # Cambiar a un valor adecuado si necesario
                    df[col] = df[col].fillna(fill_value).astype(expected_dtype)
                else:
                    df[col] = df[col].astype(expected_dtype)
        else:
            new_cols.append(col)

    gc.collect()
    logger.warning(f"Cantidad de columnas sin usar: {len(new_cols)}")
    logger.warning(f"Columnas sin usar: {new_cols}")
    return df


# 6. validar que no hay duplicados por cada periodo-id
def validate_unique_id_period_pd(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Valida que no haya duplicados por cada combinación de ID y Periodo en un DataFrame.
    Si se encuentran duplicados, informa cuántos hay por cada periodo y elimina los duplicados,
    quedándose con la última aparición de cada registro.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas de ID y Periodo a validar.
    params: Dict[str, Any]
        Diccionario de parámetros raw.

    Returns
    -------
    pd.DataFrame: DataFrame sin duplicados, manteniendo solo la última aparición por combinación de ID y Periodo.
    """
    # Registra un mensaje de información indicando el inicio de la validación
    logger.info("Iniciando la validación de duplicados por ID y Periodo...")
    gc.collect()
    # Extrae los nombres de las columnas desde el diccionario de parámetros
    id_col = params["id"]
    period_col = params["period_col"]

    # Identifica duplicados en la combinación de ID y Periodo
    logger.info(f"Validando {id_col} y {period_col} únicos...")
    duplicates = df[df.duplicated(subset=[id_col, period_col], keep=False)]

    if not duplicates.empty:
        # Informa cuántos duplicados hay por cada valor de Periodo
        duplicates_count = duplicates.groupby(period_col).size()
        for period, count in duplicates_count.items():
            logger.warning(f"Periodo: {period} Se encontraron {count} duplicados.")
            duplicates_col = duplicates[duplicates[period_col] == period]
            duplicates_id_period_count = duplicates_col.groupby(id_col).size()
            logger.warning(
                "Validando que estas observaciones duplicadas tengan los mismos valores unicos..."
            )
            for ids, counts in duplicates_id_period_count.items():
                duplicates_id_period = duplicates_col[duplicates_col[id_col] == ids]
                cont_full = duplicates_id_period.drop_duplicates().shape[0]
                if duplicates_id_period.shape[0] == cont_full:
                    raise ValueError(
                        f"Cliente: {ids} repetido {counts} veces en {period} y con valores replicables"
                    )
                if cont_full >= 2:
                    raise ValueError(
                        f"Cliente: {ids} repetido {counts} veces en {period} y con valores replicables"
                    )
            logger.warning(f"Ok la validacion del {period} en cada {id_col} ")
        # Elimina los duplicados, quedándose solo con la última aparición
        df = df.drop_duplicates(subset=[id_col, period_col], keep="last")

        logger.info(
            "Duplicados eliminados, conservando la última aparición de cada registro."
        )
    else:
        # Registra un mensaje de información indicando la validación exitosa
        logger.info("Validación exitosa: No se encontraron duplicados.")
    gc.collect()
    # Retorna el DataFrame procesado
    return df


# 7. Modificar targets actuales
# Funcion Auxiliar para crear la variables objetivo
def create_targets(
    df: pd.DataFrame, variable_apertura: Any, target: Any, params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Crea una variable target segun la variable_apertura marcando en x los resultados de x+1,x+2,..,x+shift_max

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de entrada.

    variable_apertura: Any
        String de la variable existente en df sobre la cual se usara para crear la variable objetivo

    target: Any
        String para identificar la variable creada

    params: Dict[str, Any]
        Diccionario de parámetros que contiene las variables necesarias.

    Returns
    -------
    pd.DataFrame: DataFrame del id, period_col, la variable_apertura y target.
    """
    # target = params['target']
    id_col = params["id"]
    periodo_col = params["period_col"]
    # variable_apertura = params['variable_apertura']
    shift_max = params["future_target_window"][variable_apertura]
    tenencia_target = params["modelar_tenencia"]
    gc.collect()
    logger.info(f"Creando la variable {target} con la variable {variable_apertura}")
    logger.info(
        f"Usando {shift_max} periodos de rezago o los pronosticos cuentan con {shift_max} periodos de vida..."
    )
    if variable_apertura in df.columns:
        temp_df = df[[id_col, periodo_col, variable_apertura]].copy()
    else:
        logger.info(f"No esta la variable {variable_apertura}")
        temp_df = pd.DataFrame(columns=[id_col, periodo_col])
    order_months = sorted(temp_df[periodo_col].unique())
    result_all = pd.DataFrame()
    period_all = []
    for t in range(len(order_months) - shift_max):
        gc.collect()
        period_window = []
        df_temps = pd.DataFrame()
        for i in range(0, shift_max + 1, 1):
            period_window.append(order_months[t + i])
            tempo = temp_df[temp_df[periodo_col] == order_months[t + i]].rename(
                columns={variable_apertura: t + i}
            )
            if i == 0:
                df_temps = tempo.copy()
            else:
                df_temps = pd.merge(
                    df_temps, tempo[[id_col, t + i]], on=[id_col], how="left"
                )
        #        df_temps.append(temp_df[temp_df[periodo_col]==order_months[t+i]].rename(columns = {variable_apertura:t+i}))
        df_temps = df_temps.sort_values(by=id_col)
        pto_partida = 0
        print(
            f"Para el periodo {period_window[pto_partida]}, se analiza el comportamiento de la variable objetivo en {period_window[pto_partida+1:]}"
        )
        period_all = list(set(period_all + [period_window[0]]))
        # display(df_temps)
        # elimino las aperturas del momento 0 o las aperturas en el momento equivalente al periodo
        result = df_temps.set_index([id_col, periodo_col]).iloc[:, pto_partida + 1 :]
        # acumulo la cantidad de aerturas futuras
        result = result.sum(axis=1).to_frame()
        # ajustamos el nombre de la variable objetivo
        result = result.rename(columns={0: target}).reset_index()
        # display(result)
        # consolidamos los resultados
        result_all = pd.concat([result_all, result], axis=0)
    # unificamos la data
    temp_df = pd.merge(temp_df, result_all, on=[id_col, periodo_col], how="left")
    # los los meses en donde el periodo+shift se conoce
    temp_df1 = temp_df[temp_df[periodo_col].isin(period_all)]

    # considerando la ventanda de tiempo periodo:periodo+shift_max se pierden los ultimos shift_max periodos
    temp_df2 = temp_df[~temp_df[periodo_col].isin(period_all)]
    rest_periods = sorted(list(temp_df2[periodo_col].unique()))
    # se contara la cantidad de aperturas realizadas conocidas en los ultimos shift_max periodos
    temp_df3 = (
        pd.DataFrame()
    )  # aca guardaremos lo que sabemos que se aperturo en un futuro
    for t in range(shift_max):
        gc.collect()
        # filtramos del momento t hasta el fin de la data
        temp_df2_temp = temp_df2[temp_df2["periodo"] >= rest_periods[t]].copy()
        # ordenamos de la data nueva a la mas vieja
        temp_df2_temp = temp_df2_temp.sort_values(by=periodo_col, ascending=False)
        # tomamos el momento t mas viejo como 0 porque en t=t no tenemos aperturas presentes
        temp_df2_temp.loc[
            temp_df2_temp[temp_df2_temp["periodo"] == rest_periods[t]].index,
            variable_apertura,
        ] = 0
        # agrupamos las aperturas futuras entre t hasta el fin de los tiempos y acumulamos las aperturas
        temp_df2_temp.loc[:, target] = (
            temp_df2_temp.groupby([id_col]).cumsum()[variable_apertura].values.tolist()
        )
        # en la variable target tenemos las aperturas acumuladas entre t+1 hasta el fin de los tiempo ubicado en el periodo t
        moment_target = rest_periods[t]
        fin_target = rest_periods[-1]
        print(
            f"Se incluyen los clientes que tuvieron tenencia despues de {moment_target} y hasta {fin_target}"
        )
        temp_df2_temp = temp_df2_temp[temp_df2_temp[periodo_col] == moment_target].drop(
            variable_apertura, axis=1
        )
        # consolidamos las acumulaciones de aperturas
        temp_df3 = pd.concat([temp_df3, temp_df2_temp], axis=0)

    # ordenamos las aperturas del mas antiguo al reciente
    temp_df3 = temp_df3.sort_values(by=periodo_col, ascending=True)
    # En donde no tenemos aperturas acumuladas futuras no tenemos certeza si entre el fin de los tiempos o fin de la data
    # hasta el fin de la ventana shift max en realidad si se haga una apertura
    temp_df3 = temp_df3.replace(0, np.nan)
    # unificamos las aperturas de cada periodo con las aperutruas acumuladas futuras
    temp_df2 = pd.merge(
        temp_df2.drop(target, axis=1), temp_df3, on=[id_col, periodo_col], how="left"
    )
    # quitamos los clientes donde realmente no sabemos si van a aperturar porque no tenemos certeza desde el periodo en cuestion hasta periodo+shift_max
    temp_df2 = temp_df2[~temp_df2[target].isnull()]
    # se unifica la data completamente cierta contra la data recuperada:
    temp_df_final = pd.concat([temp_df1, temp_df2], axis=0)
    if tenencia_target is True:
        logger.info("Ajustando la cantidad de aperturas por tenencia de apertura ....")
        temp_df_final[target] = (temp_df_final[target] > 0).astype(int)
    logger.info(f"Finalizacion de creacion de variable {target}")
    gc.collect()
    return temp_df_final


# Funcion Auxiliar para crear todas las variables objetivo y se pegan a la data original
def create_targets_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea todas las variables target segun sus variablees de apertura respectivas marcando en x los resultados de apertura evidente en x+1,x+2,..,x+shift_max

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de entrada.
    params: Dict[str, Any]
        Diccionario de parámetros que contiene las variables necesarias.

    Returns
    -------
    pd.DataFrame: DataFrame del id, period_col, la variable_apertura y target.
    """
    id_col = params["id"]
    periodo_col = params["period_col"]
    dict_all_variables = params["todas_variables_apertura"]
    t = 0
    logger.info("Validando la existencia de todas las variables que se crearan: ")
    variables_insumos = list(dict_all_variables.keys())
    total_t = 0

    for variable_apertura in variables_insumos:
        try:
            df[variable_apertura]
            total_t = total_t + 1
        except KeyError:
            logger.error(
                f"La variable {variable_apertura} no está en los datos. El proceso se detendrá."
            )
            raise  # Esto detendrá el proceso

    logger.info(
        f"Se crearan {total_t} variables cada una sobre: {dict_all_variables.keys()}"
    )
    for variable_apertura, target in zip(
        dict_all_variables.keys(), dict_all_variables.values()
    ):
        gc.collect()
        variable_entrenamiento = create_targets(df, variable_apertura, target, params)
        if t == 0:
            all_variables = (
                variable_entrenamiento.drop(variable_apertura, axis=1)
                .copy()
                .set_index([id_col, periodo_col])
            )
        else:
            all_variables = pd.concat(
                [
                    all_variables,
                    variable_entrenamiento.drop(variable_apertura, axis=1).set_index(
                        [id_col, periodo_col]
                    ),
                ],
                axis=1,
            )
        t = t + 1
        # Eliminando los rezagos en la variable objetivo
        df.drop(variable_apertura, axis=1, inplace=True)
        gc.collect()
        logger.info(f"Llevamos {t} variables creadas de {total_t}")
    logger.info("Finalización de creacion de variables ...")
    # se le pegan las caracteristicas al lado izquierdo de la base que tiene las variables creadas
    # la data disminuye porque por ejemplo:
    # si un cliente es nuevo, apertura en t y entre t-shift_max a t-1 no habia informacion pues no se podria pronosticar
    # si por ejemplo en el pernultimo periodo y el ultimo periodo el clienete no aperturo nada y se programa para predecir a t+3:
    # entonces falta un mes donde podria aperturar pero coo ahi incertidumbre entonces el cliente desaparece
    logger.info("Agregando las variables creadas al dataframe...")
    resultado_final = pd.merge(df, all_variables, on=[id_col, periodo_col], how="right")
    logger.info("Variables creadas agregadas correctamente...")
    gc.collect()
    return resultado_final
