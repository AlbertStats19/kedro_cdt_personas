"""
This is a boilerplate pipeline 'model_output'
generated using Kedro 0.18.14
"""
# Fabrica peronsas model_output nodes
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict
import warnings
import data_bbog_integration_fabrica_personas.pipelines.raw.nodes as raw
from data_bbog_integration_fabrica_personas.pipelines.intermediate import (
    nodes as intermediate,
)
import data_bbog_integration_fabrica_personas.pipelines.primary.nodes as primary
import data_bbog_integration_fabrica_personas.pipelines.feature.nodes as feature
import data_bbog_integration_fabrica_personas.pipelines.model_selection.nodes as ms
import data_bbog_integration_fabrica_personas.pipelines.modelo_360.nodes as modelo_360


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Funciones Auxiliares para procesar la data
def prepare_data_primary(
    df: pd.DataFrame,
    info_save_select: Any,  # Dic con artefactos
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Prepara los datos para realizar el backtesting y predicciones del modelo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original en formato pandas.
    info_save_select: Any
        Modelos de prediccion junto con el ransformador o
        pipeline de escalado que se aplicará a las variables seleccionadas.
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    pd.DataFrame
        DataFrame procesado hasta raw o primary junto con el scaler respectivo.
    """
    logger.info("Iniciando la preparación de datos para el modelo...")
    # Parámetros importantes
    id_col = params["id"]
    # target_real = params["variable_apertura"]
    # target_modelo = params["target"]
    filtros_intermediate = params["re_ajuste_filtros"]
    filtros_primary = params["re_ajuste_filtros_negocio"]

    dataset_name = params["dataset_name"]
    periodo = params["period_col"]

    if isinstance(dataset_name, str) is True:
        dataset_name = int(dataset_name)
    elif isinstance(dataset_name, int) is True:
        dataset_name = int(dataset_name)
    elif isinstance(dataset_name, float) is True:
        dataset_name = int(dataset_name)
    else:
        raise ValueError('Parametro "dataset_name" no introducido correctamente')

    periods_all = df[periodo].drop_duplicates().values.tolist()
    logger.info("Validando la periodicidad de la base de datos")
    if len(periods_all) == 1:
        if periods_all[0] == dataset_name:
            logger.info("Base de datos con la periodicidad consistente")
        else:
            raise ValueError(
                "Corte de los datos incosistente."
                f"Periodo input deseado: {dataset_name}. "
                f"Periodo input cargado {periods_all[0]}"
            )
    else:
        warnings.warn(
            f"Estas prediciendo mas de 1 periodo: {periods_all}"
        )  # ,RuntimeWarning)

    if "Scaler" not in info_save_select["modelo_produccion"]:
        if info_save_select["modelo_produccion"]["type"] == "Ensamble":
            nodo_i = info_save_select["modelo_produccion"]["nodos_select"][0]
            logger.info(
                "Todos los modelos del Ensamble tienen el mismo procesamiento de datos"
            )
            scaler_transform = info_save_select[nodo_i]["Scaler"]
        else:
            raise ValueError("No identifica el Scaler del modelo...")
    else:
        scaler_transform = info_save_select["modelo_produccion"]["Scaler"]
        if info_save_select["modelo_produccion"]["type"] == "Ensamble":
            logger.info(
                "Todos los modelos del Ensamble tienen el mismo procesamiento de datos"
            )
    try:
        df = df.to_pandas()
        logger.info("f Paso la Data de formato polars a pandas")
    except Exception:
        pass
    try:
        # Validar la existencia de las columnas necesarias
        paso1 = raw.validar_columnas(df, params)
        hashvalue1 = paso1[id_col]
        if scaler_transform.reindex_OneHotEncoding.shape == (0, 1):
            # solo variables continuas
            paso4 = raw.values_to_null(paso1)
        else:
            # Convertir los nombres de las columnas a minúsculas
            paso2 = raw.convertir_a_minusculas(paso1, params)

            # Estandarizar los strings en el DataFrame
            paso3 = raw.standardize_strings(paso2, params)

            # Reemplazar ciertos valores por nulos según los parámetros
            paso4 = raw.values_to_null(paso3)
        # Cambiar los tipos de datos según la configuración especificada
        paso5 = raw.change_dtypes(paso4, params)

        # Validar que los identificadores únicos y los periodos sean correctos
        paso5[id_col] = hashvalue1
        paso6 = raw.validate_unique_id_period_pd(paso5, params)

        # Filtrar datos según el segmento especificado
        if filtros_intermediate["want"] is True:
            logger.info("Activando los filtros de intermediate ...")
            params["filter_segment"] = filtros_intermediate["filter_segment"]
            paso7 = intermediate.filter_data_segment_pd(paso6, params)
        else:
            logger.info("Desactivando los filtros de intermediate ...")
            paso7 = paso6.copy()

        # Filtrar datos relacionados con productos
        # logger.info("Filtrando datos de productos...")
        # paso8 = interm.filter_data_prod_pd(paso6, params)
        paso8 = paso7.copy()
        # Aplicar filtro de datos de negocio
        if filtros_primary["want"] is True:
            logger.info("Activando los filtros de primary ...")
            params["filter_business"] = filtros_primary["filter_business"]
            paso9 = primary.filter_business_data_pd(paso8, params)
        else:
            logger.info("Desactivando los filtros de primary ...")
            paso9 = paso8.copy()
    except Exception as e:
        logger.error(f"Error en la preparación de datos desde raw hasta primary: {e}")
        raise
    return paso9, scaler_transform


def prepare_data_model_input(
    paso9: pd.DataFrame,
    hashvalue1,
    feature_selected_list: pd.DataFrame,
    homologate_region_model,
    scaler_transform: Any,  # Dic con artefactos
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Prepara los datos desde Feature hasta Model.

    Parameters
    ----------
    paso9 : pd.DataFrame
        DataFrame hasta Primary en formato pandas.
    hashvalue1:
        pd.Series con los ids
    feature_selected_list: pd.DataFrame
        Lista de características seleccionadas para el modelo.
    homologate_region_model: Any
        Pickle con toda la parametrizacion de como
        homologar los datos antes del preprocesamiento
    scaler_transform: Any
        Transformador o pipeline de escalado que se
        aplicará a las variables seleccionadas.
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en
        el proceso de preparación de datos.

    Returns
    -------
    pd.DataFrame
        DataFrame procesado hasta desde Feature hasta model Input.
    """
    try:
        # Calcular nuevas variables necesarias para el modelo
        logger.info("Calculando nuevas variables para el modelo...")
        # paso10 = feature.calculate_new_variables_pd(paso8)
        paso10 = feature.calculate_new_variables_pd(paso9, params)

        # Homologar regiones en los datos
        logger.info("Homologando regiones...")
        paso11 = feature.homologate_region(paso10, homologate_region_model)

        # Seleccionar las variables que se utilizarán para el modelo
        logger.info("Seleccionando las características especificadas...")
        list_var = list(feature_selected_list["Feature"])
        # list_var = list(feature_selected_list.columns)
        for drop in [params["target"], params["id"], params["variable_apertura"]]:
            if drop in list_var:
                list_var.remove(drop)
        # paso12 = paso11[list_var]

        # Aplicar transformación o escalado a los datos seleccionados
        logger.info("Aplicando transformación a las variables seleccionadas...")
        paso13, y = scaler_transform.transform(
            paso11
        )  # inputes de los datos, hashvalue
        for col in scaler_transform.order_col_all:
            if col not in paso13.columns:
                logger.info(
                    f"La variable: '{col}' no esta en el procesamiento de datos.."
                )
        paso13 = paso13[scaler_transform.order_col_all]
        paso13 = pd.DataFrame(paso13, index=hashvalue1.index)

        logger.info("Preparación de datos completada exitosamente.")
    except Exception as e:
        logger.error(
            f"Error en la preparación de datos desde Feature hasta model_input: {e}"
        )
    return paso13


# Funcion Auxiliar para predecir
def predicciones_data(
    df_inputs: pd.DataFrame,
    info_save_select: Any,  # Dic con artefactos
    params: Dict[str, Any],
):
    """
    Realiza predicciones.

    Parameters
    ----------
    df_inputs : pd.DataFrame
        DataFrame de entrada que contiene las
        características para realizar las predicciones.
    info_save_select : Any
        Modelo ya entrenado que se utilizará para predecir.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye el id y la variable objetivo.

    Returns
    -------
    pd.DataFrame
        Array con las predicciones, las probabilidades y los hashvalue
    """
    # Parámetros
    id_col = params["id"]
    type_model = info_save_select["modelo_produccion"]["type"]
    if id_col in df_inputs:
        logger.info("Asegurese que los inputs esten ordenados como se entregaron ...")
        df_inputs = df_inputs.drop(columns=[id_col], axis=1)

    # inputs asegurando el mismo orden de entrenamiento y testeo
    # Realizar las predicciones
    logger.info("Realizando predicciones con el modelo entrenado...")
    if type_model == "Ensamble":
        logger.info("Es un Ensamble")
        lopps = list(
            range(0, len(info_save_select["modelo_produccion"]["nodos_select"]), 1)
        )
    elif type_model == "Models":
        logger.info("Es un Modelo de Models")
        lopps = list(range(0, 1, 1))
    else:
        raise ValueError("No identifica la llave con la que se optimizo el modelo...")
    # numero de obs a seleccionar
    if params["n_obs_filter"] <= 1:
        limit_ = int(df_inputs.shape[0] * params["n_obs_filter"])
    else:
        limit_ = params["n_obs_filter"]
    y_pred, y_proba, results_before = ms.forecast_probs(
        info_save_select, lopps, df_inputs, limit_, params
    )

    logger.info(f"Y_true_predict: {y_pred['y_pred'].sum()}. Want Select: {limit_}")
    y_pred = np.array(y_pred["y_pred"])
    y_proba = np.array(y_proba["y_p"])
    y_proba = np.nan_to_num(y_proba, nan=0)
    return y_pred, y_proba


def pre_calificar_base(
    df: pd.DataFrame,
    feature_selected_list: pd.DataFrame,
    homologate_region_model,
    info_save_select: Any,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Ejecuta todo el procesamiento de datos y retorna
     un DataFrame procesado junto con la base calificada

    Args:
        df: DataFrame
            DataFrame de entrada que será procesado.
        feature_selected_list: pd.DataFrame
            Lista de características seleccionadas para el modelo.
        homologate_region_model: Any
            Pickle con toda la parametrizacion de como homologar
            los datos antes del preprocesamiento
        info_save_select: Any
            Modelos entrenados junto con el Transformador o
            pipeline de escalado que se aplicará a las variables seleccionadas.
        params: Dict[str, Any]
            Diccionario de parámetros utilizados en el proceso de preparación de datos.
    Returns:
        DataFrame procesado con nuevas variables calculadas.
    """
    id_col = parameters["id"]
    fecha_ejecucion = parameters["fecha_ejecucion"]
    manejo_resultados = parameters["filtrar_y_prob_mayor_0"]

    logger.info("Iniciando preprocesamiento...")
    paso9, scaler_transform = prepare_data_primary(df, info_save_select, parameters)
    logger.info(
        f"Copiando y eliminando la columna {id_col}" " del dataframe de entrada..."
    )
    hashvalue1 = paso9[id_col]
    paso9 = paso9.drop(columns=[id_col], axis=1)

    paso13 = prepare_data_model_input(
        paso9,
        hashvalue1,
        feature_selected_list,
        homologate_region_model,
        scaler_transform,
        parameters,
    )

    # Aplica el orden de los inputs y predice
    logger.info("Iniciando el proceso de predicciones...")
    y_pred, y_proba = predicciones_data(paso13, info_save_select, parameters)

    base_calificada = pd.DataFrame(
        {id_col: np.array(hashvalue1), "y_pred": y_pred, "y_pred_proba": y_proba}
    )

    base_calificada = base_calificada.sort_values(by="y_pred_proba", ascending=False)
    col_names = base_calificada.columns.tolist()
    base_calificada["periodo"] = fecha_ejecucion
    base_calificada = base_calificada[["periodo"] + col_names]
    # volviendo el ID de la base calificada en mayuscula
    base_calificada[id_col] = base_calificada[id_col].apply(
        lambda x: x.upper() if isinstance(x, str) else x
    )
    # donde la prob es 0 entonces se pronostica 0
    base_calificada.index = list(range(base_calificada.shape[0]))
    # filtrnado los que enviamos como true por default
    base_calificada.loc[
        base_calificada[base_calificada["y_pred_proba"].round(4) == 0].index, "y_pred"
    ] = 0
    if manejo_resultados is True:
        logger.info('Filtrando los pronosticos "y_pred_proba" > 0 ')
        # tengo los params['n_obs_filter'] mas
        # propensos ajustados por los que la propension >0
        base_calificada = base_calificada[base_calificada["y_pred_proba"].round(4) > 0]
        base_calificada = base_calificada[base_calificada["y_pred"] == 1]
    else:
        msj = parameters["n_obs_filter"]
        if msj <= 1:
            msj = "el " + str(msj * 100) + "% del total de IDs"
        else:
            msj = str(msj) + " ids"
        logger.info(f'Se tiene toda la distribucion de "y_pred_proba" hasta {msj} ')

    logger.info(f"Numero de Pronosticos {base_calificada.shape[0]}")
    logger.info(f"Numero de predicciones True {(base_calificada['y_pred'] == 1).sum()}")
    logger.info(
        "Numero de observaciones con P(x>0) "
        f"{(base_calificada['y_pred_proba'] > 0).sum()}"
    )
    return paso13, base_calificada


def calificar_base(
    df: pd.DataFrame,
    feature_selected_list: pd.DataFrame,
    homologate_region_model: Any,
    info_save_select: Any,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Ejecuta la base de datos calificada (pronosticos)
    y la adecua para obtener todos las observaciones mas.

    Args:
        df: DataFrame
            DataFrame de entrada que será procesado.
        feature_selected_list: pd.DataFrame
            Lista de características seleccionadas para el modelo.
        homologate_region_model: Any
            Pickle con toda la parametrizacion de como homologar
            los datos antes del preprocesamiento
        info_save_select: Any
            Modelos entrenados junto con el Transformador o pipeline
            de escalado que se aplicará a las variables seleccionadas.
        params: Dict[str, Any]
            Diccionario de parámetros utilizados en el proceso de preparación de datos.
    Returns:
        DataFrame procesado con nuevas variables calculadas.
    """
    id_col = parameters["id"]
    variable_apertura = parameters["variable_apertura"]
    carpeta = parameters["vinculacion_productos"][variable_apertura]
    paso13, base_calificada = pre_calificar_base(
        df, feature_selected_list, homologate_region_model, info_save_select, parameters
    )
    if parameters["adjust_y_pred"]["want"]:
        n_true_forecast = parameters["refactor_backtesting"][
            "alcance_decil_monto_efect"
        ][carpeta]
        if n_true_forecast <= 1:
            logger.info(
                "Ajustando los y_pred == 1 y tomando las "
                f"{np.round(n_true_forecast*100, 2)}"
                "% de las observaciones mas probables como 1"
            )
            n_true_forecast = int(df.shape[0] * n_true_forecast)
        else:
            logger.info(
                f"Ajustando los y_pred == 1 y tomando las {n_true_forecast}"
                " observaciones mas probables como 1"
            )

        base_calificada.loc[base_calificada.iloc[n_true_forecast:].index, "y_pred"] = 0
        logger.info(
            f"Numero de predicciones True {(base_calificada['y_pred'] == 1).sum()}"
        )
    if parameters["add_anexos"] is True:
        otras_bd = modelo_360.anexos_campañas(df, parameters)
        base_calificada = pd.merge(base_calificada, otras_bd, on=id_col, how="left")
    logger.info(f"Se ha calificado la base con exito'{base_calificada.shape}'...")
    return base_calificada
