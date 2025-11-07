"""
Nodos de la capa backtesting
"""
import pandas as pd
import numpy as np
import bigframes.pandas as bpd

from typing import Dict, Any
import logging
from IPython.display import display

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
)
from sklearn.calibration import calibration_curve

import data_bbog_integration_fabrica_personas.pipelines.model_selection.nodes as ms
import data_bbog_integration_fabrica_personas.pipelines.model_output.nodes as mo

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# funciones axuliliares para cargar la dataF
# funciones auxiliares:
# Python function that loads a parquet from a path
def load_parquet(path):
    return pd.read_parquet(path)


# Python function that loads csv from a path
def load_csv(path):
    return pd.read_csv(path)


# modelo de pronosticos de contactabilidad
def create_curve_backtesting(output, n, params):
    """
    Toma la base de datos de los pronosticos y modela la contactabilidad bajo cierto
    niveles de contactabilidad

    Parameters
    ----------
    output: pd.DataFrame
        pd.DataFrame de los pronosticos junto con sus probabilidades de certeza
    n: np.int
        np.int asociado al numero total de la base de datos
    params:
        Dict de parametros

    Returns
    -------
        pd.DataFrame de los id sobre los cuales deberia contactarse segun el limite
        maximo de contactos
    """

    # niveles de contactabilidad
    n_tops = params["n_tops"]
    output2 = output.sort_values(by="y_pred_proba", ascending=False)
    output_vp1 = pd.DataFrame()
    for i in n_tops:
        j = i
        if i <= 1:
            j = int(n * i)
        output_temp = output2.iloc[:j]
        output_temp = output_temp[[params["id"]]].rename(columns={params["id"]: i})
        output_vp1 = pd.concat([output_vp1, output_temp], axis=1)
    output_vp1.index = range(output_vp1.shape[0])
    return output_vp1


# 1. Procesamiento de data
def prepare_data_pd(
    feature_selected_list: pd.DataFrame,
    homologate_region_model: Any,
    info_save_select: Any,  # Dic con artefactos
    params: Dict[str, Any],
):
    """
    Prepara los datos para realizar el backtesting y calcula las predicciones
    del modelo.

    Parameters
    ----------
    feature_selected_list: pd.DataFrame
        Lista de características seleccionadas para el modelo.

    homologate_region_model: Any
        Pickle con toda la parametrizacion de como homologar los datos antes
        del preprocesamiento

    info_save_select: Any
        Pickle con toda la informacion del modelo.

    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación
        de datos.

    Returns
    -------
    Pickle que contiene por mes:
        pd.DataFrame
            DataFrame final que incluye id, las variables escaladas.
        pd.DataFrame
            DataFrame con las predicciones realizadas
    """
    save_backtest = {}
    ### parametros generales
    ids = params["id"]

    ### parametros de entrenamiento/re-entrenamiento
    # rutas inputs para prediccion
    reads_df_input = params["rutas_inputs"]
    # otros parametros predictivos
    method_backtesting = params["n_obs_backtesting"]
    filtros_intermediate_backtesting = params["re_ajuste_filtros_backtesting"]
    filtros_primary_backtesting = params["re_ajuste_filtros_negocio_backtesting"]

    ### parametros de monitoreo
    monitoreo = params['monitoreo']

    # rutas de monitoreo    
    reads_df_monitoreo = monitoreo['rutas_inputs']
    
    # metodo de ejecucion
    monitoreo_mlops = monitoreo['want']

    if monitoreo_mlops==True:
        # backtesting a modo de monitoreo
        reads_df = reads_df_monitoreo
    else:
        reads_df = reads_df_input
        # re ajustando los filtros intermediate y primary de backtesting para la predicción
        params["re_ajuste_filtros"] = filtros_intermediate_backtesting
        params["re_ajuste_filtros_negocio"] = filtros_primary_backtesting
    # re-ajustando los filtros de prediccion
    if method_backtesting == "all":
        logger.info("Se hara el backtesting sobre todos los pronosticos")
        params["n_obs_filter"] = 1
    elif isinstance(method_backtesting, (float, int, np.float64, np.float32)):
        if method_backtesting <= 1:
            logger.info(
                f"Se hara el backtesting sobre el "
                f"{np.round(method_backtesting*100, 4)}% de los datos"
            )
        else:
            logger.info(f"Se hara el backtesting sobre {method_backtesting} datos")
        params["n_obs_filter"] = method_backtesting
    else:
        msg = params["n_obs_filter"]
        if msg <= 1:
            logger.info(
                "Por default se hara el backtesting sobre el "
                f"{np.round(msg*100, 4)}% de los datos"
            )
        else:
            logger.info(f"Por default se hara el backtesting sobre el {msg} datos")

    for t, dataset_name, df_temp in reads_df:
        logger.info(f"Data con rezago t-{t}")
        if monitoreo_mlops==True:
            logger.info(f"Periodo predicho a cargar : {dataset_name}")
        else:
            logger.info(f"Periodo utilizado para predecir: {dataset_name}")
        logger.info(f"Ruta: {df_temp}")
        params["dataset_name"] = dataset_name

        df_temp = load_parquet(df_temp)
        try:
            df_temp = df_temp.to_pandas()
        except Exception as e:
            logger.warning(f"Error converting to pandas: {str(e)}")
            pass
        save_backtest[t] = {}
        if monitoreo_mlops==True:
            base_calificada = df_temp.copy()
            base_calificada = base_calificada[['periodo',ids,'y_pred','y_pred_proba']]
            paso13 = pd.DataFrame(index= base_calificada.index.tolist())
            # generacion de mensajes
            logger.info(f"Numero de Pronosticos {base_calificada.shape[0]}")
            logger.info(f"Numero de predicciones True {(base_calificada['y_pred'] == 1).sum()}")
            logger.info(
                "Numero de observaciones con P(x>0) "
                f"{(base_calificada['y_pred_proba'] > 0).sum()}"
            )
            logger.info(f'Enlace de Pilotaje a MLops')
            n_true_forecast = params["n_obs_filter"]
            if n_true_forecast <= 1:
                n_true_forecast = int(base_calificada.shape[0] * n_true_forecast)
            base_calificada.loc[base_calificada.iloc[:n_true_forecast].index, "y_pred"] = 1
            logger.info(f"Numero de predicciones True {(base_calificada['y_pred'] == 1).sum()}")
        else:
            paso13, base_calificada = mo.pre_calificar_base(
                df_temp,
                feature_selected_list,
                homologate_region_model,
                info_save_select,
                params,
            )
        save_backtest[t]["contactabilidad"] = create_curve_backtesting(
            base_calificada, paso13.shape[0], params
        )
        save_backtest[t]["prediccion"] = base_calificada
        save_backtest[t]["prepare_data"] = paso13
        save_backtest[t]["dataset_name"] = dataset_name
        logger.info("------------------------------")
    logger.info("Iniciamos el proceso de combinacion de pronosticos...")
    all_probs = pd.DataFrame()
    if "Combined" in save_backtest.keys():
        del save_backtest["Combined"]

    rezagos = list(save_backtest.keys())
    cantidad_pronosticos_true = []
    for t in rezagos:
        realizar_predicciones_pd = save_backtest[t]["prediccion"]
        realizar_predicciones_pd = realizar_predicciones_pd.set_index(ids)
        cantidad_pronosticos_true.append(realizar_predicciones_pd["y_pred"].sum())
        for col in ["y_pred_proba"]:
            realizar_predicciones_pd_col = realizar_predicciones_pd[[col]]
            if t == rezagos[0]:
                realizar_predicciones_pd_col.columns = [
                    i + "_" + str(int(t)) for i in realizar_predicciones_pd_col.columns
                ]
                all_probs = realizar_predicciones_pd_col.copy()
            else:
                realizar_predicciones_pd_col.columns = [
                    i + "_" + str(int(t)) for i in realizar_predicciones_pd_col.columns
                ]
                all_probs = pd.concat([all_probs, realizar_predicciones_pd_col], axis=1)
    # combinando los resultados finales
    all_probs = all_probs.reset_index().set_index([ids]).fillna(0).max(axis=1)
    all_probs = all_probs.to_frame().sort_values(by=0, ascending=False)
    realizar_predicciones_window = all_probs.reset_index().rename(
        columns={0: "y_pred_proba"}
    )
    realizar_predicciones_window["y_pred"] = 0
    # numero de obs a seleccionar
    limit_ = np.min(cantidad_pronosticos_true)
    # pronosticando los 'limit_' mas probables
    realizar_predicciones_window.loc[:limit_, "y_pred"] = 1
    realizar_predicciones_window = realizar_predicciones_window[
        realizar_predicciones_window["y_pred"] == 1
    ]
    # guardandando la data
    logger.info("Guardando la combinacion de pronosticos...")
    t = "Combined"
    save_backtest[t] = {}
    save_backtest[t]["contactabilidad"] = create_curve_backtesting(
        realizar_predicciones_window, paso13.shape[0], params
    )
    save_backtest[t]["prediccion"] = realizar_predicciones_window
    save_backtest[t]["prepare_data"] = None
    save_backtest[t]["dataset_name"] = t
    llave_name = (
        info_save_select["modelo_produccion"]["model_name"]
        + ","
        + info_save_select["modelo_produccion"]["name_model"]
    )
    return save_backtest, llave_name

# Funcion Aux para calculo de deciles
def deciles_func(
    predicciones_pd, number=10, flexibility=True, msj="las probabilidades predichas"
):
    """
    Calcula de los deciles del array que reciba

    Parameters
    ----------
    predicciones_pd : pd.DataFrame
        Serie sobre la cual se calcularan los deciles

    number: np.int
        Numero de grupos a separar. Por Default 10

    flexibility: Bool
        Booleano que permite que identificar que tan flexible es la separacion
        de grupos. Por Default True

    msj: Str
        String asociado a la mensaje log sobre el cual se esta procesando
        la data. Por default el mensaje son las probabilidades predichas

    Returns
    -------
    np.Array
        Vector con los deciles respectivos a la base de datos
    """

    # Verificar el número de valores únicos
    logger.info(f"Calculando {number} grupos a partir de {msj}...")
    try:
        y_proba = predicciones_pd.astype(float).copy()
    except Exception as e:
        logger.warning(f"Error: {str(e)}")
        y_proba = predicciones_pd.copy()

    num_unique = pd.Series(y_proba).nunique()
    # Usar pd.qcut con duplicates='drop'
    try:
        exito = False
        for value in range(number, 2, -1):
            corte = min(num_unique, number)
            try:
                try:
                    deciles = pd.qcut(y_proba, corte, labels=False) + 1
                    exito = True
                except Exception as e:
                    logger.warning(f"Error: {str(e)}")
                    deciles = (
                        pd.qcut(y_proba, corte, labels=False, duplicates="drop") + 1
                    )
                    exito = True
            except ValueError as e:
                print(f"Error: {e}")
                deciles = pd.Series(
                    [1] * len(y_proba)
                )  # Asignar un valor predeterminado si ocurre un error
            finally:
                if exito is True:
                    break
    except Exception:
        try:
            deciles = pd.cut(y_proba, bins=number, labels=False) + 1
        except ValueError as e:
            print(f"Error: {e}")
            deciles = pd.Series([1] * len(y_proba))
    if flexibility is False:
        serie = pd.Series(y_proba)
        serie.index.name = "posicion"
        serie = serie.to_frame().reset_index()
        columnas = serie.columns.tolist()
        columnas.remove("posicion")
        serie = serie.sort_values(by=columnas, ascending=False)
        n_obs = serie.dropna().shape[0]
        grupos = int(n_obs / number)
        serie["Grupo"] = np.nan
        j = 0
        j_antes = 0
        for i in range(number, 0, -1):
            j = j + grupos
            if i == 1:
                j = j + grupos
            j = np.min([j, n_obs])
            serie.loc[serie.iloc[j_antes:j].index, "Grupo"] = i
            j_antes = j
        deciles = serie.sort_values(by="posicion")["Grupo"]
    n_deciles_temp = len(set(deciles))
    n_deciles_temp1 = pd.Series(deciles).drop_duplicates()
    n_deciles_temp = np.min([n_deciles_temp, n_deciles_temp1.shape[0]])
    logger.info(f"Cantidad de grupos: {n_deciles_temp}")
    if n_deciles_temp1.shape[0] < 15:
        logger.info(f"Grupos: {n_deciles_temp1.values}")
    return deciles


# 2. cruzar predicciones y generar la curva de backtesting


# funciones auxiliares:
# Graficas de las metricas en un solo mes o un solo corte
def graficar_backtesting(contactabilidad, t, n_x=1000):
    """
    Funcion que genera grafico de Aciertos vs % Aciertos x Cliente y
    Aciertos vs % Aciertos

    Parameters
    ----------
    contactabilidad: pd.DataFrame
        DataFrame con las variables:
            Modelo: Identificar algoritmos
            Aciertos, % Aciertos x Cliente, % Aciertos, N resultados
            tecnicos del modelo en un backtesting
    t: np.inf
        Asociado al rezago o corte del mes input usado para generar
        el pronositco.
    n_x: np.inf
        Asociado al denominador del grafico

    Returns
    -------
    None
    """
    if n_x == 1000:
        x_msj = "Numero de Clientes en Miles"
    elif n_x == 1:
        x_msj = "Numero de Clientes"
    else:
        x_msj = f"Numero de Clientes. Magnitudes en {n_x}"

    for tipo in contactabilidad["Tipo"].unique():
        contactabilidad_filt = contactabilidad[contactabilidad["Tipo"] == tipo]
        contactabilidad_filt = contactabilidad_filt[
            ~contactabilidad_filt["Aciertos"].isnull()
        ]
        title = f"Data t-{t}.Tipo: {tipo}."
        for graficos in [
            ["Aciertos", "% Aciertos x Cliente"],
            ["Aciertos", "% Aciertos"],
        ]:
            plt.figure(figsize=(15, 5))
            plt.style.use("ggplot")
            ax1 = plt.subplot(1, 1, 1)
            ax1.set_title(title, fontsize=15)  # +": " +save_str)
            ax1.set_xlabel(x_msj, fontsize=15)
            for f, eje in enumerate(graficos, start=1):
                x_ticks = contactabilidad_filt["N"]
                ejecutar = False
                if all(contactabilidad_filt[eje].isnull()) is False:
                    if f == 2:
                        ax1 = ax1.twinx()
                    sns.scatterplot(
                        data=contactabilidad_filt, x="N", y=eje, c="black", ax=ax1
                    )
                    lineplot = sns.lineplot(
                        data=contactabilidad_filt, x="N", y=eje, hue="Modelo", ax=ax1
                    )
                    ax1.set_ylabel(eje, fontsize=15)
                    if f == 1:
                        # ax1.set_yticks(contactabilidad_filt[eje].dropna(),contactabilidad_filt[eje].dropna().astype(int),fontsize=9,rotation=0)
                        handles, labels = lineplot.get_legend_handles_labels()
                        plt.legend().set_visible(False)
                        # y_ticks = ax1.get_yticks()
                    if f == 2:
                        ax1.set_xticks(
                            x_ticks.dropna().tolist(),
                            ((x_ticks / n_x).dropna().astype(int)).tolist(),
                            fontsize=9,
                            rotation=15,
                        )
                        # ax1.set_yticks(contactabilidad_filt[eje].dropna(),contactabilidad_filt[eje].dropna().round(2),fontsize=9,rotation=0)
                        if ejecutar is True:
                            handles, labels = lineplot.get_legend_handles_labels()
                else:
                    if f == 1:
                        ejecutar = True

            # handles, labels = ax1.get_legend_handles_labels()
            # Obtener los handles y labels
            ax1.legend(
                handles, labels, title="Modelo"
            )  # asegura que los legends sean iguales en cada grafico
            plt.tight_layout()
            plt.show()


# Funcion auxiliar:: Graficas de todas las curvas para todos los cortes
def plotear_curvas_rezago(data_process, parametros):
    """
    Funcion que me genera grafico de Aciertos vs % Aciertos x Cliente y
    Aciertos vs % Aciertos

    Parameters
    ----------
    data_process:
        Pickle que contiene en cada key el mes de rezago usado para
        pronosticar un corte de mes especifico. Adicionalmente, cada llave
        del diccionario contiene un dataframe con las metricas o resultados
        del backtesting

    Returns
    ----------
    None
    """

    n_x = parametros["x_dim_plot"]
    for t in data_process.keys():
        contactabilidad = data_process[t]["contactabilidad_curva"]
        contactabilidad = contactabilidad.loc[
            contactabilidad.select_dtypes(exclude=["object"])
            .dropna(how="all")
            .index.tolist()
        ]
        graficar_backtesting(contactabilidad, t, n_x)
        logger.info("------------------------------")

    nametag_model = ["Modelo"]
    backtesting_all = [data_process]
    consolidacion_tipo = {}
    for t, save_backtesting in enumerate(backtesting_all):
        for i in save_backtesting.keys():
            name = nametag_model[t] + " t-" + str(i)
            df_i = save_backtesting[i]["contactabilidad_curva"]  # .keys()
            df_i = df_i[~df_i["Aciertos"].isnull()]
            df_i = df_i.replace("Fabrica " + parametros["target"].split("_")[0], name)
            df_i = df_i[df_i["Modelo"] == name]
            for tipo in df_i["Tipo"].unique():
                df_ii = df_i[df_i["Tipo"] == tipo]
                df_ii = df_ii.set_index("COLUMN")  # df_ii = df_ii.set_index('N')
                if tipo not in consolidacion_tipo:
                    consolidacion_tipo[tipo] = {
                        "Aciertos": pd.DataFrame(),
                        "% Aciertos x Cliente": pd.DataFrame(),
                        "% Aciertos": pd.DataFrame(),
                    }
                for col in ["Aciertos", "% Aciertos x Cliente", "% Aciertos"]:
                    consolidacion_tipo[tipo][col] = pd.concat(
                        [
                            consolidacion_tipo[tipo][col],
                            df_ii[[col]].rename(columns={col: name}),
                        ],
                        axis=1,
                    )

    for col in list(consolidacion_tipo.keys()):
        for tipo in ["Aciertos", "% Aciertos x Cliente", "% Aciertos"]:
            if tipo == "Aciertos":
                graph = consolidacion_tipo[col][tipo].T / 1000
                msj = f"Backtesting {tipo} en Miles {col}"
            else:
                graph = consolidacion_tipo[col][tipo].T
                msj = f"Backtesting {tipo}  {col}"
            plt.figure(figsize=(15, 5))
            sns.heatmap(graph, annot=True, fmt="0.02f")
            plt.title(msj)
            plt.show()


# Funcion auxiliar: crear las metricas y hacer los plot de las curvas del backtesting
def generar_curvas(data_process, master_real, parametros):
    """
    Funcion que me genera las metricas de Aciertos vs % Aciertos x Cliente y
    Aciertos vs % Aciertos junto con el insumo para el modelo 360.

    Parameters
    ----------
    data_process:
        Pickle que contiene los pronosticos en cada key del diccionario.
        Entendiendo que cada key es un mes de rezago.

    master_real:
        pd.DataFrame que contiene los resultados reales del mes de corte
        que se evalua

    parametros:
        Dict

    Returns
    ----------
    Pickle que contiene las metricas de Aciertos vs % Aciertos x Cliente y
    Aciertos vs % Aciertos en cada key del diccionario en la ubicacion:

        res = generar_curvas(data_process,master_real,parametros)
        t = list(res.keys())[0] # algun rezago
        res[t]['contactabilidad_curva']
    """

    id_col = parametros["id"]
    apertura_variable = parametros["variable_apertura"]
    produc_name = apertura_variable.split("_")[0]
    names_loop = []

    # filtrando que sean unicos:
    tot = master_real.shape[0]
    master_real.index = range(master_real.shape[0])
    master_real = master_real.loc[
        master_real[[id_col, "periodo"]].drop_duplicates().index
    ]
    if tot != master_real.shape[0]:
        logger.info(f'Idenficamos {id_col} en el mismo "periodo" duplicados...')
        logger.info(
            f"Se eliminaron {tot - master_real.shape[0]} duplicados. "
            f"Esto puede alterar positiva o negativamente los resultados del modelo"
        )

    # añadiendo las variables con las que se medira el problema
    names_loop.append(["ID", apertura_variable + "_ID"])
    master_real[apertura_variable + "_ID"] = (
        master_real[apertura_variable] >= 1
    ).astype(int)
    names_loop.append(["Freq Producto", apertura_variable + "_Freq"])
    master_real[apertura_variable + "_Freq"] = master_real[apertura_variable]

    for t in data_process.keys():
        logger.info(f"Iniciando Rezago t-{t}..")
        contactabilidad = data_process[t]["contactabilidad"]
        bh_model = pd.DataFrame()  # best human model
        all_filt_res = []
        for ticket, medicion in names_loop:
            logger.info(
                f"Iniciando el calculo por {ticket} con la variable: {medicion}...."
            )
            for filt in contactabilidad.columns:
                n_backtest_fab = (
                    contactabilidad[[filt]]
                    .rename(columns={filt: id_col})
                    .replace("", np.nan)
                    .dropna()
                )
                # filtrando todos los que aperturan
                y_real_copy = master_real[master_real[medicion] >= 1].copy()
                y_real_copy = y_real_copy[[id_col, medicion]].copy()
                n_backtest = n_backtest_fab[[id_col]]
                if filt == contactabilidad.columns[-1]:
                    tipo_id = n_backtest_fab[id_col].str[:2].unique()
                    logger.info(f"Tipo de Hashvalue1: {tipo_id}")
                if bh_model.shape[0] != 0:
                    ml_filt = bh_model.iloc[: n_backtest.shape[0]]
                    ml_filt = ml_filt[[id_col]]
                else:
                    ml_filt = pd.DataFrame()
                all_tp = y_real_copy[medicion].sum()
                # if ticket == "ID":
                # all_tp = y_real_copy.shape[0]
                # else:
                # y_real_copy = y_real_copy.groupby(id_col).sum().reset_index()
                # all_tp = y_real_copy[id_col].sum()
                orden = ["Fabrica " + produc_name, "Best Human Model"]
                for tt, n_backtest in enumerate([n_backtest_fab, ml_filt]):
                    lista_result = []
                    if n_backtest.shape[0] != 0:
                        n_backtest_all = pd.merge(
                            n_backtest, y_real_copy, on=id_col, how="left"
                        ).fillna(0)
                        tp = n_backtest_all[medicion].sum()
                        n_shape = n_backtest_all.shape[0]
                    else:
                        tp = None
                        n_shape = None
                        filt = None
                    try:
                        metric = np.round(100 * tp / n_shape, 4)  # efectividad
                    except Exception:
                        metric = None
                    try:
                        metric2 = np.round(100 * tp / all_tp, 4)  # recall
                    except Exception:
                        metric2 = None
                    lista_result.append(tp)
                    lista_result.append(n_shape)
                    lista_result.append(filt)
                    lista_result.append(metric)
                    lista_result.append(metric2)
                    lista_result.append(orden[tt])
                    lista_result.append(ticket)
                    all_filt_res.append(lista_result)
        df_temp = pd.DataFrame(
            all_filt_res,
            columns=[
                "Aciertos",
                "N",
                "COLUMN",
                "% Aciertos x Cliente",
                "% Aciertos",
                "Modelo",
                "Tipo",
            ],
        )
        data_process[t]["contactabilidad_curva"] = df_temp

    plotear_curvas_rezago(data_process, parametros)
    return data_process


# Funcion Auxiliar para dejar el insumo del modelo 360 en cada retrazo del pronostico
def modelo_360_input(backesting_dictt, parameters):
    """
    Funcion que me genera en deciles el performance del modelo en 10 deciles.
    Donde determina la ganancia de los Aciertos o %Aciertos por contactabilidad
    u ordenamiento para un pronostico/backtsting especifico. Lo anterior lo hace
    en funcion de la configuracion de los parametros.

    Parameters
    ----------
    backesting_dict:
        pickle con todos la infomracion del backtesting
    parameters:
        Parametros relacionados a la forma como debe ejecutarse el codigo.

    Returns
    ----------
    Pickle del backtesting espefico en donde incluye los Deciles en donde
    performa mejor el modelo, la version suavizada de este resultado y su
    version categorizada para el modelo 360.
    """

    column_pend = parameters["column_pend"]
    tipo = parameters["tipo"]
    n_mult = parameters["n_mult"]
    produc_name = parameters["variable_apertura"].split("_")[0]
    try:
        df_temp = backesting_dictt["contactabilidad_curva"].copy()
        df_temp["N"] = df_temp["N"] / n_mult
        df_temp["xN"] = n_mult
        df_temp = df_temp[["N", "xN", "COLUMN", column_pend, "Modelo", "Tipo"]]
        # df_temp = df_temp[df_temp['COLUMN']
        # .isin([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])]
        # df_temp = df_temp.drop('COLUMN', axis = 1)
        df_temp = df_temp[df_temp["Modelo"] == "Fabrica " + produc_name]
        df_temp = df_temp[df_temp["Tipo"] == tipo]
        df_temp = df_temp.drop(["Modelo", "Tipo"], axis=1)
        df_temp = df_temp.sort_values(by="N", ascending=True)
        df_temp.index = range(df_temp.shape[0])

        df_temp["pend"] = df_temp[column_pend].diff() / df_temp["N"].diff()
        df_temp.loc[0, "pend"] = df_temp.loc[0, column_pend] / df_temp.loc[0, "N"]
        # df_temp.index = ['Muy Alto','Alto','Medio Alto','Medio','Medio Bajo'
        # ,'Medio Bajo','Bajo','Bajo','Muy Bajo','Muy Bajo']
        # df_temp.index.name = 'Categoria'
        # df_temp = df_temp.reset_index()
        df_temp["Decil Pend"] = pd.qcut(df_temp["pend"], q=10, labels=range(1, 11, 1))
        # funcion que separa en deciles segun una expanding window:
        df_temp["Decil Suavizado"] = df_temp["Decil Pend"].expanding().min()
        # decil categoria
        new_cortes = []
        for pos_t, v in enumerate(df_temp["Decil Suavizado"]):
            transfer = np.max([v, df_temp["Decil Suavizado"].iloc[pos_t]])
            if pos_t == 0:
                val = 10
            else:
                val = np.max([new_cortes[-1] - 1, transfer])
                val = np.min([new_cortes[-1], val])
            new_cortes.append(val)
        df_temp["Decil Categoria"] = new_cortes
    except Exception as e:
        # En caso de error, se captura y se imprime el error
        print(f"Ha ocurrido un error: {e}")
    return df_temp


# Funcion Auxiliar para dejar todos los insumo del
# modelo 360 en todos los retrazos del pronostico
def modelo_360_full(backesting_dict, parameters):
    """
    Funcion que me genera en deciles el performance del modelo en 10 deciles.
    Donde determina la ganancia de los Aciertos o %Aciertos por contactabilidad
    u ordenamiento para todos los rezagos de los pronosticos. Lo anterior lo
    hace en funcion de la configuracion de los parametros.

    Parameters
    ----------
    backesting_dict:
        pickle con todos la infomracion del backtesting
    parameters:
        Parametros relacionados a la forma como debe ejecutarse el codigo.

    Returns
    ----------
    Pickle del backtesting los diferentes backtesting o pronosticos en donde
    incluye que contiene los Deciles en donde performa mejor el modelo, la
    version suavizada de este resultado y su version categorizada para el
    modelo 360.
    """

    res = {}
    for rezago in backesting_dict.keys():
        res[rezago] = {}
        logger.info(f"Iniciando insumo modelo 360 t-{rezago}")
        try:
            res[rezago]["Discriminacion"] = modelo_360_input(
                backesting_dict[rezago], parameters
            )
        except Exception as e:
            print(f"Ha ocurrido un error: {e}")
            logger.info("No se calculara")
        finally:
            logger.info(f"Finalizando insumo modelo 360 t-{rezago}")
    return res
## Funcion Auxiliar para definir las columnas/observaciones que se quieren
## descargar desde bigquery
def filtros_query(ruta_data_real,params):
    filter_obs = ruta_data_real['filter_value']
    filter_cols = ruta_data_real['filter_column']
    logger.info(f'Iniciando la definicion de variables requeridas a extraer')
    cols_target = []
    if filter_cols['params'] == 'None':
        pass
    else:
        for col in filter_cols['params']:
           col_name = params[col]
           if col_name not in cols_target:
               logger.info(f'Columna a descargar: {col_name}')
               cols_target.append(col_name)

    if filter_cols['add_vars'] == 'None':
        pass
    else:
        for col_name in filter_cols['add_vars']:
           if col_name not in cols_target:
               logger.info(f'Columna a descargar: {col_name}')
               cols_target.append(col_name)
    logger.info(f'Variables a extraer: {cols_target}')
    logger.info(f'Estructurando las condiciones de filtro de observaciones')
    dictt_filt = {}
    if filter_obs['params'] == 'None':
        pass
    else:
        for col,value in zip(filter_obs['params'].keys(),filter_obs['params'].values()):
            value_query = []
            for val in value:
                print(val)
                if isinstance(val, int):
                    value_query.append(int(val))
                    try:
                        value_query.append(str(val))
                    except:
                        pass
                elif isinstance(val, float):
                    value_query.append(float(val))
                    try:
                        value_query.append(str(val))
                    except:
                        pass
                elif isinstance(val, str):
                    value_query.append(str(val))
                    try:
                        value_query.append(int(val))
                    except:
                        pass
                elif isinstance(val, object):
                    value_query.append(str(val))
                    try:
                        value_query.append(int(val))
                    except:
                        pass
                else:
                    pass
                if val not in value_query:
                    value_query.append(val)
            col_name = params[col]
            logger.info(f'Filtrando {value_query} en {col_name}')
            dictt_filt[col_name] = value_query
            if col_name not in list(dictt_filt.keys()):
                dictt_filt[col_name] = value_query
    logger.info(f'Finalizando los filtros sobre variables que ya estan definidas en los parametros')           
    if filter_obs['add_vars'] == 'None':
        pass
    else:
        for col_name,value in enumerate(zip(filter_obs['add_vars'].keys(),filter_obs['add_vars'].values())):
            value_query = []
            for val in value:
                if isinstance(val, int):
                    value_query.append(int(val))
                elif isinstance(val, float):
                    value_query.append(float(val))
                else:
                    pass
                try:
                    value_query.append(str(val))
                except:
                    pass
                if val not in value_query:
                    value_query.append(val)
            logger.info(f'Filtrando {value_query} en {col_name}')
            if col_name not in list(dictt_filt.keys()):
                dictt_filt[col_name] = value_query
    logger.info(f'Finalizando los filtros sobre variables que no estan definidas en los parametros')
    logger.info(f'Filtro de observaciones: {dictt_filt}')
    return dictt_filt,cols_target

# Funcion Auxiliar que descarga a memoria query de bigquery
def download_query(df_retail_required):

    df_retail_df = df_retail_required.to_pandas()
    return df_retail_df

# Funcion Auxiliar que ejecuta el query de bigquery
def generate_query(ruta_data_real,params):
    project = ruta_data_real['project']
    read_gbq = ruta_data_real['read_gbq']
    logger.info(f'Comenzando con la definicion de variables y observaciones requeridas a extraer...')
    obs_filt,cols_target = filtros_query(ruta_data_real,params)
    
    # Configura tu proyecto de GCP
    logger.info("Apuntando el proyecto de datos de Bigquery...")
    bpd.options.bigquery.project = project

    logger.info("Apuntando el dataset con los filtros de información necesaria...")
    # Carga la tabla como un DataFrame
    df_retail = bpd.read_gbq(read_gbq)

    for col_name,value_query in zip(obs_filt.keys(),obs_filt.values()):
        df_retail = df_retail[ df_retail[col_name].isin(value_query)]

    df_retail_required = df_retail[cols_target]
    ### realizando calculos extras sobre la base....
    logger.info(f'Iniciando calculos extras que se requieran')
    logger.info(f'Finalizando los calculos extras')
    logger.info('Iniciando la descarga de la base')
    df_retail_df = download_query(df_retail_required)
    logger.info('Finalizando la descarga de la base')
    return df_retail_df

# Funcion del 2. Pipeline
def combinar_predicciones_reales(
    save_backtesting: Any, llave_name: Any, params: Dict[str, Any]  # Dic con artefactos
):
    """
    Toma las predicciones dadas, calcula los deciles basándose en las
    probabilidades predichas, y junta la variable de interes real en el dataset
    de pronosticos. Adicionalmente guarda y ejecuta los insumos resultados del
    backtesting necesarios para el modelo 360.
    Ademas 'n' representa la cantidad de observaciones donde no se tiene
    referencia de los datos y por ende no es medible

    Parameters
    ----------
    save_backtesting : Pickle
        Contiene la data para las predicciones y los pronosticos.
    llave_name : Str
        String que contempla el nombre del modelo
    params : Dict[str, Any]
        Diccionario de parámetros que incluye el id y la variable objetivo.

    Returns
    -------
    Pickle
        Pickle que contiene por mes toda la informacion respectiva.
        Pickle que contiene los resumenes del resultado del modelo 360.
    """

    # Parámetros
    id_col = params["id"] 
    target_real = params["variable_apertura"]

    # Ajustes data real
    ruta_output = params["ruta_output"]
    metodo = ruta_output['metodo']
    logger.info("Cargando la base de datos con los resultados reales...")

    # extraccion de bases de datos
    if metodo == 's3': 
        ruta_data_real = ruta_output[metodo]
        logger.info("Cargando la base de datos con los resultados reales desde s3...")
        logger.info(f"{ruta_data_real}")
        df2 = load_parquet(ruta_data_real)
        try:
            df2 = df2.to_pandas()
        except Exception:
            pass
    else:
        ruta_data_real = ruta_output[metodo]
        df2 = generate_query(ruta_data_real, params)
    #display(df2.head())
    df3 = df2.reset_index()
    if params["adj_real_target"] is True:
        logger.info(
            "Ajustando la variable objetivo real. "
            f"De modo que {target_real}>0 entonces es 1 y si no es 0"
        )
    
        df3[target_real] = (df3[target_real] > 0).astype(int)
    #display(df3.head())

    save_backtesting_temp = save_backtesting.copy()
    logger.info(
        "Iniciando el cruce de informacion entre "
        "los pronosticos y los resultados reales"
    )
    for t in save_backtesting_temp.keys():
        realizar_predicciones_pd = save_backtesting_temp[t]["prediccion"]
        hashvalue1 = realizar_predicciones_pd[id_col]
        # Metrica de calidad de datos
        df4 = df3[df3[id_col].isin(list(set(hashvalue1.tolist())))]
        #display(df4.head())
        #display(hashvalue1.head())
        n = df3.shape[0] - df4.shape[0]
        logger.info(
            f"Encontramos {n} personas con pronostico y "
            f"sin la apertura real. Datos con el rezago: {t}"
        )
        try:
            df4 = df4.rename(columns={target_real: "y_real"})
        except Exception:
            pass
        df5 = df4[[id_col, "y_real"]]
        #display(df5[df5[id_col].isin(['CC1070588085','CC40305180','CC17116673','CC1098638701','CC1102883460'])])
        #display(df5.head())
        realizar_predicciones_pd = pd.merge(
            realizar_predicciones_pd, df5, on=id_col, how="left"
        )
        display(realizar_predicciones_pd[realizar_predicciones_pd[id_col].isin(['CC1070588085','CC40305180','CC17116673','CC1098638701','CC1102883460'])])
        display(realizar_predicciones_pd.head())
        realizar_predicciones_pd["model_name"] = llave_name
        # Calcular la curva de calibración
        logger.info(
            "Calculamos los insumos para tener la "
            f"curva de calibracion del modelo para el rezago {t}..."
        )
        realizar_predicciones_pd1 = realizar_predicciones_pd[
            ~realizar_predicciones_pd["y_real"].isnull()
        ]
        if realizar_predicciones_pd1["y_pred_proba"].max() > 1:
            prob_true, prob_pred = calibration_curve(
                realizar_predicciones_pd1["y_real"],
                realizar_predicciones_pd1["y_pred_proba"] / 100,
                n_bins=10,
            )
        else:
            prob_true, prob_pred = calibration_curve(
                realizar_predicciones_pd1["y_real"],
                realizar_predicciones_pd1["y_pred_proba"],
                n_bins=10,
            )
        deciles = deciles_func(realizar_predicciones_pd["y_pred_proba"])
        realizar_predicciones_pd["decil_apertura"] = deciles
        realizar_predicciones_pd = realizar_predicciones_pd[
            [id_col, "y_real", "y_pred", "y_pred_proba", "decil_apertura", "model_name"]
        ]
        logger.info(f"Guardamos los resultados para el rezago {t}...")
        save_backtesting_temp[t]["prediccion"] = realizar_predicciones_pd
        save_backtesting_temp[t]["n_observables"] = n
        save_backtesting_temp[t]["prob_true"] = prob_true
        save_backtesting_temp[t]["prob_pred"] = prob_pred
        logger.info("------------------------------")

    save_backtesting_temp = generar_curvas(save_backtesting_temp, df2, params)

    logger.info("Reduciendo memoria en el archivo...")
    for t in save_backtesting_temp.keys():
        for remove in ["prepare_data", "contactabilidad"]:
            if remove in save_backtesting_temp[t].keys():
                del save_backtesting_temp[t][remove]
    logger.info("Optimizacion realizada...")
    logger.info("Generando los insumos del modelo 360...")
    insumo_model_360 = modelo_360_full(save_backtesting_temp, params)
    return save_backtesting_temp, insumo_model_360


# 3. Calcular metricas
# funcion auxiliar para ejecutar por mes
def generate_metrics_pd(
    df: pd.DataFrame, n, params: Dict[str, Any], msg
) -> pd.DataFrame:
    """
    Evalúa las predicciones de clasificación binaria de un DataFrame y retorna
    un DataFrame con varias métricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas ['hashvalue1', 'y_real', 'y_pred'].

    n: np.float, np. int
        Numero de observaciones con pronostico y sin la apertura real.

    params : Dict[str, Any]
        Diccionario que contiene el nombre del modelo, tipo de modelo y
        nombre del conjunto de datos.

    msg: str
        Mensaje asociado al tipo de base de datos df que recibe para
        identificar que se esta procesando.

    Returns
    -------
    pd.DataFrame
        DataFrame con las métricas de evaluación y la marca de tiempo en
        minutos, con valores formateados a 4 decimales.
    """

    # Parámetros
    model_name = params["target"].split("_")[0].upper()
    model_type = df["model_name"].iloc[0]
    dataset_name = params["dataset_name"]
    top_n = params["n_obs_filter_select"]
    if top_n <= 1:
        top_n = int(df.shape[0] * top_n)
    else:
        top_n = top_n
    logger.info(f"N_top: {top_n}")
    try:
        ms.generate_plots(df, 10, "Data Backtesting: " + str(msg))
    except Exception:
        pass
    # extraigo las etiquetas de predicciones reales que no se conocen
    filtered__without_real = df[df["y_real"].isna()]
    no_medible = filtered__without_real["y_pred"].value_counts().to_dict()
    key_map = {0: "No medible 0", 1: "No medible 1"}
    no_medible = {key_map[k]: v for k, v in no_medible.items()}
    no_medible1 = {
        k + " %": np.round(100 * v / df.shape[0], 4) for k, v in no_medible.items()
    }
    no_medible = (
        no_medible | no_medible1 | {"Pronosticos sin el hashvalue en la data real": n}
    )
    logger.info("Escenarios no medibles: {:<50}{}".format("", no_medible))
    # Extraer las etiquetas reales y las predicciones del DataFrame
    filtered_df = df.dropna(subset=["y_real"])
    y = filtered_df["y_real"]
    y_pred = filtered_df["y_pred"]

    # Obtener la marca de tiempo actual en minutos
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info(f"Marca de tiempo actual: {timestamp}")

    # backtesting
    try:
        backtest_metric = np.round(
            ms.backtesting_top_probabilities(
                filtered_df["y_real"].values, filtered_df["y_pred_proba"].values, top_n
            )
            * 100,
            4,
        )
    except Exception:
        backtest_metric = None

    view_metrics = pd.DataFrame()

    labels = [1, 0]
    precision, recall, f_score, true_sum = ms.precision_recall_fscore_support_fabrica(
        filtered_df["y_real"].values,
        filtered_df["y_pred_proba"].values,
        n_top=top_n,
        labels=labels,  # [0,1]
        metric_calcs="all",
    )
    index_name = ["Precision n_top", "recall n_top", "f1-score n_top", "true_sum n_top"]
    # conciliando metricas top
    temp2 = pd.DataFrame(["N n_top", top_n, "1", "n_top"])
    temp2.index = ["metric_name", "value", "class_name", "metric_type"]
    temp2 = temp2.T
    view_metrics = pd.concat([view_metrics, temp2], axis=0)

    temp2 = pd.DataFrame(["N all", df.shape[0], "dim shape", "dim shape"])
    temp2.index = ["metric_name", "value", "class_name", "metric_type"]
    temp2 = temp2.T
    view_metrics = pd.concat([view_metrics, temp2], axis=0)

    temp2 = pd.DataFrame(["backtest_metric n_top", backtest_metric, "1.0", "n_top"])
    temp2.index = ["metric_name", "value", "class_name", "metric_type"]
    temp2 = temp2.T
    view_metrics = pd.concat([view_metrics, temp2], axis=0)

    for t, index in enumerate([precision, recall, f_score, true_sum]):
        if t == 3:
            temp = pd.DataFrame(
                index, index=[index_name[t]] * len(labels), columns=["value"]
            )
        else:
            temp = pd.DataFrame(
                np.round(index * 100, 4),
                index=[index_name[t]] * len(labels),
                columns=["value"],
            )
        temp.index.name = "metric_name"
        temp.reset_index(inplace=True)
        temp["class_name"] = labels
        temp["metric_type"] = "n_top"
        view_metrics = pd.concat([view_metrics, temp], axis=0)

    view_metrics["timestamp"] = timestamp
    view_metrics["model_name"] = model_name
    view_metrics["model_type"] = model_type
    view_metrics["dataset_name"] = dataset_name

    view_metrics = view_metrics[
        [
            "timestamp",
            "model_name",
            "model_type",
            "dataset_name",
            "metric_name",
            "metric_type",
            "class_name",
            "value",
        ]
    ]
    view_metrics.index = list(range(view_metrics.shape[0]))
    logger.info("Metricas calculadas en %")
    logger.info("Metricas N_top de Backtesting: ")
    logger.info(
        view_metrics[
            view_metrics["class_name"].isin(["1.0", "1", 1, 1.0, "dim shape"])
        ].drop(["timestamp", "metric_type", "class_name"], axis=1)
    )

    # Calcular la matriz de confusión y desglosar sus componentes
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    logger.info(
        "Matriz de confusión calculada toda la data: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )
    # Calcular las métricas generales
    logger.info("Calculando metricas generales..")

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
    overall_metrics = no_medible | overall_metrics
    # Formatear los valores numéricos a 4 decimales
    # overall_metrics = {k: f"{v:.4f}" if isinstance(v, (float, np.float64))
    # else v for k, v in overall_metrics.items()}

    # Formatear los valores numéricos a 4 decimales sin convertir a string
    overall_metrics = {
        k: round(v, 4) if isinstance(v, (float, np.float64)) else v
        for k, v in overall_metrics.items()
    }
    logger.info("Ok metricas generales")

    # Crear un DataFrame con las métricas generales
    overall_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "model_name": model_name,
                "model_type": model_type,
                "dataset_name": dataset_name,
                "metric_name": metric,
                "metric_type": "overall metric",
                "class_name": "overall",
                "value": value,
            }
            for metric, value in overall_metrics.items()
        ]
    )

    # Generar el reporte de clasificación
    report = classification_report(y, y_pred, output_dict=True)
    logger.info("Ok Reporte de clasificación")

    # Crear un DataFrame con las métricas por clase
    class_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "model_name": model_name,
                "model_type": model_type,
                "dataset_name": dataset_name,
                "metric_name": metric,
                "metric_type": "class metric",
                "class_name": class_name,
                "value": round(value * 100, 4)
                if metric != "support" and isinstance(value, (float, np.float64))
                else value
                # 'value': f"{value * 100:.4f}" if metric != 'support' and
                # isinstance(value, (float, np.float64)) else f"{value:.0f}"
            }
            for class_name, metrics in report.items()
            if class_name not in ["accuracy", "macro avg", "weighted avg"]
            for metric, value in metrics.items()
        ]
    )

    # Combinar las métricas generales y las métricas por clase
    combined_df = pd.concat([view_metrics, overall_df, class_df], ignore_index=True)
    logger.info("Ok Reporte completo")

    # Reordenar las columnas para que la columna de timestamp esté al inicio
    column_order = ["timestamp"] + [
        col for col in combined_df.columns if col != "timestamp"
    ]

    result_df = combined_df[column_order]

    # Registrar la finalización del proceso de evaluación
    logger.info(
        f"Evaluación completada. Dimensión del DataFrame resultante: {result_df.shape}"
    )
    return result_df


# 3. Calcular metricas: funcion para ejecutar todos los meses
def generate_metrics_all(save_backtesting_temp: Any, parametros: Dict[str, Any]):
    for t in save_backtesting_temp.keys():
        logger.info(f"CALCULOS: {t}")
        realizar_predicciones_pd = save_backtesting_temp[t]["prediccion"]
        n = save_backtesting_temp[t]["n_observables"]
        parametros["dataset_name"] = save_backtesting_temp[t]["dataset_name"]
        result_df = generate_metrics_pd(realizar_predicciones_pd, n, parametros, str(t))
        save_backtesting_temp[t]["metrics"] = result_df
        logger.info("------------------------------")
    for t in save_backtesting_temp.keys():
        for remove in ["n_observables"]:
            if remove in save_backtesting_temp[t].keys():
                del save_backtesting_temp[t][remove]
    return save_backtesting_temp


# 4. Funcion Axuliar: Analisis KS
# Aplicar el formato de mapa de calor
def color_scale(val):
    color = plt.cm.viridis(val / 100)  # Usar colormap viridis
    return (
        f"background-color: rgba("
        f"{int(color[0]*255)}, "
        f"{int(color[1]*255)}, "
        f"{int(color[2]*255)}, "
        f"{color[3]})"
    )


def ks_analysis_pd(backtesting: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza un análisis KS (Kolmogorov-Smirnov) basado en los deciles de
    probabilidad predicha y aperturas reales.

    Parameters
    ----------
    backtesting : pd.DataFrame
        DataFrame que contiene las predicciones, probabilidades y resultados
        reales con columnas ['decil_apertura', 'y_pred_proba', 'y_real'].

    Returns
    -------
    pd.DataFrame
        DataFrame con las estadísticas por decil, incluyendo tasa de
        aperturas, distribución acumulada, aperturas predichas y el
        estadístico KS.
    """

    logger.info("Iniciando análisis KS por deciles de probabilidad...")

    try:
        # Agrupar por decil de apertura para obtener el
        # mínimo, máximo y total de clientes
        # logger.info("Agrupando datos por decil de apertura
        # para calcular estadísticas básicas...")
        grouped = (
            backtesting.groupby("decil_apertura")["y_pred_proba"]
            .agg(["min", "max", "count"])
            .reset_index()
        )

        # Renombrar las columnas para mayor claridad
        grouped.columns = ["decil", "prob_min", "prob_max", "total_clientes"]

        # Agrupar por decil para calcular las aperturas reales
        # logger.info("Calculando el total de aperturas reales por decil...")
        total_aperturas = (
            backtesting[backtesting["y_real"] == 1]
            .groupby("decil_apertura")["y_real"]
            .count()
            .reset_index()
        )
        total_aperturas.columns = ["decil", "total_aperturas"]

        # Agrupar por decil para calcular las aperturas predichas
        # logger.info("Calculando el total de aperturas predichas por decil...")

        # Fusionar los resultados con el DataFrame original grouped
        # logger.info("Fusionando las estadísticas de aperturas reales con
        # las estadísticas generales por decil...")

        grouped = grouped.merge(total_aperturas, on="decil", how="left")
        # Calcular la tasa de aperturas
        # logger.info("Calculando la tasa de aperturas por decil...")
        grouped["tasa_aperturas"] = (
            grouped["total_aperturas"] / grouped["total_clientes"]
        )
        # Ordenar por decil descendente
        # logger.info("Ordenando los datos por decil en orden descendente...")
        grouped = grouped.sort_values(by="decil", ascending=False)

        # Calcular la distribución acumulada de aperturas observadas
        # logger.info("Calculando la distribución acumulada de aperturas observadas...")
        grouped["cum_aperturas"] = (
            grouped["total_aperturas"].cumsum() / grouped["total_aperturas"].sum()
        )

        # Calcular la distribución acumulada de las
        # probabilidades esperadas (clientes por decil)
        # logger.info("Calculando la distribución acumulada
        # de probabilidades esperadas...")
        grouped["cum_prob_esperadas"] = (
            grouped["total_clientes"].cumsum() / grouped["total_clientes"].sum()
        )

        # Calcular el estadístico KS para cada decil
        logger.info("Calculando el estadístico KS para cada decil...")
        grouped["ks_stat"] = np.abs(
            grouped["cum_aperturas"] - grouped["cum_prob_esperadas"]
        )
        # Restablecer el índice para incluir el decil como columna
        # logger.info("Restableciendo el índice del DataFrame...")
        grouped.reset_index(drop=True, inplace=True)

        # logger.info("Calculando el total de aperturas reales por decil...")
        try:
            total_aperturas_predichas = (
                backtesting[backtesting["y_pred"] == 1]
                .groupby("decil_apertura")["y_pred"]
                .count()
                .reset_index()
            )
            total_aperturas_predichas.columns = ["decil", "total_aperturas_predichas"]
            # Agrupar por decil para calcular las aperturas predichas
            # logger.info("Calculando el total de aperturas predichas por decil...")
            # Fusionar los resultados con el DataFrame original grouped
            # logger.info("Fusionando las estadísticas de aperturas reales
            # con las estadísticas generales por decil...")
            grouped = grouped.merge(total_aperturas_predichas, on="decil", how="left")
            grouped["tasa_aperturas_predichas"] = (
                grouped["total_aperturas_predichas"] / grouped["total_clientes"]
            )
        except Exception:
            pass
        # Reordenar las columnas para que el decil esté al inicio
        # logger.info("Reordenando las columnas del DataFrame...")
        grouped = grouped[
            [
                "decil",
                "prob_min",
                "prob_max",
                "total_clientes",
                "total_aperturas",
                "tasa_aperturas",
                "cum_aperturas",
                "cum_prob_esperadas",
                "ks_stat",
            ]
        ]
        decil_congruente = (
            (grouped.set_index("decil")["ks_stat"].diff() > 0)
            .replace(False, np.nan)
            .dropna()
            .index
        )
        logger.info(f"Deciles donde el ks aumenta:  {list(decil_congruente)}")

        for col in [
            "prob_min",
            "prob_max",
            "tasa_aperturas",
            "cum_aperturas",
            "cum_prob_esperadas",
            "ks_stat",
        ]:
            grouped[col] = (grouped[col]).apply(
                lambda x: float("{0:.2f}".format(x * 100))
            )

        return grouped

    except Exception as e:
        logger.error(f"Error durante el análisis KS: {e}")
        raise


# 4. Algoritmo KS: funcion para ejecutar todos los meses
def generate_ks_all(save_backtesting_temp: Any, parametros: Dict[str, Any]):
    """
    Realiza un análisis KS (Kolmogorov-Smirnov) basado en los deciles de
    probabilidad predicha y aperturas reales.

    Parameters
    ----------
    save_backtesting_temp : Dict
        Pickle en formato diccionario que comprende en cada llave la
        'prediccion' del modelo en un formato DataFrame y ejecuta el
        algoritmo KS

    Returns
    -------
    Dict
        Pickle que contiene las predicciones junto con el algoritmo ks por
        cada prediccion o rezago.
    """

    for t in save_backtesting_temp.keys():
        logger.info(f"CALCULOS: {t}")
        realizar_predicciones_pd = save_backtesting_temp[t]["prediccion"]
        grouped = ks_analysis_pd(realizar_predicciones_pd)
        save_backtesting_temp[t]["ks"] = grouped
        # try:
        #     ks_analysis_ = grouped.style.applymap(color_scale,
        # subset=['cum_aperturas','ks_stat'])
        # except:
        ks_analysis_ = grouped
        # finally:
        display(ks_analysis_)
        logger.info("------------------------------")
    return save_backtesting_temp
