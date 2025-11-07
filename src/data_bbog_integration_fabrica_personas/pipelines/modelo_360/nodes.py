"""
Nodos de la capa modelo_360
"""
import s3fs
import warnings
import pickle

from scipy import stats
import pandas as pd
import numpy as np
import logging
from IPython.display import display

import data_bbog_integration_fabrica_personas.pipelines.feature.nodes as feature
import data_bbog_integration_fabrica_personas.pipelines.backtesting.nodes as backtesting

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# funciones auxiliares:
def asignar_categoria(row, parameters):
    """
    Segmenta o estratifica al cliente segun los ingresos brutos mensuales de la variable "vlr_bru_mes"

    Parameters
    ----------
    row : pd.DataFrame
        DataFrame original en formato pandas con la columna "vlr_bru_mes"
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    array asociado a la estratificacion de ingreso mensual
    """
    try:
        row = row.to_pandas()
    except Exception:
        pass

    valor = row["vlr_ing_bru_mes"]
    valor.index = list(range(row.shape[0]))
    salario = parameters["smmlv"]
    try:
        valor = valor.astype(float)
    except (ValueError, TypeError):
        return None
    if valor is None or all(valor.isnull()):
        return None

    result = np.array([4] * valor.shape[0])
    result[valor.index[valor.isnull()].tolist()] = 0
    result[valor.index[valor == None].tolist()] = 0  # noqa: E711
    result[valor.index[valor <= salario].tolist()] = 1

    cond = valor.index[(4 * salario) <= valor].tolist()
    cond = cond + valor.index[valor <= (10 * salario)].tolist()
    cond = list(set(cond))
    result[cond] = 2

    result[valor.index[(valor > (10 * salario))].tolist()] = 3
    return result


# Funcion auxiliar al  Primer Pipeline
def anexos_campañas(inputs_df, params):
    """
    Funcion auxiliar que calcula las variables requeridas para el formato de entrega a campañas

    Parameters
    ----------
    inputs_df : pd.DataFrame
        DataFrame original sobre el cual se realizo los pronosticos de la data
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
        Variables complementarias a los pronosticos
    """
    ids = params["id"]
    logger.info(
        "Iniciando con los calculos de las variables anexas a la entrega del modelo"
    )
    params_want_homologate = {}
    variables_target = ["region"]
    for col in params["homologacion_x_variable"].keys():
        if params["homologacion_x_variable"][col]["nombre"] in variables_target:
            params_want_homologate[col] = params["homologacion_x_variable"][col]

    necesary_columns = ["periodo", ids, "vlr_ing_bru_mes"]
    for col in params_want_homologate.keys():
        if col not in necesary_columns:
            necesary_columns.append(params_want_homologate[col]["insumo"])
    print(necesary_columns)
    inputs_df = inputs_df[necesary_columns]
    logger.info(f"Mes insumo de los pronosticos: {inputs_df['periodo'].iloc[0]}")
    inputs_df = feature.homologate_region(inputs_df, params_want_homologate)
    msj = "los ingresos brutos mensuales"
    decil_ingreso = backtesting.deciles_func(
        predicciones_pd=inputs_df["vlr_ing_bru_mes"],
        number=10,
        flexibility=False,
        msj=msj,
    )
    inputs_df["decil_ingreso"] = decil_ingreso.values
    inputs_df["categoria"] = asignar_categoria(inputs_df, params)
    inputs_df["categoria"] = inputs_df["categoria"].replace(0, None)
    inputs_df = inputs_df[[ids, "region", "decil_ingreso", "categoria"]]
    logger.info(f"Deciles vinculados al ingreso {inputs_df['decil_ingreso'].unique()}")
    return inputs_df


# 1. Primer pipeline: Extraer las variables adicionales del modelo 360.
def anexos_modelo_360(params):
    """
    Extrae las variables requeridas sobre la base que se parametriza en los params para complementar el modelo 360

    Parameters
    ----------
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    Variables añadidas al modelo 360 como complemento junto con la base de datos de las cantidades que tienen los clientes
    """
    ruta = params["mes_input"]
    ids = params["id"]
    variables_extras = params["variables_insumo_homologar"]
    if ruta[-3:].lower() == "csv":
        try:
            inputs_df = backtesting.load_csv(ruta)
        except ValueError as e:
            logger.info(e)
            logger.info("No carga archivo csv")
    else:
        try:
            inputs_df = backtesting.load_parquet(ruta)
            try:
                inputs_df = inputs_df.to_pandas()
            except Exception:
                pass
        except ValueError as e:
            logger.info(e)
            logger.info("No carga archivo parquet")

    clientes_lib = inputs_df[[ids] + variables_extras]
    inputs_df = anexos_campañas(inputs_df, params)
    return (inputs_df, clientes_lib)


# 2. Segundo pipeline: Cargar los pronosticos.
def cargar_bases(clientes_lib, params):
    """
    Carga los pronosticos de los productos de forma individual

    Parameters
    ----------
    total_clientes: Int
        Numero total de clientes en el corte de la consolidacion del modelo
    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    pd.DataFrame asociado a los pronosticos de la fabrica
    """
    # parametros
    mes = params["mes_vig"]
    ids = params["id"]

    ruta_base = params["ruta_base"]
    archivo_nombre = params["archivo_nombre"]
    productos = params["productos_carpeta"]

    homolog = params["homologacion_360"]
    metodo_consolidacion = params["desear_todos_id_x_archivo"]

    pickle_backtest_all = params["pickle_backtest_all"]
    rezagos_predict = params["rezagos_predict"]
    refactor_backtesting = params["refactor_backtesting"]
    homolog_decil_etiquetas = params["homolog_decil_etiquetas"]

    probs_normalizadas = params["reprocesar_probs_norm"]
    metodo_estandarizar_probs = params["metodo_estandarizar_probs"]
    multiplicar_recall_probs = params["multiplicar_recall_probs"]
    priorizar_modelo = params["priorizar_modelo"]

    logger.info(
        f"Empezando la creacion de la base de campaña. Vigencia del archivo: {mes}"
    )
    propension_adq_productos = pd.DataFrame()

    for producto in productos:
        df_temp = pd.DataFrame()
        try:
            # pronosticos de los modelos
            ruta_pronostico = f"{ruta_base}/{producto}/{archivo_nombre}"
            ruta_pronostico = params["base_calificada_path"].replace(
                "{PRODUCT}", producto
            )
            logger.info(f"RUTA BASE CALIFICADA: {ruta_pronostico}")
            # Crear una instancia del sistema de archivos S3
            fs = s3fs.S3FileSystem()
            # ruta en S3 para pickle
            ruta_backtesting = f"{ruta_base}/{producto}/{pickle_backtest_all[producto]}"
            ruta_backtesting = params["pickle_backtest_path"].replace(
                "{PRODUCT}", producto
            )
            logger.info(f"RUTA BACKTESTING: {ruta_backtesting}")
            logger.info("------------")
            logger.info("------------")
            try:
                logger.info("CARGANDO PRONOSTICO...")
                logger.info(ruta_pronostico)
                try:
                    df_temp = backtesting.load_csv(ruta_pronostico)
                except Exception:
                    df_temp = backtesting.load_parquet(ruta_pronostico)
            except Exception:
                logger.info("FALLO. VOLVIENDO A CARGAR PRONOSTICO...")
                logger.info(ruta_pronostico)
                try:
                    df_temp = backtesting.load_csv(ruta_pronostico)
                except Exception:
                    df_temp = backtesting.load_parquet(ruta_pronostico)
            try:
                logger.info("CARGANDO BACKTESTING...")
                logger.info(ruta_backtesting)
                # Leer el archivo directamente desde S3
                with fs.open(ruta_backtesting, "rb") as file:
                    curva_efect = pickle.load(file)
            except Exception as e:
                logger.info(f"FALLO LA LECTURA DEL BACKTESTING: {ruta_backtesting}")
                logger.info(f"ERROR TIPO: {e}")
            logger.info("------------")
            logger.info("------------")
            try:
                logger.info(
                    f"Extrayendo el resultado del ordenamiento de backtesting. Rezago tipo t-{rezagos_predict[producto]}..."
                )
                curva_efect = curva_efect[rezagos_predict[producto]]["Discriminacion"]
            except KeyError as e:  # Capturamos un KeyError si no se encuentra la clave
                logger.error(f"ERROR: Clave no encontrada: {e}")
            except IndexError as e:  # Si ocurre un error de índice, se captura aquí
                logger.error(f"ERROR: Índice fuera de rango: {e}")
            except Exception as e:  # Captura cualquier otro error no anticipado
                logger.error(f"ERROR INESPERADO: {e}")
            # Filter only clients that will open the product
            if metodo_consolidacion is True:
                logger.info("Tomando todas las propensiones en la data")
            else:
                logger.info('Filtrando las propensiones donde "y_pred" es 1')
                df_temp = df_temp[df_temp["y_pred"] == 1]
            # aseguramos ordenamiento de mayor a menor para que mas adelante sea consistente con los deciles
            df_temp = df_temp.sort_values(["y_pred_proba"], ascending=False)
            df_temp.index = range(df_temp.shape[0])
            df_temp.index.name = "Contactabilidad"

            # Manejo de deciles
            # Ordenamiento por Backtesting
            curva_efec = curva_efect.copy()
            curva_efec = curva_efec.sort_values("N", ascending=True)
            curva_efec.index = list(range(curva_efec.shape[0]))

            # parametros de reajuste de backtesting
            value_decil = refactor_backtesting["decil_monto_efect_etiqueta"]
            decil_soporte = refactor_backtesting["alcance_decil_monto_efect"][producto]
            variable_backtesting = refactor_backtesting["tipo_decil_backtesting"]
            variable_ordenamiento_360 = "Ordenamiento Seleccionado"
            escalamiento = curva_efec["xN"].drop_duplicates().iloc[0]

            logger.info(
                f'Metodo de segmentacion del ordenamiento proveniente de "{variable_backtesting}" segun el backtesting.'
            )
            curva_efec[variable_ordenamiento_360] = curva_efec[variable_backtesting]
            # resumen de ordenamientos
            sum_calc_real_final = (
                curva_efec[["N", variable_backtesting]]
                .groupby(variable_backtesting)
                .max()
                .sort_values(by="N", ascending=True)
            )
            sum_calc_real_final = (sum_calc_real_final * escalamiento).rename(
                columns={"N": "Ordenamiento Backtesting/Modelo"}
            )
            if refactor_backtesting["nivel_modelo"] is True:
                pass
            else:
                # desarrollo del modelo
                logger.info(
                    f"Realizando el refactor de categorias para el producto {producto}..."
                )
                # Mensaje de llamado de atencion:
                sum_calc_real = (
                    curva_efec[curva_efec["COLUMN"] == decil_soporte][["N", "xN"]]
                    .iloc[[-1]]
                    .product(axis=1)
                )
                sum_calc_real1 = int(sum_calc_real.iloc[0])
                logger.info(
                    f'La categoria "{homolog_decil_etiquetas[value_decil]}" asociada al decil {value_decil} se generara hasta con {sum_calc_real1} observaciones asociado al {decil_soporte} de la curva del backtesting.'
                )
                sum_calc_real = (
                    curva_efec[curva_efec[variable_backtesting] == value_decil][
                        ["N", "xN"]
                    ]
                    .iloc[[-1]]
                    .product(axis=1)
                )
                sum_calc_real = int(sum_calc_real.iloc[0])
                decil_model = curva_efec[
                    curva_efec[["N", "xN"]].product(axis=1) == sum_calc_real1
                ][variable_backtesting].iloc[-1]
                logger.info(
                    f'Tenga encuenta que a nivel modelo la categoria "{homolog_decil_etiquetas[value_decil]}" sugiere {sum_calc_real} observaciones y {sum_calc_real1} observaciones corresponden al decil {decil_model} o "{homolog_decil_etiquetas[decil_model]}".'
                )
                # rangos de ajuste y desarrollo
                posicion_decil_want = curva_efec[
                    curva_efec["COLUMN"] == decil_soporte
                ].index[0]
                posicion_decil_want1 = curva_efec[
                    curva_efec[variable_backtesting] == value_decil
                ].index[0]
                if (
                    curva_efec.loc[posicion_decil_want, variable_backtesting]
                    < value_decil
                ):
                    curva_efec.loc[
                        posicion_decil_want1:posicion_decil_want,
                        variable_ordenamiento_360,
                    ] = value_decil
                    for t in list(
                        range(posicion_decil_want + 1, curva_efec.shape[0], 1)
                    ):
                        change = (
                            curva_efec.loc[t - 1, variable_ordenamiento_360]
                            - curva_efec.loc[t, variable_ordenamiento_360]
                        )
                        if change > 0:
                            curva_efec.loc[t, variable_ordenamiento_360] = (
                                curva_efec.loc[t - 1, variable_ordenamiento_360] - 1
                            )
                            if curva_efec.loc[t, variable_ordenamiento_360] < 1:
                                curva_efec.loc[t, variable_ordenamiento_360] = 1
            # resumen de ajustes del refactor:
            sum_calc_real_final1 = (
                curva_efec[["N", variable_ordenamiento_360]]
                .groupby(variable_ordenamiento_360)
                .max()
                .sort_values(by="N", ascending=True)
            )
            sum_calc_real_final1 = (sum_calc_real_final1 * escalamiento).rename(
                columns={"N": variable_ordenamiento_360}
            )
            sum_calc_real_final = pd.concat(
                [sum_calc_real_final, sum_calc_real_final1], axis=1
            )
            sum_calc_real_final.index.name = "Decil"
            sum_calc_real_final = sum_calc_real_final.reset_index()
            # etiquetando la categoria al decil
            for pos, col in enumerate(sum_calc_real_final["Decil"]):
                sum_calc_real_final.loc[pos, "segment"] = homolog_decil_etiquetas[col]
            sum_calc_real_final = sum_calc_real_final[
                [
                    "Decil",
                    "segment",
                    "Ordenamiento Backtesting/Modelo",
                    variable_ordenamiento_360,
                ]
            ]
            logger.info("RESULTADO DE ORDENAMIENTO PROVENIENTE DEL BACKTESTING:")
            # se quita el decil visualmente porque este proviende de backtesting y la confianza del modelo individual.
            # mas no de que el decil 10 corresponde al 10% del volumen de datos mas probable
            display(sum_calc_real_final.drop("Decil", axis=1))

            # Incorporando los deciles en los pronosticos

            # datos etiquetativos en cada categoria
            logger.info("Los deciles se realizan de forma equivalente en cada grupo")
            total_clientes = df_temp.shape[0]
            n_deciles = curva_efec[variable_ordenamiento_360].max()
            looop = int(total_clientes / n_deciles)
            last_looop = 0
            for decil, n_max_id in enumerate(
                range(looop, total_clientes + looop + 1, looop)
            ):
                if n_max_id < df_temp.shape[0]:
                    df_temp.loc[df_temp.index[last_looop:n_max_id], "Decil"] = decil
                else:
                    df_temp.loc[df_temp.index[last_looop:], "Decil"] = decil
                    break
                last_looop = n_max_id
            # primer decil o decil 0 es el numero 10 o el mas top
            df_temp["Decil"] = n_deciles - df_temp["Decil"]
            pd.set_option("future.no_silent_downcasting", True)
            df_temp["Decil"] = df_temp["Decil"].replace(0, 1)
            deciles_unique = (
                df_temp["Decil"].value_counts().sort_index().to_frame()
            )  # df_temp['Decil'].unique()
            logger.info("Conteo de Deciles:")
            display(deciles_unique.T)
            # homologando los deciles a etiquetas de decil
            # logger.info(f'Las etiquetas de los deciles se realizan de forma equivalente en cada grupo')
            # n_deciles = df_temp['Decil'].unique().tolist()
            # for decil_value in n_deciles:
            #    df_temp['segment'] = df_temp['Decil'].replace(decil_value,homolog_decil_etiquetas[decil_value]).values
            # las etiquetas de decil se haran segun el ordenamiento del backtesting
            logger.info(
                "Las etiquetas de las propensiones se realizan segun los resultados del backtesting"
            )
            # etiquetando_df_temp = sum_calc_real_final[['Decil','segment',variable_ordenamiento_360]]#.groupby('segment').max()
            # etiquetando_df_temp = etiquetando_df_temp.set_index("Decil")
            etiquetando_df_temp = (
                sum_calc_real_final[["segment", variable_ordenamiento_360]]
                .groupby("segment")
                .max()
            )
            # aseguramos que las propensiones mas altas son las primeras
            etiquetando_df_temp = etiquetando_df_temp.sort_values(
                by=variable_ordenamiento_360, ascending=True
            )
            # extrapolo de modo que la ultima categoria o la peor quede al tamaño de la base de datos
            etiquetando_df_temp.loc[
                etiquetando_df_temp.index[-1], variable_ordenamiento_360
            ] = (df_temp.shape[0] - 1)
            logger.info("ORDENAMIENTO RESPECTIVO: ")
            # logger.info(f'{etiquetando_df_temp}')
            df_temp["segment"] = None
            # df_temp['Decil'] = None
            n_max_id_last = 0
            # for decil_value,n_max_id in zip(etiquetando_df_temp.index,etiquetando_df_temp[variable_ordenamiento_360]):
            # decil_cat = homolog_decil_etiquetas[decil_value]
            for decil_cat, n_max_id in zip(
                etiquetando_df_temp.index,
                etiquetando_df_temp[variable_ordenamiento_360],
            ):
                print(
                    f' La Categoria "{decil_cat}" estara entre la observacion {n_max_id_last} y {n_max_id}.'
                )
                # df_temp.loc[n_max_id_last:n_max_id,'Decil'] = decil_value
                df_temp.loc[n_max_id_last:n_max_id, "segment"] = decil_cat
                n_max_id_last = n_max_id
            decil_cat = df_temp["segment"].unique()
            logger.info(f"Tipo de categorias: {decil_cat}")
            # Z-score
            z_scorer_column = "pred_proba_normalized"  # variable nombre de como se llamaran los z-score de probabilidad
            z_scorer_column_origin = "y_pred_proba"  # variable origen para calcular los z-score de probabilidad
            # Z-score a nivel producto
            # df_temp[z_scorer_column] = stats.zscore(df_temp[z_scorer_column_origin])
            # Z-score a nivel producto y Decil
            # estandarizando por etiqueta (Muy Alto, etc) o por decil.
            logger.info(
                f"Las probabilidades estandarizadas se hacen con el metodo {metodo_estandarizar_probs} y a nivel de propension o segmento!"
            )
            subgrupo_z_scorer = "segment"  # 'Decil', 'segment'
            df_temp[z_scorer_column] = None
            for decil_cat in df_temp[subgrupo_z_scorer].unique():
                if metodo_estandarizar_probs == "Z-Score":
                    z_scorer = stats.zscore(
                        df_temp[df_temp[subgrupo_z_scorer] == decil_cat][
                            z_scorer_column_origin
                        ]
                    )
                if metodo_estandarizar_probs == "Min-Max":
                    z_scorer = df_temp[df_temp[subgrupo_z_scorer] == decil_cat][
                        z_scorer_column_origin
                    ]
                    z_scorer = (z_scorer - z_scorer.min()) / (
                        z_scorer.max() - z_scorer.min()
                    )
                    z_scorer = z_scorer * 100
                rangos_msj = [np.round(z_scorer.min(), 2), np.round(z_scorer.max(), 2)]
                logger.info(
                    f"Segmento de {decil_cat} propension. Rango de Prob Min/Max Estandarizada: {rangos_msj}"
                )
                df_temp.loc[z_scorer.index, z_scorer_column] = z_scorer.values
            if probs_normalizadas is True:
                variable_order = z_scorer_column
            else:
                variable_order = z_scorer_column_origin
            logger.info(
                f"Afectamos el ordenamiento y multiplicamos en {priorizar_modelo[producto]} la variable ``{variable_order}`` para ``{producto}`` "
            )
            df_temp[variable_order] = (
                priorizar_modelo[producto] * df_temp[variable_order]
            )
            logger.info(
                f"{len(df_temp)} clientes predichos para el producto de {producto}"
            )
            if multiplicar_recall_probs is True:
                posibles_analisis_backtesting = [
                    "% Aciertos",
                    "Aciertos",
                    "% Aciertos x Cliente",
                ]
                metrica_view = curva_efec[
                    curva_efec.columns[
                        curva_efec.columns.isin(posibles_analisis_backtesting)
                    ]
                ]
                if metrica_view.shape[1] == 1:
                    if metrica_view.columns[0] == "% Aciertos":
                        col_confianza = "% Aciertos"
                    elif metrica_view.columns[0] == "% Aciertos x Cliente":
                        col_confianza = "% Aciertos x Cliente"
                    else:  # parametrizar los otros metodos
                        col_confianza = None

                    if col_confianza in ["% Aciertos", "% Aciertos x Cliente"]:
                        if (
                            probs_normalizadas is True
                        ):  # escalamiento: z-score* metrica/100 donde metrica>1
                            factor_mult = (
                                1 / 100
                            )  # aplicable a los % Aciertos x Cliente
                        else:  # escalamiento: %probabilidad* metrica donde metrica>1 y 0<probabilidad<1
                            factor_mult = 1

                    logger.info(
                        f"Asociando la variable {variable_order} del modelo 360 con la variable {col_confianza} del backtesting para ajustar el ordenamiento..."
                    )
                    curva_efec[variable_ordenamiento_360] = (
                        curva_efec["N"] * curva_efec["xN"]
                    )
                    metrica_view = curva_efec[
                        [col_confianza, variable_ordenamiento_360]
                    ]
                    resumen_backtest = pd.merge(
                        etiquetando_df_temp.reset_index(),
                        metrica_view,
                        on=variable_ordenamiento_360,
                        how="left",
                    )
                else:
                    logger.info(
                        f"No se encuentra ninguna de las siguientes variables {posibles_analisis_backtesting}"
                    )
                    logger.info(
                        "No se puede incorporar factores como la confianza de cada modelo sobre el ordenamiento para todos los productos"
                    )
                    raise ValueError(
                        f"Debe colocar el parametro``multiplicar_recall_probs`` en False. Ahorita es {multiplicar_recall_probs}"
                    )
                if any(resumen_backtest[col_confianza].isnull()):
                    # si tenemos datos nulos en el merge
                    nulos_adj = resumen_backtest[
                        resumen_backtest[col_confianza].isnull()
                    ]
                    for pos, n_contactable in zip(
                        nulos_adj.index, nulos_adj[variable_ordenamiento_360]
                    ):
                        try:
                            minima_metrica = metrica_view[
                                metrica_view[variable_ordenamiento_360] < n_contactable
                            ][col_confianza].iloc[-1]
                            minimo_n_contactable = metrica_view[
                                metrica_view[variable_ordenamiento_360] < n_contactable
                            ][variable_ordenamiento_360].iloc[-1]
                        except Exception:
                            maxima_metrica = metrica_view[
                                metrica_view[variable_ordenamiento_360] >= n_contactable
                            ][col_confianza].iloc[0]
                            maximo_n_contactable = metrica_view[
                                metrica_view[variable_ordenamiento_360] >= n_contactable
                            ][variable_ordenamiento_360].iloc[0]
                            minima_metrica = maxima_metrica
                            minimo_n_contactable = maximo_n_contactable

                        try:
                            maxima_metrica = metrica_view[
                                metrica_view[variable_ordenamiento_360] >= n_contactable
                            ][col_confianza].iloc[0]
                            maximo_n_contactable = metrica_view[
                                metrica_view[variable_ordenamiento_360] >= n_contactable
                            ][variable_ordenamiento_360].iloc[0]

                        except Exception:
                            maxima_metrica = minima_metrica
                            maximo_n_contactable = minimo_n_contactable
                        # interpolacion lineal suavizada por logaritmos
                        if maximo_n_contactable == minimo_n_contactable:
                            value = minima_metrica
                        else:
                            value = (n_contactable - minimo_n_contactable) / (
                                maximo_n_contactable - minimo_n_contactable
                            )
                            value = np.exp(
                                value * np.log(maxima_metrica)
                                + (1 - value) * np.log(minima_metrica)
                            )
                        # guardando interpolacion
                        resumen_backtest.loc[pos, col_confianza] = value
                if any(resumen_backtest[col_confianza].isnull()):
                    logger.info(
                        "Datos nulos que impiden incorporar la velocidad de convergencia de cada modelo sobre el ordenamiento"
                    )
                    raise ValueError(
                        f"Debe colocar el parametro``multiplicar_recall_probs`` en False. Ahorita es {multiplicar_recall_probs}"
                    )

                for decil_cat, value_adj in zip(
                    resumen_backtest["segment"], resumen_backtest[col_confianza]
                ):
                    logger.info(
                        f"Ajustando el segmento ``{decil_cat}`` en {value_adj*factor_mult} en la variable ``{variable_order}`` que se usara para el ordenamiento"
                    )
                    # clientes que sufren el impacto:
                    # estan en el segmento
                    if col_confianza in ["% Aciertos", "% Aciertos x Cliente"]:
                        # en este caso, se ajusta solo cuando la probabilidad>0
                        # lo anterior es porque si se trabajan con z-score entonces:
                        # si x% > y% , z<0 encontramos que z*x% <z*y% generando un ordenamiento donde el que esta de 1era es el peor producto
                        target = df_temp[
                            (df_temp["segment"] == decil_cat)
                            & (df_temp[variable_order] >= 0)
                        ].index
                    elif col_confianza == "Aciertos":
                        target = df_temp[(df_temp["segment"] == decil_cat)].index
                    df_temp.loc[target, variable_order] = (
                        df_temp.loc[target, variable_order] * value_adj * factor_mult
                    )
                logger.info(
                    "Incorporamos el ordenamiento con el grado de confianza existosamente!"
                )
            else:
                pass
            df_temp = df_temp[
                [ids, z_scorer_column_origin, z_scorer_column, "Decil", "segment"]
            ].copy()
            if producto in homolog.keys():
                logger.info(f"Analizando las particularidades de {producto}")
                df_temp1 = pd.merge(
                    df_temp, clientes_lib[[ids, homolog[producto]]], on=ids, how="left"
                )
                df_temp1[homolog[producto]] = df_temp1[homolog[producto]].fillna(0)
                df_temp1 = df_temp1.drop_duplicates()
                logger.info(f"Segmentando {ids} que no tienen el producto: {producto}")
                # clientes sin el producto vigente
                df_temp2 = df_temp1[df_temp1[homolog[producto]] == 0]
                df_temp2["product"] = producto.upper()
                logger.info(
                    f"Segmentando {ids} que ya tienen el producto(>0): {producto}"
                )
                # clientes que ya cuentan con este producto
                df_temp3 = df_temp1[df_temp1[homolog[producto]] > 0]
                # etiqueca nueva para ofertar diferente a los clientes que tienen mas de 1 producto como este
                new_product = homolog[producto + ">0"].upper()
                logger.info(f"Etiquetando la nueva oferta de {new_product}")
                df_temp3["product"] = new_product
                # unificando los clientes que ya tienen o no el producto
                logger.info(
                    f"Agregando la nueva oferta: {new_product} al producto: {producto}"
                )
                df_temp = pd.concat([df_temp2, df_temp3], axis=0)
                df_temp = df_temp.drop(homolog[producto], axis=1)
                logger.info("Ok producto especifico!")
            else:
                df_temp["product"] = producto.upper()
            df_temp["periodo"] = int(mes)
            logger.info(
                f'Producto: {producto} y etiquetas {df_temp["product"].unique()}'
            )
            # Ordena las columnas
            col_order = [
                "periodo",
                ids,
                "product",
                z_scorer_column_origin,
                z_scorer_column,
                "segment",
                "Decil",
            ]
            df_temp = df_temp[col_order]
            # Concat to base_campanas
            propension_adq_productos = pd.concat(
                [propension_adq_productos, df_temp], ignore_index=True
            )
            logger.info(
                f"Añadiendo {len(df_temp)} pronosticos de {producto} a la base de campañas"
            )
            logger.info(
                "-----------------------------------------------------------------------------------"
            )
        except Exception as e:
            logger.info(e)
            logger.error(f"Error de procesamiento {producto}: {str(e)}")

    # Rename the DataFrame to include the month
    logger.info("Archivo creado completamente")
    logger.info(f"Total observaciones: {len(propension_adq_productos)}")
    logger.info(f"Total productos: {len(propension_adq_productos['product'].unique())}")
    logger.info(f"Tipo de productos: {propension_adq_productos['product'].unique()}")

    for col in [z_scorer_column_origin, z_scorer_column]:
        propension_adq_productos[col] = propension_adq_productos[col].astype(float)
    propension_adq_productos = propension_adq_productos.rename(
        columns={z_scorer_column_origin: "pred_proba"}
    )
    # Control Process:
    logger.info("Ejecutando los controles del modelo...")
    mismo_pronostico = []
    ya_valide = []
    for col_a in propension_adq_productos["product"].unique():
        review = propension_adq_productos[propension_adq_productos["product"] == col_a]
        review = (
            review[[ids, "pred_proba"]]
            .set_index(ids)
            .rename(columns={"pred_proba": col_a})
        )
        for col in propension_adq_productos["product"].unique():
            if (
                (col != col_a)
                & ([col_a, col] not in ya_valide)
                & ([col, col_a] not in ya_valide)
            ):
                df_temp = propension_adq_productos[
                    propension_adq_productos["product"] == col
                ]
                df_temp = (
                    df_temp[[ids, "pred_proba"]]
                    .set_index(ids)
                    .rename(columns={"pred_proba": col})
                )
                review1 = pd.concat([review, df_temp], axis=1)
                if all(review1.diff(axis=1).iloc[:, 1] == 0) is True:
                    mismo_pronostico.append((col_a, col))
                ya_valide.append([col_a, col])
                ya_valide.append([col, col_a])

    if len(mismo_pronostico) > 0:
        logger.info(
            "Revisar que el modelo sea distinto para los siguientes productos descritos:"
        )
        logger.info(mismo_pronostico)
        logger.info("Esto es porque los pronosticos son los mimos!")
        raise
    else:
        logger.info("Control calidad superado")

    logger.info("Primeras pocas filas resultantes de consolidar la informacion: ")
    logger.info("\n" + propension_adq_productos.head().to_string())
    return propension_adq_productos


# Funcion auxiliar al Tercer pipeline
def guardando_ordenamiento_optimo(base_full, base_info):
    """
    Guardaa la informacion de una base pequeña con ciertas columnas e indices en una base gigante que ya contiene dichas columnas e indices.
    Ademas existe una condicion de que la base grande no puede tener informacion en dichas columnas e indices donde se guardara la informacion.

    Parameters
    ----------
    base full:
        DataFrame gigante en donde se guardara informacion

    base_info:
        DataFrame numerico con la informacion que se quiere guardar

    Returns
    -------
    pd.DataFrame gigante con la informacion ya incorporada
    """

    # validacion del proceso de guardado
    if (
        base_full.loc[base_info.index, base_info.columns].dropna(how="all").shape[0]
        == 0
    ):  # = oferta_unica
        # pd.set_option('mode.chained_assignment', None)
        # Ignorar FutureWarning específicos
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*incompatible dtype.*"
        )
        base_full.loc[base_info.index, base_info.columns] = base_info.loc[
            base_info.index, base_info.columns
        ]
    else:
        raise ValueError(
            "No se puede realizar la actualización del ordenamiento porque ya hay datos no nulos en las filas."
        )
    return base_full


def optimizando_propension(comparacion, comparacion_prob, new_columns, total_modelos):
    """
    Identificar el producto con mayor propension entre los productos detectados

    Parameters
    ----------
    comparacion:
        DataFrame numerico asociado al nivel del crecimiento de la efectividad (Muy Alto es 5, Alto es 4, etc...)

    comparacion_prob:
        DataFrame numerico asociado a las probabilidades con las que voy a comparar las probabilidades dentro del mismo nivel de crecimiento o comparacion

    new_columns:
        Lista asociada al nombre de los productos

    total_modelos:
        Numero total de modelos

    Returns
    -------
    pd.DataFrame asociado al producto ordenado de mayor propension a menor propension por id (fila) y columna (ordenamiento)
    """
    # n_maximos = pd.DataFrame()
    comparacion_loop = comparacion.copy()
    # creando el archivo en donde se guardara el ordenamiento perfecto
    ordenamiento_perfecto = pd.DataFrame(index=comparacion.index.tolist())
    for i in range(1, 1 + total_modelos, 1):
        col_name = f"producto_{i}"
        ordenamiento_perfecto[col_name] = np.nan

    for i in range(1, 1 + total_modelos, 1):
        logger.info(f"Identificando la oferta: {i}")
        col_name = f"producto_{i}"
        ids_objetivo = ordenamiento_perfecto[
            ordenamiento_perfecto[col_name].isnull()
        ].index.tolist()
        cal_pct = np.round(100 * len(ids_objetivo) / ordenamiento_perfecto.shape[0], 1)
        logger.info(f"Procesando ordenamiento para el {cal_pct} % de los ids.")
        # seleccionando el producto por mejor etiqueta
        # max_i = comparacion_loop.idxmax(axis=1).to_frame().rename(columns={0:col_name})
        # max_i_val = comparacion_loop.max(axis=1).to_frame().rename(columns={0:col_name})
        max_i = (
            comparacion_loop.loc[ids_objetivo]
            .idxmax(axis=1)
            .to_frame()
            .rename(columns={0: col_name})
        )
        max_i_val = (
            comparacion_loop.loc[ids_objetivo]
            .max(axis=1)
            .to_frame()
            .rename(columns={0: col_name})
        )
        productos_select = list(set(max_i.values.flatten()))
        # revisando el siguiente mejor producto por la siguiente mejor etiqueta
        # comparacion_loop_next = comparacion_loop.copy()
        comparacion_loop_next = comparacion_loop.loc[ids_objetivo].copy()
        for j in range(i + 1, i + 2, 1):
            for prod in productos_select:
                comparacion_loop_next.loc[
                    max_i[max_i.values == prod].index.tolist(), prod
                ] = np.nan
            # max_i_next = (
            #     comparacion_loop_next.idxmax(axis=1)
            #     .to_frame()
            #     .rename(columns={0: f"producto_{j}"})
            # )
            max_i_val_next = (
                comparacion_loop_next.max(axis=1)
                .to_frame()
                .rename(columns={0: f"producto_{j}"})
            )
            # validando que la mejor etiqueta siguiente este en una categoria inferior a la mejor actual:
            validacion = (
                pd.concat([max_i_val, max_i_val_next], axis=1)
                .diff(axis=1)
                .astype(float)
                .iloc[:, 1]
            )
            # filtrando los ids que ya sabemos que estan en la mejor categoria
            best_ids = validacion[validacion != 0].index
            review_ids = validacion[validacion == 0].index
            logger.info(
                f"Del {cal_pct}% de procesamiento, se reprocesa el: {np.round(100*len(review_ids)/max_i.shape[0], 2)}%"
            )
            break
        # sin el loop anterior y ejecutando esta linea sin el loc, el proceso de ejecusion dura el doble
        maximos_multiple = comparacion_loop.loc[review_ids].apply(
            lambda row: row[row == row.max()].index.tolist(), axis=1
        )
        # ya identificados los maximos multiples, actualizamos los productos que tienen la mejor oferta unica
        oferta_unica = max_i.loc[best_ids]
        prod_clean = list(set(oferta_unica[col_name].values))
        for prod in prod_clean:
            # actualizando para eliminar seleccion de este producto en futuras iteraciones
            comparacion_loop.loc[
                oferta_unica[oferta_unica[col_name] == prod].index, prod
            ] = np.nan

        n_comparables = maximos_multiple.apply(len)
        # se optimizara primero los ids que tengan menos productos con el mismo nivel de propension hacia los ids que tengan mas productos con el mismo nivel de propension
        order_opt = n_comparables.unique()
        order_opt = sorted(order_opt)
        if len(order_opt) > 0:
            logger.info(
                f"Existen al menos {j-i+1} productos que tienen el mismo nivel de propension:"
            )
            logger.info(f"{order_opt}")
        for n_compare in order_opt:
            logger.info("------------------")
            # objeto en donde quedara la optimizacion de izquierda a derecha
            logger.info(
                f"Optimizando el ordenamiento segun la probabilidad para los ids que contienen {n_compare} productos con la misma propension mas alta..."
            )
            comparacion_opt_all = pd.DataFrame()
            # los ids que tienen la msima cantidad  de productos en el mismo nivel de propension maximo
            ids_compare = np.array(n_comparables[n_comparables == n_compare].index)
            # los ids y los productos que se deben comparar
            max_multiple_target = maximos_multiple.loc[ids_compare]
            # filtrando o quedandonos con la combinacion de productos unica
            combinaciones_unicas = np.array(
                max_multiple_target.drop_duplicates().values
            )
            # optimizando cada combinacion unica
            for combined in combinaciones_unicas:
                # filtrando los ids con la misma cantidad de productos y la misma combinacion de productos en el nivel mas alto
                max_multiple_target_combined = max_multiple_target[
                    max_multiple_target.isin([combined])
                ]
                # extrayendo sus propensiones similares (id y productos):
                comparacion_temp = comparacion_prob.loc[
                    max_multiple_target_combined.index, combined
                ]
                # optimizando la combinacion unica por probabilidad de izquierda a derecha:
                comparacion_opt = comparacion_temp.apply(
                    lambda row: np.array(row.sort_values(ascending=False).index), axis=1
                )
                comparacion_opt = comparacion_opt.apply(pd.Series)
                comparacion_opt_all = pd.concat(
                    [comparacion_opt_all, comparacion_opt], axis=0
                )
                # validacion
                validacion_mult = comparacion_loop.loc[
                    comparacion_opt.index, combined
                ].std(axis=1)
                if validacion_mult.max() > 0:
                    raise ValueError(
                        "Se estan comparando probabilidades con diferente nivel de propension..."
                    )
                else:
                    # actualizando para eliminar seleccion de estos producto en futuras iteraciones
                    comparacion_loop.loc[comparacion_opt.index, combined] = np.nan
            comparacion_opt_all.columns = [f"producto_{i+j}" for j in range(n_compare)]
            logger.info(
                f"Guardando el ordenamiento para las siguientes ofertas: {comparacion_opt_all.columns.tolist()}"
            )
            ordenamiento_perfecto = guardando_ordenamiento_optimo(
                ordenamiento_perfecto, comparacion_opt_all
            )
            logger.info(
                f"Se actualizo la oferta de multiple propension similar: {i} a {i+n_compare-1}! "
            )
        ordenamiento_perfecto = guardando_ordenamiento_optimo(
            ordenamiento_perfecto, oferta_unica
        )
        logger.info(f"Se actualizo la mejor oferta: {i}! ")
        logger.info("------------------")
        logger.info("------------------")
    if (
        len(
            ordenamiento_perfecto.isnull().sum()[
                ordenamiento_perfecto.isnull().sum() != 0
            ]
        )
        != 0
    ):
        logger.info("Identificamos ordenamientos nulos en:")
        logger.info(
            f"{ordenamiento_perfecto.isnull().sum()[ordenamiento_perfecto.isnull().sum() != 0]}"
        )
        raise ValueError("Revisar la función: ``optimizando_propension``")
    else:
        logger.info("Finalizacion del producto mas propenso. Proceso exitoso")
    # ai el proceso quedo correcto el archivo comparacion_loop debe quedar con solo NAN
    return ordenamiento_perfecto


# Tercer pipeline
def reshape_dataframe(df, params):
    """
    Tomar los pronosticos individuales por modelo, los ordena por id y producto segun la optimizacion del de mayor propension y los entrega en el formato de la linea de negocio.

    Parameters
    ----------
    df:
        DataFrame con los pronosticos de cada producto de manera indifivual concatenado por fila

    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    pd.DataFrame del modelo 360 optimizado
    """
    # Parametros
    probs_normalizadas = params["reprocesar_probs_norm"]
    diccionario_original = params[
        "homolog_decil_etiquetas"
    ]  # Diccionario original de propensiones en valores
    ruta_save = params["ruta_save"]

    ids = params["id"]
    # obtener los productos en el orden del dataframe de entrada
    products = df["product"].unique().tolist()
    # Pivot the dataframe
    logger.info("Generando la tabla dinamica..")
    display(
        df[["pred_proba", "product", "segment"]]
        .groupby(["product", "segment"])
        .describe()["pred_proba"]
        * 100
    )
    df_wide = df.pivot(
        index=["periodo", ids],
        columns="product",
        values=["pred_proba_normalized", "pred_proba", "segment", "Decil"],
    )
    # Renombrando las columnas
    df_wide.columns = [f"{col[1]}_{col[0]}" for col in df_wide.columns]

    # Reiniciando el indice para tener el 'periodo' en columna y el id en el indice
    df_wide = df_wide.reset_index()
    df_wide = df_wide.set_index(ids)

    # base de los pred_proba
    df_wide_probs = extraer_variable_x_producto(df_wide, products, "pred_proba")
    # base de los tickets de los modelos
    df_wide_propen = extraer_variable_x_producto(df_wide, products, "segment")
    # base de las proabilidades estandarizadas
    df_wide_probs_norm = extraer_variable_x_producto(
        df_wide, products, "pred_proba_normalized"
    )
    # base de los tickets de los deciles
    df_wide_decil = extraer_variable_x_producto(df_wide, products, "Decil")
    # logger.info(f"Analizando los {ids} con niveles de propension repetidas en distintos productos...")
    # duplicados_por_fila = df_wide_propen.apply(lambda x: x.value_counts().max() - 1, axis=1)
    logger.info("Creando un valor asociado a cada nivel de propension")
    comparacion = df_wide_propen.fillna("nan")
    # Diccionario nuevo para almacenar la llave más alta por etiqueta
    diccionario_nuevo = {}  # Diccionario de comparacion
    # Recorrer el diccionario original
    for llave, etiqueta in diccionario_original.items():
        # Si la etiqueta aún no está en el diccionario nuevo o la llave es mayor que la actual, la actualizamos
        if etiqueta not in diccionario_nuevo or llave > diccionario_nuevo[etiqueta]:
            diccionario_nuevo[etiqueta] = llave

    # tiqueta asociada al peso mas bajo
    # min_ticket = diccionario_original[np.min(list(diccionario_nuevo.values()))]
    # actualizando los valores asociados a la etiqueta mas baja (Propension mas baja)
    for v in ["nan", "None"]:
        diccionario_nuevo[v] = np.nan
        # si activo estas dos lineas de codigo me va a salir el retanqueo de libranza como 1 producto diferente
        # diccionario_nuevo[min_ticket]-1
        # comparacion = comparacion.fillna(diccionario_nuevo[min_ticket]-1)
    logger.info(
        "Priorizacion el producto segun el nivel de propensión de mas alta categorización :"
    )
    logger.info(f"{diccionario_nuevo}|{np.nan}.")
    for segmento, value in diccionario_nuevo.items():
        pd.set_option("future.no_silent_downcasting", True)
        comparacion = comparacion.replace(segmento, value)
    if probs_normalizadas is True:
        logger.info(
            "Para los productos con el mismo nivel de propension se ordenara segun los proabilidades estandarizadas"
        )
        df_wide_order = df_wide_probs_norm.copy()
    else:
        logger.info(
            "Para los productos con el mismo nivel de propension se ordenara segun los proabilidades directas del modelo"
        )
        df_wide_order = df_wide_probs.copy()
    ordenamiento_perfecto = optimizando_propension(
        comparacion, df_wide_order, products, len(params["productos_carpeta"])
    )
    logger.info(
        "Generando Backup del avance en la ruta destinada para guardar el 360 ..."
    )
    ordenamiento_perfecto.to_parquet(ruta_save)
    # ordenando las probabilidades segun el ordenamiento del producto perfecto
    probs_ord = adjust_format(ordenamiento_perfecto, df_wide_probs, "probabilidad")
    # ordenando el nivel de propension segun el ordenamiento del producto perfecto
    propension_ord = adjust_format(ordenamiento_perfecto, df_wide_propen, "propension")
    # ejecutando el control de propensiones
    propension_ord2 = propension_ord.copy()
    for n, tick in params["homolog_decil_etiquetas"].items():
        propension_ord2 = propension_ord2.replace(tick, n)
    validacion = propension_ord2.fillna(0).diff(axis=1).max(axis=1)
    if validacion.max() == 0:
        logger.info(
            "Validacion aprobada en cuanto al ordenamiento a nivel de propensiones"
        )
    else:
        raise ValueError(
            "El modelo no esta ordenando a nivel de propensiones eficientemente"
        )

    # ordenando el decil segun el ordenamiento del producto perfecto
    deciles_ord = adjust_format(ordenamiento_perfecto, df_wide_decil, "decil")
    # validacion = deciles_ord.fillna(0).diff(axis=1).max(axis=1)
    # if validacion.max() == 0:
    #    logger.info(f'Validacion aprobada en cuanto al ordenamiento a nivel decil')
    logger.info("Consolidando la informacion del modelo 360")
    df_final = pd.concat(
        [probs_ord, ordenamiento_perfecto, propension_ord, deciles_ord], axis=1
    )
    logger.info(
        "Generando Backup del avance en la ruta destinada para guardar el 360 ..."
    )
    df_final.to_parquet(ruta_save)
    df_final = pd.concat(
        [
            df_final,
            comparacion.sum(axis=1).to_frame().rename(columns={0: "propension_360"}),
        ],
        axis=1,
    )
    df_final.to_parquet(ruta_save)
    msj = "las propensiones 360"
    number = 10
    ress = backtesting.deciles_func(
        predicciones_pd=df_final["propension_360"],
        number=number,
        flexibility=True,
        msj=msj,
    )
    if isinstance(ress, pd.Series):
        pass
    elif isinstance(ress, np.ndarray) | isinstance(ress, list):
        ress = pd.Series(ress)
    if ress.drop_duplicates().shape[0] in [number - 1, number, number + 1]:
        pass
    else:
        logger.info('Variable "propension_360"  fuera de rango')
    logger.info("Continuando con el ordenamiento de consumo de clientes ...")
    df_final["propension_360"] = ress
    df_final.sort_values(by="propension_360", ascending=False, inplace=True)
    # # elimina todas las columnas donde todo es nulo
    df_final = df_final.T.dropna(how="all").T
    return df_final


def extraer_variable_x_producto(df_wide, products, variable):
    """
    Funcion que extrae de un dataframe pivoteado una base de datos segun la variable de interes para cada producto
    Parameters
    ----------
    df_wide:
        DataFrame privoteada

    products:
        Lista de productos en el pivoteo.

    variable:
        String asociado a la caracteristica que se quiere extraer del pivoteo.

    Returns
    -------
    pd.DataFrame con todos los productos y con la informacion asociado a la caracterestica input

    """
    column_filt = []
    for product in products:
        column_filt.append(product + "_" + variable)
    # Filtrando la variable del input en cada producto
    df_wide_filt = df_wide[column_filt]
    df_wide_filt.columns = products
    return df_wide_filt


def adjust_format(df_order, df_wide_values, column_name):
    """
    Ordena o formatea el datrame 'df_wide_values' segun el ordenamineto optimo del producto en el dataframe 'df_order' para que tengan el mismo formato.

    Parameters
    ----------
    df_order:
        DataFrame String que contiene en cada columna el producto con mayor propension hacia menor propension por persona o fila

    df_wide_values:
        DataFrame con los datos sin el ordenamiento de 'df_order'
    column_name:
        String asociado al nametag asociado a los valores del dataframe 'df_wide_values'

    Returns
    -------
    pd.DataFrame 'df_wide_values' con la estructura de 'df_order' y la identificacion de columnas seguns 'column_names'
    """
    pd.set_option("future.no_silent_downcasting", True)
    propension_val = pd.DataFrame(
        index=df_wide_values.index, columns=df_order.columns
    ).replace(np.nan, 3)
    n_maximos = df_order.fillna(np.nan)
    logger.info(
        f"Adecuando el formato de ordenamiento de la variable {column_name} segun la priorización optima del producto. "
    )
    for col in n_maximos:
        for labels in n_maximos[col].unique():
            cond = False
            try:
                cond = np.isnan(labels)
                #                print(labels, cond)
                if np.isnan(labels) is True:
                    filt_index = n_maximos.index[n_maximos[col].isnull()].tolist()
                else:
                    filt_index = n_maximos.index[n_maximos[col] == labels].tolist()
            except Exception:
                #                print(labels, type(labels))
                filt_index = n_maximos.index[n_maximos[col] == labels].tolist()
            if len(filt_index) != 0:
                if cond is True:
                    propension_val.loc[filt_index, col] = np.nan
                else:
                    propension_val.loc[filt_index, col] = df_wide_values.loc[
                        n_maximos.index[n_maximos[col] == labels], labels
                    ].values
    propension_val.columns = [
        f"{column_name}_{i}" for i in range(1, n_maximos.shape[1] + 1, 1)
    ]
    return propension_val

    # return df_final


# 4to pipeline
def union_frames(campana, inputs_df, params):
    """
    Tomar los pronosticos del modelo 360, le añade las variables extras y el DEFINIT en el formato deseado por la linea de negocio.
    Posteriormente guarda los resultados

    Parameters
    ----------
    campana:
        DataFrame con los pronosticos optimizados del modelo 360
    inputs_df:
        DataFrame con las variables extras deseadas por la linea de negocio

    params: Dict[str, Any]
        Diccionario de parámetros utilizados en el proceso de preparación de datos.

    Returns
    -------
    pd.DataFrame del modelo 360 optimizado con las variables extras
    """
    ids = params["id"]
    ruta_save = params["ruta_save"]
    logger.info("Unificando Modelo 360 con las variables anexos...")
    campana.index.name = ids
    campana = campana.reset_index()
    campana2 = pd.merge(campana, inputs_df, on=ids, how="left")
    columns = campana2.columns.tolist()
    columns.insert(1, "DEFINIT")
    # ajuste efectivo:
    adj = campana2[ids].str[:2].to_frame().rename(columns={ids: "DEFINIT"})
    adj2 = campana2[ids].str[2:].to_frame().rename(columns={ids: "DEFINIT"})

    adj3 = (
        adj.replace("CC", "C")
        .replace("TI", "T")
        .replace("CE", "E")
        .replace("RC", "R")
        .replace("PS", "P")
        .replace("NI", "L")
    )
    adj3 = adj3 + adj2
    campana2 = pd.concat([adj3, campana2], axis=1)
    campana2 = campana2[columns]
    display(campana2.head())
    logger.info(f"Guardando archivo formato parquet: {ruta_save}")
    for col in [params["id"], "DEFINIT"]:
        logger.info(
            f"{campana2[params['id']].drop_duplicates().shape[0]} observaciones unicas en la variable {col}"
        )
    logger.info(f"{campana2.shape[0]} observaciones totales")
    campana2.to_parquet(ruta_save)
    return campana2
