import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import logging
import gc

# optimizador
from sklearn.calibration import calibration_curve

# metricas
import data_bbog_integration_fabrica_personas.pipelines.backtesting.nodes as backtesting
from sklearn.metrics._classification import (
    _prf_divide,
    _nanaverage,
    multilabel_confusion_matrix,
)
from sklearn.metrics._classification import _check_zero_division, _check_set_wise_labels


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# funcion auxiliar para extraer datos por si se requiere fitear:
# def extract_data(info_save_temp):
#    """
#    Funcion que tiene como objetivo extraer la data
#    balanceada y de entrenamiento del pickle donde se guarda
#   """
#    X_balance = info_save_temp["modelo_produccion"]["X_balance"]
#    y_balance = info_save_temp["modelo_produccion"]["y_balance"]
#    X_i2 = info_save_temp["modelo_produccion"]["X_train"]
#    y_i2 = info_save_temp["modelo_produccion"]["y_train"]

# Convertir X_train y y_train a arrays de NumPy
#    X_train = X_balance.values
#    y_train = y_balance.values.ravel()

# Convertir X_train y y_train sin balancear a arrays de NumPy
#    X_i = X_i2.values
#    y_i = y_i2.values.ravel()
#    return X_train,y_train,X_i,y_i


# funcion auxiliar para eliminar la data del objeto a predecir
def clean_dataset(dicc):
    """
    Funcion que tiene como objetivo eliminar la data balanceada del
    pickle donde se encuentra para que el pickle sea mas liviando
    """
    del dicc["X_balance"]
    del dicc["y_balance"]
    del dicc["X_train"]
    del dicc["y_train"]
    del dicc["X_test"]
    del dicc["y_test"]
    return dicc


def generacion_ks(info_save, info_save2, params):
    """
    El objetivo de esta funcion es implementar la segmentacion
    del algoritmo ks en el modelo de producción

    Parametros
    --------------------------------------
    info_save:
        Pickle asociado al modelo de experimentos
    info_save2:
        Pickle asociado al modelo de produccion sin integrar
    params:
        Parametros de la configuracion del modelo

    Retorno
    --------------------------------------
        Pickle asociado al modelo de produccion integrado

    """
    n_deciles = 95
    top_n = params["n_obs_filter_select"]
    info_save_temp2 = info_save2["modelo_produccion"]
    llave = info_save_temp2["model_name"]

    info_save2["Votaciones"] = {}
    info_save2["Forecast"] = {}
    info_save2["Forecast_probs"] = {}

    if llave == "Ensamble":
        lopps = list(range(0, len(info_save_temp2["nodos_select"]), 1))
        # probs=info_save_loop['best_model'].predict_proba(info_save_loop[dataset_x].values)[:,1]
    else:
        lopps = list(range(0, 1, 1))

    logger.info("Incorporando el KS en el modelo de produccion")
    for dataset_x, dataset_y in [["X_train", "y_train"], ["X_test", "y_test"]]:
        if top_n <= 1:
            if dataset_x in list(info_save_temp2.keys()):
                top_ni = info_save_temp2[dataset_x].shape[0]
            else:
                top_ni = np.unique(
                    [
                        info_save[i][dataset_x].shape[0]
                        for i in list(info_save.keys())
                        if isinstance(i, int)
                    ]
                )
                top_ni = top_ni[0]
                top_ni = int(top_ni * top_n)
        else:
            top_ni = top_n
        logger.info(f"TOP N: {top_ni}.")
        dataset_str = dataset_x[1:]
        for i in lopps:
            if llave == "Ensamble":
                nodo_i = info_save_temp2["nodos_select"][i]
            else:
                nodo_i = info_save_temp2["nodos_select"]
            logger.info(f"Nodo : {i}. Data: {dataset_str[1:]}")
            info_save_loop_loop = info_save[nodo_i].copy()
            probs = info_save_loop_loop["best_model"].predict_proba(
                info_save_loop_loop[dataset_x].values
            )[:, 1]
            y_true = info_save_loop_loop[dataset_y].values
            train_pd = pd.DataFrame(probs, columns=["y_pred_proba"])
            # aggregar los deciles:
            n_deciles_temp = n_deciles
            logramos = True
            while logramos:
                try:
                    # Supongamos que esto lanza una excepción
                    # si n_deciles es menor que 5
                    if n_deciles_temp < 5:
                        raise ValueError("Demasiados pocos deciles")
                    # Aquí es donde harías el cálculo
                    m = (
                        pd.qcut(train_pd["y_pred_proba"], n_deciles_temp, labels=False)
                        + 1
                    )
                    train_pd["decil_apertura"] = m
                    logramos = False  # Si no hay excepción, se sale del bucle
                except Exception:
                    logramos = True  # Se queda en el bucle
                    n_deciles_temp -= 5  # Reduce n_deciles en 5
                    logger.info(f"Try Segment Dist Probs: {n_deciles_temp}")
                    if n_deciles_temp <= 0:
                        m = pd.cut(train_pd["y_pred_proba"], bins=10, labels=False) + 1
                        train_pd["decil_apertura"] = m
                        logramos = False
            train_pd = pd.concat(
                [
                    train_pd,
                    pd.DataFrame(y_true, columns=["y_real"], index=train_pd.index),
                ],
                axis=1,
            )
            # Algoritmo KS
            train_pd["model_name"] = (
                info_save_loop_loop["model_name"]
                + ","
                + info_save_loop_loop["name_model"]
            )
            # Deshabilitar el logger
            # logger.disabled = True
            grouped = backtesting.ks_analysis_pd(train_pd)
            # Habilitar el logger
            # logger.disabled = False

            grouped = grouped.sort_values("tasa_aperturas", ascending=False)
            grouped["cum_total_clientes"] = grouped["total_clientes"].cumsum()
            # grouped_filt = grouped[grouped["cum_total_clientes"] <=top_ni]
            grouped_filt = grouped.copy()
            grouped_filt = grouped_filt[["prob_min", "prob_max", "tasa_aperturas"]]
            logger.info("Iniciando el procesamiento del ks")
            change_size_ks = grouped_filt.shape[0]
            ini_size_ks = grouped_filt.shape[0]
            while change_size_ks != 0:
                grouped_filt = optimizando_dist_prob(grouped_filt.fillna(0))
                change_size_ks = (
                    grouped_filt.shape[0] - ini_size_ks
                )  # nuevo menos al anterior
                ini_size_ks = grouped_filt.shape[0]
            grouped_filt = revisar_probs_interpolada(grouped_filt)
            logger.info("Finalizando el procesamiento del ks")
            if llave != "Ensamble":
                display("ks" + dataset_str)
                display(grouped_filt)
                info_save2["modelo_produccion"]["ks" + dataset_str] = grouped_filt
            else:
                info_save2[nodo_i]["ks" + dataset_str] = grouped_filt
        results, results_probs, results_before = forecast_probs(
            info_save2, lopps, info_save_loop_loop[dataset_x], top_ni, params
        )
        info_save2["Votaciones"][dataset_str[1:]] = pd.concat(
            [results_before, pd.DataFrame(y_true, columns=["y"])], axis=1
        )
        train_pd = pd.concat([results, train_pd], axis=1)
        realizar_predicciones_pd = (
            train_pd.reset_index()
            .drop("index", axis=1)
            .rename(columns={"index": "hashvalue"})
        )
        generate_plots(
            realizar_predicciones_pd, n_deciles_temp, "Data: " + dataset_x[2:]
        )
        info_save2["Forecast"][dataset_str[1:]] = results
        info_save2["Forecast_probs"][dataset_str[1:]] = results_probs
    return info_save2


# Funciones de prediccion
def optimizando_dist_prob(umbral_ks):
    """
    Funcion que tiene como objetivo disminuir el numero de observaciones
    resultantes del algoritmo ks para iterar menos veces en el proceso de
    transformar la distribucion de probabilidad

    params
    ---------------------
        umbral_ks:
            pd.DataFrame con la prob_max, prob_min
            y tasa_aperturas resultante del modelo

    return
    ---------------------
        pd.DataFrame del dataframe de entrada con mejor presentado

    """
    # revisando la tasa de exito replicada
    adj = umbral_ks.groupby("tasa_aperturas").count()
    # filtrando los intervalos de probabilidad donde la tasa de exito es la misma
    adj = (adj > 1).apply(all, axis=1)
    # filtrando los intervalos de probabilidad con tasas de exitos iguales
    aperturas = adj.index[adj is True]
    # filtrando los intervaslos de probabilidad con tasas de exito unicas
    # y guardandolas en opt_ks
    aperturas_unicas = adj.index[adj is not True]
    opt_ks = umbral_ks[umbral_ks["tasa_aperturas"].isin(aperturas_unicas)]
    # iterando los intervalos de probabilidad con tasas de exito iguales
    for t in range(len(aperturas)):
        # tasa de exito duplicada
        tasa = aperturas[t]
        # intervalos de probabilidad duplicados
        sub_group = umbral_ks[umbral_ks["tasa_aperturas"] == tasa]
        # creando lista donde identificare los probabilidades que se puede eliminar
        remove_list = []
        # evaluando en cada intervalo
        for index in sub_group.index:
            # si la probabilidad minima del intervalo esta en
            # l aprobabilidad maxima de otro intervalo
            if any(sub_group["prob_max"] == sub_group.loc[index, "prob_min"]):
                # identifique la posicion del intervalo donde se cumple la regla
                pd.options.mode.chained_assignment = None
                pos_index = sub_group.index[
                    sub_group["prob_max"] == sub_group.loc[index, "prob_min"]
                ].tolist()
                # la probabilidad minima del intervalo que se revisa
                # equivale a la del intervalo asociado
                value_copy = sub_group.loc[pos_index[0], "prob_min"].copy()
                sub_group.loc[index, "prob_min"] = value_copy
                # eliminaremos el intervalo asociado para hacer
                # menos iteraciones en los pronosticos
                remove_list.append(pos_index[0])
        sub_group = sub_group.drop(remove_list, axis=0)
        # guardamos los nuevos intervalos agrupados
        opt_ks = pd.concat([opt_ks, sub_group], axis=0)
    opt_ks = opt_ks.sort_values(by="prob_min", ascending=False)
    opt_ks.index = list(range(opt_ks.shape[0]))
    return opt_ks


# funcion auxiliar de prediccion para evitar 0
def revisar_probs_interpolada(opt_ks):
    """
    Funcion que tiene como objetivo asegurar que las probabilidades minimas
    dentro del rango y maximas dentro del rango sean distintas y
    asegurar que las probabilidades de certeza interpoladas no sean 0

    params
    ---------------------
        umbral_ks:
            pd.DataFrame con la prob_max, prob_min y tasa_aperturas
            resultante del modelo de "optimizando_dist_prob" ya efectuado

    return
    ---------------------
        pd.DataFrame del dataframe de entrada ajustado para predecir
        y realizar predicciones

    """
    # de la data con tasas de apertura distintas y rangos de probabilidad unificados
    # filtramos los casos donde la prob min = prob maximo
    # con tasas de apertura distintas
    adj = opt_ks["prob_min"] == opt_ks["prob_max"]
    rangos_unicos_review = adj.index[adj is True]

    # creando lista donde identificare los rangos de probabilidad a eliminar
    remove_list = []

    # evaluando en cada intervalo con prob min = prob max
    for index in rangos_unicos_review:
        # si el caso a evaluar no esta entre los ya manipulados
        if index not in remove_list:
            # identifique la posicion del ks de prob min tiene la proabilidad maxima
            pos_index = opt_ks.index[
                opt_ks["prob_max"] == opt_ks.loc[index, "prob_min"]
            ].tolist()
            # grupo sobre los cuales se podria unir el intervalo
            # pero cuantan con tasa de ap distintas
            sub_group = opt_ks.loc[pos_index]
            # en el caso de ser una sola fila unica y sin ninguna otra fila de nada
            # entonces unifiquelo con el siguiente nodo
            if (sub_group.shape[0] == 1) & (pos_index[0] + 1 in opt_ks.index.tolist()):
                sub_group = opt_ks.loc[pos_index + [pos_index[0] + 1]]
            # agrupamos la tasa de apertura de la siguiente forma:
            value_copy = np.median(sub_group["tasa_aperturas"])
            # esto permitira que cuando se interpole entonces
            # la prob_min -prob_max !=0 en el pronostico

            # prob min del grupo:
            prob_min = sub_group["prob_min"].min()
            # prob max del grupo:
            prob_max = sub_group["prob_max"].max()
            # eliminados las filas manupuladas
            opt_ks = opt_ks.drop(sub_group.index.tolist(), axis=0)
            # agregando el intervalo revisado
            pd.options.mode.chained_assignment = None
            opt_ks.loc[index, "tasa_aperturas"] = value_copy
            opt_ks.loc[index, "prob_min"] = prob_min
            opt_ks.loc[index, "prob_max"] = prob_max
            # actualziando los casos ya manipulados:
            remove_list = remove_list + sub_group.index.tolist()
    # ordenamos las probabilidades de ordenamiento
    opt_ks = opt_ks.sort_values(by="prob_min", ascending=False)
    # ajustamos los indices (se mantengan ordenados y optimizados
    opt_ks.index = list(range(opt_ks.shape[0]))
    # se ejecuta por ultima vez el optimizador de pronosticos
    opt_ks = optimizando_dist_prob(opt_ks.fillna(0))
    return opt_ks


def select_forecast_ks(umbrales, y_proba, limit_):
    """
    Metodo de ordenamiento optimizado usando los resultados del KS
    -----------------------------------
    Params:
        umbral: Metricas de Ks
        y_proba: Probabilidades obtenidas
        limit_: Numero de personas que se desean para ofertar el producto
    -----------------------------------
    Return:
        np.array que comprende 0 para los casos donde no se pronostica
        una apertura para la observacion y la tasa de acierto asociado
        al pronostico Positivo segun su probabilidad
    """
    probs_rule = np.array([False] * len(y_proba)).astype(float)
    umbral = umbrales.copy()
    umbral = umbral.fillna(0)  # .replace(np.nan,0).replace(None,0)
    # array_vacio = np.empty((len(y_proba), umbral.shape[0]))
    for pos_ks, j in enumerate(umbral.index):
        prob_min = umbral.loc[j, "prob_min"] / 100
        prob_max = umbral.loc[j, "prob_max"] / 100
        if prob_max == umbral["prob_max"].max() / 100:
            prob_max = 1
        if prob_min == umbral["prob_min"].min() / 100:
            prob_min = 0
        # los pronosticos dentro dentro del rango de probabilidades
        probs_rule0 = (y_proba > prob_min) & (prob_max >= y_proba)
        n = probs_rule0.sum()  # cantidad de pronosticos en el rango
        # identificando los pronosticos ya seleccionados previamente al ciclo
        probs_rule_selected = probs_rule > 0  # antes
        diff = probs_rule_selected.astype(int).sum() - limit_
        # los que cumplen la condiccion tendran la tasa de apertura interporlada
        max_value = umbral.loc[j, "tasa_aperturas"]  # prob del modelo mas alta
        try:
            min_value = umbral.loc[
                j + 1, "tasa_aperturas"
            ]  # probabilidad del modelo mas baja
        except Exception:
            min_value = 0  # sino existe otra obs entonces 0
        w_pond = (y_proba - prob_min) / (prob_max - prob_min)  # ponderamos
        probs_rule_temp = (
            max_value - min_value
        ) * w_pond + min_value  # interpolamos todo
        probs_rule0 = probs_rule_temp * probs_rule0.astype(
            float
        )  # los que estan por fuera del rango son 0
        # msg de cuantos llevamos antes del ciclo
        logger.info(
            f"Update probs: Selected: {probs_rule_selected.astype(int).sum()},"
            f" Target Select: {limit_}, Diff %: {diff/limit_}"
        )

        if diff > 0:  # si nos sobran pronosticos
            adj = probs_rule.copy()
            # probs_rule2 = probs_rule.copy()
            indices_altos = np.argsort(adj)[-limit_:]
            probs_rule_temp = np.array([0.0] * len(probs_rule))
            probs_rule_temp[indices_altos] = probs_rule[indices_altos]
            # array_vacio[:,pos_ks] = probs_rule_temp/100
            probs_rule = probs_rule_temp
            probs_rule_selected2 = probs_rule_temp > 0
            diff2 = probs_rule_selected2.astype(int).sum() - limit_
            logger.info(
                f"Finish. Selected: {probs_rule_selected2.astype(int).sum()},"
                f" Target Select: {limit_}, Diff %: {diff2/limit_}"
            )
            break
        elif diff < 0:
            adj = probs_rule0.copy()
            if np.abs(diff) >= n:
                # si faltan pronosticos y ademas los que estan dentro del intervalo
                # superan los que se requieren
                want = -n
            elif np.abs(diff) < n:
                # si faltan pronosticos y ademas con este intervalo
                # nos seguira haciendo faltan
                want = diff
            indices_altos = np.argsort(adj)[want:]
            probs_rule_temp = np.array([0.0] * len(probs_rule0))
            probs_rule_temp[indices_altos] = probs_rule0[indices_altos]
            # array_vacio[:,pos_ks] = probs_rule_temp /100
            probs_rule = probs_rule + probs_rule_temp
        elif diff == 0:
            logger.info("Finish")
            break
        else:
            logger.info("Revisar")
    return probs_rule / 100


# 2. Optimizar predicciones:
def forecast_probs(info_save2, loops, inputs, top_n, params):
    """
    Funcion que tiene como objetivo ajustar los pronosticos segun
    el tipo de modelo calibrado (Ensamble, Modelo Unico) y segun
    el metodo seleccionado de prediccion (Optimizar por Ks, o por umbral).

    params
    ---------------------------------
    info_save2:
        Pickle con el modelo o los modelos respectivos y sus configuraciones realizadas
    loops:
        Lista asociada a la ubicacion o el key asociado al modelo que
        compone parte del resultado del pronostico
    inputs:
        DataFrame procesado y isto para generar las predicciones
    top_n:
        np.int, float, asociado a la cantidad de true que se desean
        en el modelo si se quisiera predecir a partir de la
        optimizaicon ks y se desearan los pronosticos mas probables
    params:
        Dict que contiene el metodo seleccion de prediccion
        (Umbral o Theshold lineal vs Threhold no lineal con el algoritmo ks
    return
    ---------------------------------
    return results:
        prediccion 1 o 0 definido por la configuracion del threshold
    results_probs:
        grado de certeza si la configuracion fue el ks
        o la probabilidad directa del modelo
    results_before_probs:
        Si es un ensamble entregara el  grado de certeza o las
        probabilides de cada modelo individual antes del sistema de votaciones.
        Por el contrario una matriz vacia
    """
    # parametros
    llave = info_save2["modelo_produccion"]["type"]
    # numero de obs a seleccionar
    if top_n <= 1:
        top_ni = int(inputs.shape[0] * top_n)
    else:
        top_ni = top_n
    # sistema de votaciones
    weights = info_save2["modelo_produccion"]["weights"]
    if isinstance(weights, int) | isinstance(weights, float):
        weights = [weights]
    # tipo de seleccion de predicciones
    params_threshold = params[
        "threshold"
    ]  # Cambia este valor al umbral que desees (Params de backtesting)
    if params_threshold["dinamic"] is True:
        logger.info("Tipo de Threshold: KS")
        umbral = params_threshold["class_dinamic"]
    else:
        umbral = params_threshold["umbral"]
        logger.info(f"Tipo de Threshold: {umbral}")

    results = pd.DataFrame()  # prediccion
    results_probs = pd.DataFrame()  # valor asociado a la distribucion de probabilidad
    results_before_probs = pd.DataFrame()  # sistema de votaciones

    for i in loops:
        if llave == "Ensamble":
            nodo_i = info_save2["modelo_produccion"]["nodos_select"][i]
            info_save_loop_loop = info_save2[nodo_i].copy()
            col_name = info_save_loop_loop["name_model"]
        else:
            info_save_loop_loop = info_save2["modelo_produccion"].copy()
            col_name = "y_pred"
        logger.info("------------------------------")
        logger.info(
            f"Modelo: {info_save_loop_loop['model_name']},"
            f"{info_save_loop_loop['name_model']}. Weight: {weights[i]}"
        )
        probs = info_save_loop_loop["best_model"].predict_proba(inputs.values)[:, 1]
        if params_threshold["dinamic"] is False:
            probs_rule = probs.copy()
            y_pred = (probs >= umbral).astype(int)
        else:
            if umbral in info_save_loop_loop:
                probs_rule = select_forecast_ks(
                    info_save_loop_loop[umbral], probs, top_ni
                )
            else:
                probs_rule = select_forecast_ks(
                    info_save_loop_loop["ks_train"], probs, top_ni
                )
            if llave != "Ensamble":
                y_pred = (probs_rule > 0).astype(int)
            else:
                y_pred = probs_rule
        results = pd.concat([results, pd.DataFrame(y_pred, columns=[col_name])], axis=1)
        results_probs = pd.concat(
            [results_probs, pd.DataFrame(probs_rule, columns=["y_p"])], axis=1
        )
    if llave == "Ensamble":
        logger.info("Iniciando sistema de votaciones..")
        tot_sum = np.sum(weights)
        try:
            weights = pd.DataFrame(
                weights, index=results.columns.tolist(), columns=["Weights"]
            )
        except Exception as e:
            logger.info("Codigo va a fallar..")
            logger.info(
                "El numero de algoritmos identificados no coincide"
                " con el numero de pesos definidos..."
            )
            logger.info(
                "Si el tamaño de los vectores definidos en el catalogo coincide..."
                "entonces puede ser que algun nombre de los algoritmos seleccionado"
                "para el ensamblado este mal escrito..."
            )
            raise ValueError("Error:") from e
        # sistema de votaciones ponderado
        validate = results.isnull().sum().replace(0, np.nan).dropna()
        if validate.shape[0] != 0:
            logger.info("Modelos con pronosticos Nulos")
            display(validate)
            display(validate / results.shape[0])
            results = results.fillna(0)
        results_before_probs = results.copy()
        results = (results @ weights).sort_values(
            by="Weights", ascending=False
        ) / tot_sum
        results_probs = results.copy()
        logger.info("Ok Sistema de votaciones")
        results_probs.columns = ["y_p"]
        results_probs = results_probs.sort_index()
        # seleeccionando
        selected = results.iloc[:top_ni].index.tolist()
        # denfiniendo los resultados
        y_hat = np.array([False] * results.shape[0])
        y_hat[selected] = True
        y_hat = y_hat.astype(int)
        results = pd.DataFrame(y_hat, columns=["y_pred"])
    # prediccion (1 o 0, grado de certeza segun ks (usado para predecir),
    # grado de certeza antees del sistema de votaciones
    return results, results_probs, results_before_probs


# Funcion Auxiliar para graficar
def generate_plots(realizar_predicciones_pd, n_deciles, name):
    """
    Funcion auxiliar que genera los graficos para conocer
    el estatus del modelo (Overfitting, Underfitting)

    params:
    -----------------------------------
    realizar_predicciones_pd:
        DataFrame con las probabilidades y los resultados reales
    n_deciles:
        Cantidad de bins usados para discriminar el modelo
    name:
        String asociado al nombre de los graficos
    Return:
    -----------------------------------
    None
        Solo generar los plots del modelo
    """
    # Crear un histograma de la distribución de y_pred_proba
    n_deciles = np.max([n_deciles, 10])
    #
    realizar_predicciones_pd1 = realizar_predicciones_pd[
        ~realizar_predicciones_pd["y_real"].isnull()
    ]
    if realizar_predicciones_pd1["y_pred_proba"].max() > 1:
        realizar_predicciones_pd1["y_pred_proba"] = (
            realizar_predicciones_pd1["y_pred_proba"] / 100
        )

    plt.figure(figsize=(8, 4))
    plt.hist(
        realizar_predicciones_pd1["y_pred_proba"],
        bins=n_deciles,
        edgecolor="k",
        alpha=0.7,
        density=True,
    )
    plt.title("Distribución de y_pred_proba. " + name)
    plt.xlabel("y_pred_proba")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()
    # Crear el histograma
    for value in [1, 0]:
        interes = realizar_predicciones_pd1[
            realizar_predicciones_pd1["y_real"] == value
        ]["y_pred_proba"]
        plt.figure(figsize=(8, 4))
        plt.hist(
            interes,
            bins=n_deciles,
            edgecolor="k",
            alpha=0.7,
            label="y_true = " + str(value),
            density=True,
        )
        plt.title("Distribución de y_pred_proba según y_true. " + name)
        plt.xlabel("y_pred_proba")
        plt.ylabel("Frecuencia")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

    # Graficar la curva de calibración
    # Calcular la curva de calibración
    prob_true, prob_pred = calibration_curve(
        realizar_predicciones_pd1["y_real"],
        realizar_predicciones_pd1["y_pred_proba"],
        n_bins=10,
    )

    # curva > 45 grados = subestimar
    # curva < 45 grados = sobrestimar
    plt.figure(figsize=(8, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Curva de Calibración")
    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectamente calibrado"
    )
    plt.title("Curva de Calibración. " + name)
    plt.xlabel("Probabilidad Predicha")
    plt.ylabel("Frecuencia Real")
    plt.legend()
    plt.show()


# 0.1 Funcion auxiliar para generar el pickle ensamble
def procesamiento_data_ensamble_for_train(info_save, params):
    """
        Funcion auxiliar para crear el espacio en donde
        quedara el modelo de produccion del ensamble
    ------------------------------------------------
    params:
        info_save:
            Pickle en donde se encuentran los modelos usados en la experimentacion
        params:
            Dict con la configuracion del codigo
    ------------------------------------------------
    return:
        Pickle actualizado y configurado al entorno de produccion.
            Adicionalmente contiene unos dataframe con el nombre de 'X_train',
            'X_test' por si se quisiera entrenar un modlo sobre los resultados
            de cada pronostico de los experimentos.
    """
    # parametros
    combine_models_select = [i.upper() for i in params["Ensamble"]["Algoritmos"]]
    weights = params["Ensamble"]["weights"]

    # revisando la calidad de los inputs
    if len(combine_models_select) != len(weights):
        raise ValueError(
            "En el ensamblado, el numero de algoritmos"
            "seleccionados no encada con el peso"
        )

    # filtrando los modelos puestos que no tienen ninun peso
    combine_models_select_filt = []
    weights_filt = []
    for t in range(len(weights)):
        if weights[t] > 0:
            combine_models_select_filt.append(combine_models_select[t])
            weights_filt.append(weights[t])
    # ordenando la lista de los modelos y los pesos
    combine_models_select__ord = []
    weights_ord = []
    nodos_select = []
    for llave in info_save.keys():
        if isinstance(info_save[llave], dict):
            if "name_model" in list(info_save[llave].keys()):
                name = info_save[llave]["name_model"].upper()
                for name_comp, w in zip(combine_models_select_filt, weights_filt):
                    if name == name_comp:
                        combine_models_select__ord.append(name_comp)
                        weights_ord.append(w)
                        nodos_select.append(llave)
                        break

    # creadno las variables entrada y salida
    info_save2 = {"modelo_produccion": None}
    info_save2["modelo_produccion"] = {}
    info_save2["modelo_produccion"]["type"] = "Ensamble"
    # entrenando el modelo
    info_save2["modelo_produccion"]["best_model"] = "Ensamble"
    info_save2["modelo_produccion"]["model_name"] = "Ensamble"
    info_save2["modelo_produccion"]["name_model"] = ""
    info_save2["modelo_produccion"]["name_model_combined"] = combine_models_select__ord
    info_save2["modelo_produccion"]["nodos_select"] = nodos_select
    info_save2["modelo_produccion"]["weights"] = weights_ord
    for nodo in info_save2["modelo_produccion"]["nodos_select"]:
        dicc = info_save[nodo].copy()
        scaler = dicc["Scaler"]
        del dicc["Scaler"]
        if "Scaler" not in list(info_save2["modelo_produccion"].keys()):
            logger.info("Preprocesamiento de datos equivalente en todos los modelos")
            info_save2["modelo_produccion"]["Scaler"] = scaler
        info_save2[nodo] = clean_dataset(dicc)
    logger.info("Finalizacion de estructura del ensamblado ")
    return info_save2


# 1. Nodo inicial ejecucion del modelo seleccionado o superoptimizacion
def generate_modelo_produccion(info_save, params):
    """
    Se construye el Pickle en donde quedara el modelo o los
    modelos de procesamiento y prediccion para un entorno de pronosticos.

    parameters
    ---------------------------------------
    info_save:
        Corresponde al Pickle donde se encuentran los experimentos

    params:
        Dict donde estan los parametros del modelo que quiere ejecutar

    return
    ---------------------------------------

    Pickle mas liviano con el (los) modelo(s) de interes en producción
            El codigo guarda:
                Algoritmo KS en la data de entrenamiento y testeo.
                Genera y guarda las predicciones junto con el sistema de predicciones

    """
    combine_models = params["Ensamble"]["want"]
    info_save2 = {}
    if combine_models is True:
        logger.info("Iniciando el Ensamblado de modelos...")
        # creando el pickle desde 0 y quedandonos con lo relevante
        info_save2 = procesamiento_data_ensamble_for_train(info_save, params)
    else:
        if params["use_key"]["want"] is False:
            key_model = params["use_key"]["estrategia"]
            logger.info(f"Modelo optimo segun la estrategia: {key_model}")
            # identificar el modelo que se selecciono
            llave = info_save["select_model"].loc["model_name", key_model].iloc[0]
            nodo_best = int(llave.split(",")[1].split("_")[-1])
        else:
            llave = params["use_key"]["llave"][0]
            logger.info("Modelo optimo sin estrategia.")
            nodo_best = int(llave.split(",")[1].split("_")[-1])

        logger.info(f"Llave del modelo: {llave}")
        if (
            llave
            == info_save[nodo_best]["name_model"]
            + ","
            + info_save[nodo_best]["model_name"]
        ):
            logger.info("Modelo Identificado....")
            logger.info(f"Nodo: {nodo_best}. Llave {llave}")
            logger.info("Seleccionamos el modelo proveniente de models...")
            # quedarse con el modelo correcto y parametrizado en el otro artefacto nuevo
            dicc = info_save[nodo_best].copy()
            info_save2["modelo_produccion"] = clean_dataset(dicc)
            info_save2["modelo_produccion"]["type"] = "Models"
            info_save2["modelo_produccion"]["nodos_select"] = nodo_best  # No es lista
            info_save2["modelo_produccion"]["weights"] = 1  # No es lista
    info_save2 = generacion_ks(info_save, info_save2, params)
    gc.collect()
    return info_save2


# Nodo 2: Generar metricas de desempeño de datos
# similares a los vistos en backtesting

# 2.1 funciones auxiliares para generar dichas metricas


# funcion auxiliar para filtrar los mas probables
def want_res(y_probs, n_top):
    """
        Funcion auxiliar que retorna las posiciones
        asociados a los valores mas altos de un array

    parameters
    ---------------------------------------
    y_probs:
        Array con los valores sobre los cuales quiero su posicion mas alta

    n_top:
        np.float asociado al numero de valores mas altos que deseo identificar

    return
    ---------------------------------------
    Array con las posiciones de los valores mas altos

    """
    # Obtener las observaciones con las mayores probabilidades combinadas
    # ordenar de menor a mayor
    top_indices = np.argsort(y_probs)
    # quedarme con los de maxima probabilidad
    top_indices = top_indices[-n_top:]
    return top_indices


# funcion auxiliar para calcular las metricas de sklear
# ajustadas por las mas probables y optimizar en el ensamble
def precision_recall_fscore_support_fabrica(
    y_true,
    y_probs,
    n_top,
    metric_calcs,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("f-score"),  # ("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
):
    """
        Funcion para calcular las metricas de sklear
        ajustadas por las clasificaciones mas probables

    parameters
    ---------------------------------------
    y_true:
        Array con los resultados reales de toda la data

    y_probs:
        Array con los valores (propensiones o proabbilidades) de toda la data

    n_top:
        np.float asociado al numero de valores mas altos que deseo identificar

    metric_calcs:
        Pueede ser: 'precision', 'recall', 'f_score', 'true_sum'.
        Si se recibe un string diferente entonces entrega las 4 metricas

    return
    ---------------------------------------
    Array o tupla de arrays segun como se reciba el input metric_calcs
    """
    # Obtener las observaciones con las mayores probabilidades combinadas
    top_indices = want_res(y_probs, n_top)
    # Filtrar las etiquetas reales
    y_top = y_true[top_indices]

    # Filtrar las predicicones seleccionadas
    y_call = np.array([1.0] * len(top_indices))
    # y_call = y_preds[top_indices]
    _check_zero_division(zero_division)
    labels = _check_set_wise_labels(y_top, y_call, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_top,
        y_call,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    # MCM[0,:,:] # MATRIZ DE CONFUSION PARA LABELS[0] --> RESULTADO [(TN,FP) , (FN, TP)]
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta**2
    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    if np.isposinf(beta):
        f_score = recall
    elif beta == 0:
        f_score = precision
    else:
        # The score is defined as:
        # score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        # Therefore, we can express the score in terms of confusion matrix entries as:
        # score = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)
        denom = beta2 * true_sum + pred_sum
        f_score = _prf_divide(
            (1 + beta2) * tp_sum,
            denom,
            "f-score",
            "true nor predicted",
            average,
            warn_for,
            zero_division,
        )

    # Average the results
    if average == "weighted":
        weights = true_sum
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = _nanaverage(precision, weights=weights)
        recall = _nanaverage(recall, weights=weights)
        f_score = _nanaverage(f_score, weights=weights)
        true_sum = None  # return no support
    if metric_calcs == "precision":
        return precision
    elif metric_calcs == "recall":
        return recall
    elif metric_calcs == "f_score":
        return f_score
    elif metric_calcs == "true_sum":
        return true_sum
    else:
        return [precision, recall, f_score, true_sum]


# funciones auxiliar del ensamble para optimizar/validar metricas:
def custom_f1_score(y_true, y_probs, top_n=0.15, labels=[1], metric_calcs="f_score"):
    """
    Funcion que me retorna el F1 score por default de los pronosticos
    donde el modelo sugiere que son los mas probables.

    parameters
    ---------------------------------------
    y_true:
        Array con los resultados reales de toda la data

    y_probs:
        Array con los valores (propensiones o proabbilidades)
        de toda la data

    n_top:
        np.float asociado al numero o % de de datos valores mas
        altos que deseo identificar. DEFAULT 15% de la data

    metric_calcs:
        Puede ser: 'precision', 'recall', 'f_score', 'true_sum'.
        Si se recibe un string diferente entonces la
        funcion retorna la precision. Default 'f_score'

    return
    ---------------------------------------
    Unica metrica deseada
    """
    if top_n <= 1:
        top_n = len(y_probs) * top_n

    array_value = precision_recall_fscore_support_fabrica(
        y_true,
        y_probs,
        n_top=top_n,
        labels=labels,  # [0,1]
        metric_calcs=metric_calcs,  # f_score,recall,precision
    )
    return array_value[0]  # metrica unia deseada


def backtesting_top_probabilities(y_true, y_probs, top_n=0.15):
    """
    Funcion que me retorna el F1 score por default de los
    pronosticos donde el modelo sugiere que son los mas probables.

    parameters
    ---------------------------------------
    y_true:
        Array con los resultados reales de toda la data

    y_probs:
        Array con los valores (propensiones o proabbilidades)
        de toda la data

    n_top:
        np.float asociado al numero o % de datos en donde quiero evaluar
        sus valores mas altos. Default: 15% de la data

    return
    ---------------------------------------
    Metrica de linea de negocio definida como:
        return TP/TOTAL
    TP:       True Positive o Aciertos
    TOTAL:    Numero Total de pronosticos en la evaluacion
    """
    # Step 1: Sort df by 'pred_proba' from highest to lowest
    # Obtener las observaciones con las mayores probabilidades combinadas
    if top_n <= 1:
        top_n = len(y_probs) * top_n
    top_indices = want_res(y_probs, top_n)
    # top_indices_str = [str(int(index)) for index in top_indices]

    # Filtrar las etiquetas reales
    y_top = y_true[
        top_indices
    ]  # tomo las etiquetas reales que seleccione de las predicciones
    # y_top = (y_top == 1).flatten() # dejo en True las TruePositive
    y_top = np.array(list((y_top == 1).flatten()))
    true_positive_indices = np.where(y_top)[
        0
    ]  # extraigo la posicion de los TruePositive

    # Step 2: Get the unique hashvalues from df_test
    test_hashvalues = set(true_positive_indices)
    # Step 3: Get the hashvalues in the top_n sorted df
    # top_hashvalues = set(top_indices)

    # Step 4: Calculate the percentage of test_hashvalues present in top_hashvalues
    # common_hashvalues = test_hashvalues.intersection(top_hashvalues)
    # percentage_in_top = (len(common_hashvalues) / len(test_hashvalues))
    percentage_in_top = len(test_hashvalues) / np.min([top_n, y_top.shape[0]])
    return percentage_in_top


# Nodo 2: Calculo de las metricas de desempeño comparables al backtesting:
def calc_metrics_before_backtesting(info_save, info_save2, params):
    """
    Funcion que me genera las metricas del modelo segun los mas
    probables del modelo de produccion y los 3 modelos de referencia
    propuestos por default en la capa models
    {Best Model Name, Best fitting Model, Best Bies Model}

    parameters
    ---------------------------------------
    info_save:
        Pickle con la experimentacion de los modelos

    info_save2:
        Pickle del modelo o modelos que se cargaran en produccion

    params:
        Dict de parametros

    return
    ---------------------------------------
    Pickle del modelo de produccion con las metricas guardadas internamente
    segun los mas probables o en donde 'y_pred' sea 1
        Precision
        Recall
        F1-score
        True positive
        Backtesting de la linea de negocio
        Pickle liviano sin la data, ni predicciones de entrenamiento ni testeo.
    """
    logger.info("Iniciando el calculo de las metricas...")
    top_n = params["n_obs_filter_select"]
    # numero de deciles
    # n_deciles = 95
    params_threshold = params[
        "threshold"
    ]  # Cambia este valor al umbral que desees (Params de backtesting)
    if params_threshold["dinamic"] is True:
        logger.info("Tipo de Threshold: KS")
    else:
        logger.info(f"Tipo de Threshold: {params_threshold['umbral']}")

    metrics_backtest = pd.DataFrame()
    info_save2["Votaciones"] = {}
    info_save2["modelo_produccion"]["weights"] = params["Ensamble"]["weights"]

    for col in info_save["select_model"].columns.tolist() + ["modelo_produccion"]:
        if col == "modelo_produccion":
            # debe buscar en el proyecto generado de select_model
            nodo_col = info_save2[col]["nodos_select"]
            if (
                isinstance(nodo_col, (int, float))
                and info_save2[col]["type"] == "Models"
            ):
                # es un modelo preseleccionado en modelos
                llave = (
                    info_save2[col]["name_model"] + "," + info_save2[col]["model_name"]
                )
                info_save2["modelo_produccion"]["weights"] = [None]
            elif isinstance(nodo_col, list) and info_save2[col]["type"] == "Ensamble":
                llave = "Ensamble"
            else:
                raise ValueError(
                    "¿Que esta pasando? Revisar cuando se "
                    "parametrizo ´['modelo_produccion']['type']´´"
                )
            info_save_loop = info_save2["modelo_produccion"].copy()
        else:
            # debe buscar en el proyecto generado de models
            logger.info(f"Models: {col}")
            llave = info_save["select_model"].loc["model_name", col]
            nodo_col = int(llave.split(",")[1].split("_")[-1])
            info_save_loop = info_save[nodo_col].copy()
        if (
            llave == info_save_loop["name_model"] + "," + info_save_loop["model_name"]
        ) | (llave == "Ensamble"):
            logger.info(f"Estrategy: {col}")
            for dataset_x, dataset_y in [["X_train", "y_train"], ["X_test", "y_test"]]:
                if top_n <= 1:
                    if dataset_x in list(info_save_loop.keys()):
                        top_ni = info_save_loop[dataset_x].shape[0]
                    else:
                        top_ni = np.unique(
                            [
                                info_save[i][dataset_x].shape[0]
                                for i in list(info_save.keys())
                                if isinstance(i, int)
                            ]
                        )
                        top_ni = top_ni[0]
                    top_ni = int(top_ni * top_n)
                else:
                    top_ni = top_n

                logger.info(f"TOP N: {top_ni}. DATASET: {dataset_x[2:]}")
                if col == "modelo_produccion":
                    if llave == "Ensamble":
                        nodo_i = info_save2[col]["nodos_select"][0]
                    else:
                        nodo_i = info_save2[col]["nodos_select"]
                    y_true = info_save[nodo_i][dataset_y].values
                    # results= info_save2['Forecast'][dataset_x[2:]]
                    # probs = np.array(results['y_pred'])
                    results_probs = info_save2["Forecast_probs"][dataset_x[2:]]
                    probs = np.array(results_probs["y_p"])
                else:
                    y_true = info_save_loop[dataset_y].values
                    probs = info_save_loop["best_model"].predict_proba(
                        info_save_loop[dataset_x].values
                    )[:, 1]
                # calculo para todos
                labels = [1, 0]
                try:
                    backtest_metric = np.round(
                        backtesting_top_probabilities(y_true, probs, top_ni) * 100, 4
                    )
                except Exception:
                    backtest_metric = None
                (
                    precision,
                    recall,
                    f_score,
                    true_sum,
                ) = precision_recall_fscore_support_fabrica(
                    y_true,
                    probs,
                    n_top=top_ni,
                    labels=labels,  # [0,1]
                    metric_calcs="all",  # f_score,recall,precision
                )
                index_name = [
                    "Precision n_top",
                    "recall n_top",
                    "f1-score n_top",
                    "true_sum n_top",
                ]
                temp2 = pd.DataFrame(["backtest", backtest_metric, "1", dataset_y[2:]])
                temp2.index = ["metric_name", "value", "class_name", "dataset_name"]
                temp2 = temp2.T
                for t, index in enumerate([precision, recall, f_score, true_sum]):
                    if t == 3:
                        temp = pd.DataFrame(
                            index,
                            index=[index_name[t]] * len(labels),
                            columns=["value"],
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
                    temp["dataset_name"] = dataset_y[2:]
                    temp2 = pd.concat([temp2, temp], axis=0)
                if llave == "Ensamble":
                    temp2["name_model"] = str(nodo_col)
                    temp2["model_name"] = info_save2["modelo_produccion"]["model_name"]
                else:
                    temp2["name_model"] = info_save[nodo_col]["name_model"]
                    temp2["model_name"] = info_save[nodo_col]["model_name"]
                temp2["select_model"] = col
                metrics_backtest = pd.concat([metrics_backtest, temp2], axis=0)

            metrics_backtest.sort_index(ascending=False, inplace=True)
            check_df = temp2.set_index(["class_name", "metric_name", "dataset_name"])
            printed = check_df[np.in1d(check_df.index.get_level_values(0), [1])]
            printed2 = printed[
                np.in1d(
                    printed.index.get_level_values(1),
                    ["f1-score n_top", "true_sum n_top"],
                )
            ]

            printed1 = check_df[
                np.in1d(check_df.index.get_level_values(1), ["backtest"])
            ]
            printed = pd.concat([printed2, printed1], axis=0)
            display(printed)
        else:
            raise ValueError(
                "¿Que esta pasando? Revisar cuando se parametrizo"
                " ´['modelo_produccion']['type']´"
            )

    metrics_backtest.set_index(
        ["class_name", "metric_name", "dataset_name"], inplace=True
    )
    info_save2["select_model2"] = metrics_backtest

    # otras metricas:
    if info_save2[col]["type"] == "Models":
        llave = (
            info_save2["modelo_produccion"]["name_model"]
            + ","
            + info_save2["modelo_produccion"]["model_name"]
        )
        save_df = info_save["all_results"][
            info_save["all_results"]["model_name"] == llave
        ]
    else:
        save_df = info_save["all_results"]
    info_save2["all_results_select_model"] = save_df
    info_save3 = info_save2.copy()
    del info_save3["Forecast_probs"]
    del info_save3["Forecast"]
    return info_save3
