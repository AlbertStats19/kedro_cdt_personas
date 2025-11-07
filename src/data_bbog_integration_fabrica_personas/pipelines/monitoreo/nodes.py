"""
Nodos de la capa monitoreo
"""
import pandas as pd
import numpy as np
import logging
from io import StringIO
import logging

import os
import boto3

from IPython.display import display
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.platypus import TableStyle

import data_bbog_integration_fabrica_personas.pipelines.backtesting.nodes as backtesting

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Funcion auxiliar para cargar los monitoros del modelo
def load_pickle(path):
    return pd.read_pickle(path)

# Funcion auxiliar para manipular las metricas del modelo
def manipulando_metricas(insumos_ordenados,cortes_totales, params):
    """
    Toma la base de datos de los pronosticos y modela la contactabilidad bajo cierto
    niveles de contactabilidad

    Parameters
    ----------
    insumos_ordenados: Dict
        Diccionario que contiene las metricas de ordenamiento en diferentes cosechas para un corte de 
        rezago predictivo especifico
    cortes_totales: list 
        lista de cortes de tiempo que se cargaron para generar el monitoreo del modelo
    params:
        Dict de parametros

    Returns
    -------
        Diccionario de etiquetas de cortes de tiempo dell modelo y diccionario de bases de datos con las 
        metricas sobre las cuales se monitoreara el modelo
    """
    x_name = params['x_name']
    y_name = params['y_name']
    z_name = params['z_name']
    rezago = params['rezago']    
    nivel_aceptabilidad = params['sd_aceptabilidad']
    nodos = params['nodos_curva']
    periodos_ventana_corta = params['n_cortes_ventana_cp']
    periodos_ventana_larga = params['n_cortes_ventana_lp']
    corte_actual = cortes_totales[-1]
    corte_ventana_movil =  cortes_totales[-2]
    corte_ventana_movil_ini =  cortes_totales[1]
    corte_produccion = cortes_totales[0]
    logger.info(f'Se evaluara el deterorio del modelo sobre el corte actual o el desempeño en {corte_actual}.')
    logger.info(f'Consideraremos, el corte de produccion {corte_produccion}')
    logger.info(f'Y las ventana movil de los ultimos meses {corte_ventana_movil_ini} : {corte_ventana_movil}')
    cortes_totales2 = cortes_totales.copy()
    if corte_produccion in cortes_totales2:
        cortes_totales2.remove(corte_produccion)
    # guardando los cortes que estoy procesando
    etiquetas= {'cortes_totales':cortes_totales}
    etiquetas['corte_actual'] = corte_actual
    etiquetas['corte_ventana_movil_final'] = corte_ventana_movil
    etiquetas['corte_ventana_movil_ini'] = corte_ventana_movil_ini
    etiquetas['corte_produccion'] = corte_produccion
    etiquetas['cortes_totales2'] = cortes_totales2

    logger.info(f'Manipulando la información en el rezago: {rezago}')
    sub_bd = insumos_ordenados[rezago]
    if (x_name == 'N') | (y_name == 'N'):
        adj_value = sub_bd['xN'].iloc[0]
    else:
        adj_value = 1
    # efectuando el ordenamineto por contactabilidad
    sub_bd = sub_bd.sort_values(by='N')
    sub_bd.index = range(sub_bd.shape[0])
    x = sub_bd[x_name].drop_duplicates()
    y = sub_bd[y_name].drop_duplicates()
    if x_name == 'Corte':
        ciclo1 = x.copy()
        ciclo2 = y.copy()
    elif y_name == 'Corte':
        ciclo1 = y.copy()
        ciclo2 = x.copy()
    # matriz vacia donde quedaran los cortes de tiempo y el ordenamiento
    z = pd.DataFrame(index = x, columns = y)
    if x_name in ['COLUMN','N']:
        for tt in y:
            z[f'{tt}_N'] = None
    else:
        for t in x:
            z.loc[f'{t}_N'] = None
    
    # rellenando la matriz de cortes de tiempo y ordenamienoto
    errores = []
    for t in ciclo1:
        sub_bd_filt = sub_bd[sub_bd[x_name] == t]
        for tt in ciclo2:
            sub_bd_filt3 = sub_bd_filt[sub_bd_filt[y_name] == tt]
            sub_bd_filt2 = sub_bd_filt3[z_name]
            if sub_bd_filt2.shape[0] == 1:
                z.loc[t,tt] = sub_bd_filt2.iloc[0]
                if x_name in ['COLUMN','N']:
                    z.loc[t,f'{tt}_N'] = sub_bd_filt3[['N','xN']].prod(axis  =1).iloc[0]
                else:
                    z.loc[f'{t}_N',tt] = sub_bd_filt3[['N','xN']].prod(axis  =1).iloc[0]
            else:
                if x_name in ['COLUMN','N']:
                    z.loc[t,f'{tt}_N'] = t*adj_value
                    if tt not in errores:
                        errores.append(tt)
                else:
                    z.loc[f'{t}_N',tt] = tt*adj_value
                    if t not in errores:
                        errores.append(t)
    # validando los cortes donde se identifica nulidad para interpolar
    errores = sorted(errores)
    for adj in errores:
        logger.info(f'Interpolando nodos de la curva nulos para el corte {adj}')
        if x_name in ['COLUMN','N']:
            metrica_nulos = z[adj].astype(float)
            n_obs = z[f'{adj}_N'].astype(float)
        else:
            metrica_nulos = z.loc[adj].astype(float)
            n_obs = z.loc[f'{adj}_N'].astype(float)
        metrica_nulos = np.log(metrica_nulos)
        metrica_nulos = metrica_nulos.replace(np.inf,0)
        metrica_nulos = metrica_nulos.replace(-np.inf,0)
        # Identificar los índices que no son NaN
        valid_indices = metrica_nulos.dropna().index
        
        # Interpolar usando el método de interpolación lineal
        f_interp = spi.interp1d(n_obs.loc[valid_indices],metrica_nulos.loc[valid_indices],
                                kind='quadratic', fill_value="extrapolate")
        
        # Crear nuevos valores interpolados
        y_interp = f_interp(n_obs)
        for t,pos in enumerate(metrica_nulos.index):
            if metrica_nulos.isnull().loc[pos] == True:
                metrica_nulos.loc[pos] = y_interp[t]
        metrica_nulos = np.exp(metrica_nulos)
        if x_name in ['COLUMN','N']:
            z[adj] = metrica_nulos
        else:
            z.loc[adj] = metrica_nulos
    
    # limpiando la base de datos para tener el analisis grafico
    for adj in cortes_totales:
        if x_name in ['COLUMN','N']:
            z = z.drop([f'{adj}_N'], axis = 1)
        else:
            z = z.drop([f'{adj}_N'], axis = 0)
    
    # asegurando que las cifras sean numericas
    z = z.astype(float)
    z1 = z.T
    if x_name in ['COLUMN','N']:
        x = x.astype(float)
        y = y.astype(int)
    else:
        x = x.astype(int)
        y = y.astype(float)
    x = x.sort_values()
    # filtrando los escenarios de interes
    try:
        z1 = z1[x.values]
    except:
        z1 = z1[cortes_totales]
    # construyendo la malla para graficar la superficie del modelo
    x = x[~x.isin([corte_produccion, int(corte_produccion)])]
    x1, y1 = np.meshgrid(x, y)
    # parametros para generar la alerta
    try:
        z_actual = z1[corte_actual]
    except:
        z_actual = z1[int(corte_actual)]
    
    try:
        z_inicial = z1[corte_produccion]
    except:
        z_inicial = z1[int(corte_produccion)]    
    metrics = {
        'z_inicial':z_inicial,
        'z_actual':z_actual,
        'x1':x1,
        'y1':y1,
        'z1':z1
    }
    return etiquetas,metrics

# 1er pipeline del modelo
def extraer_metricas(params):
    """
    Segun los parametros definidos, el codigo carga, reprocesa y manipula las metricas de monitoreo 
    para tener una feature dentro de diccionarios que se altamente manejable a la generacion de alertas

    Parameters
    ----------
    params:
        Dict de parametros

    Returns
    -------
        Diccionario de etiquetas de cortes de tiempo dell modelo y diccionario de bases de datos con las 
        metricas sobre las cuales se monitoreara el modelo
    """
    ruta = params['ruta_insumo_monitoreo']
    archivos_360 = params['archivo_360'] 
    archivos_backtesting = params['archivo_backtesting'] 
        
    cortes_totales = list(archivos_backtesting.keys())+list(archivos_360.keys())
    cortes_totales = sorted(list(set(cortes_totales)))
    insumos_ordenados = {}
    for corte in cortes_totales:
        try:
            archivo = archivos_360[corte]
            ruta_corte = f'{ruta}{archivo}'
            insumos_360 = load_pickle(ruta_corte)
            logger.info(f'Ok insumo de ordenamiento para el corte {corte}')
        except:
            logger.info(f'No encontramos el insumo de ordenamiento y por esto cargaremos el backtesting de {corte}.')
            logger.info(f'Calculando los insumos de ordenamiento...')
            archivo = archivos_backtesting[corte]
            ruta_corte = f'{ruta}{archivo}'
            backtesting_corte = load_pickle(ruta_corte)
            insumos_360 = backtesting.modelo_360_full(backtesting_corte, params)
        finally:
            for t in insumos_360.keys():
                bd = insumos_360[t]['Discriminacion']
                bd['Corte'] = int(corte)
                if t not in insumos_ordenados:
                    logger.info(f'Construyendo monitoreo para seguir un rezago de  {t} cortes')
                    insumos_ordenados[t] = {}
                    bd_full = bd.copy()
                else:
                    logger.info(f'Añadiendo {corte} con el rezago {t} dentro del monitoreo ')
                    bd_full = insumos_ordenados[t]
                    bd_full = pd.concat([bd_full, bd] , axis = 0)
                insumos_ordenados[t] =  bd_full
            logger.info('------------------')
    
    logger.info(f'Monitoreo con los cortes: {cortes_totales}')
    etiquetas,metrics = manipulando_metricas(insumos_ordenados,cortes_totales, params)
    return etiquetas,metrics

## Funcion auxiliar
def asociando_nodos(ventana_corta,nodos_curva):
    """
        Funcion auxiliar utilizada para determinar los nodos donde se hara seguimiento el modelo
    
    Parameters
    ----------
    ventana_corta:
        pd.DataFrame que comprende en su indice el numero de observaciones 
        que se tienen en cada metrica calificada
    nodos_curva:
        List asociado a las observaciones que seran de seguimiento en el ordenamiento del modelo
    Returns
    -------
        Diccionario de medias moviles y desviaciones moviles junto con alertas de drift de datos
        del modelo.
    """
    nodos = []
    for n in nodos_curva:
        if n in ventana_corta.index:
            nodos.append(n)
        else:
            if n <=1:
                # n clientes asociados al decil
                n_estimate = ventana_corta.index.max()*n
            else:
                n_estimate = n
            # similitud entre el decil y restante
            n_ventana = (abs(ventana_corta.index-n_estimate)).tolist()
            # ubicando el n cliente asociado al decil
            n_ventana = n_ventana.index(min(n_ventana))
            n = ventana_corta.index[n_ventana]
            # guardando
            nodos.append(n)
    return nodos

# guardar informacion relevante
def update_info_pdf(pdf_info,tipo,info_save,parte):
    """
        Funcion auxiliar que guarda/actualiza mensajes o imagenes requeridas para 
        crear el pdf de alertas
    
    Parameters
    ----------
    pdf_info:
        Dict asociado a la informacion requerida para construir un pdf de alertas
    tipo:
        Str asociado al titulo de objeto que tiene "info_save" (Imagenes o mensajes)

    info_save:
        List asociado a los mensajes o imagenes que se desean producir en el pdf

    parte:
        Str asociado al titulo de alerta o mensaje
    Returns
    -------
        Diccionario de medias moviles y desviaciones moviles junto con alertas de drift de datos
        del modelo.
    """
    pdf_info['parte'] = parte
    if parte not in list(pdf_info.keys()):
        pdf_info[parte] = {tipo:info_save}
    else:
        if tipo not in list(pdf_info[parte].keys()):
            pdf_info[parte][tipo] = info_save
        else:
            pdf_info[parte][tipo] += info_save
    return pdf_info
    
## Funcion auxiliar comprende o consolida las alertas y las ventanas de tiempo
def calculando_niveles_de_alertas(metrics,etiquetas,params):
    """
        Funcion auxiliar que comprende o consolida las alertas y las ventanas de tiempo de media 
        y desviacion estandar segun la feature de metricas calculadas en metrics
    
    Parameters
    ----------
    metrics:
        Dict de feature de metricas
    etiquetas:
        Dict de etiquetas de cortes de tiempo del modelo

    params:
        Dict de parametros

    Returns
    -------
        Diccionario de medias moviles y desviaciones moviles junto con alertas de drift de datos
        del modelo.
    """
    # parametros 
    periodos_ventana_larga = params['n_cortes_ventana_lp']
    periodos_ventana_corta = params['n_cortes_ventana_cp']
    nivel_aceptabilidad = params['sd_aceptabilidad']

    # metricas
    z_actual = metrics['z_actual']
    z_inicial = metrics['z_inicial']
    z1 = metrics['z1']

    # cortes de tiempo    
    corte_ventana_movil = etiquetas['corte_ventana_movil_final']
    corte_actual = etiquetas['corte_actual']

    # definiendo la ventana sin el corte inicial ni el corte sobre el que se evaluan las metricas
    z2 = z1.drop([z_inicial.name], axis = 1)
    z2_ventanas = z2.drop([z_actual.name], axis = 1)
    # ajustando la dimensionalidad de las ventanas de tiempo
    if periodos_ventana_larga> np.min([periodos_ventana_larga,z2_ventanas.shape[1]]):
        periodos_ventana_larga = np.min([periodos_ventana_larga,z2_ventanas.shape[1]])
        logger.info(f'El parametro del numero de cortes de la ventana de largo plazo es mas grande que el set de observabilidad del modelo y se utilizaran todos los cortes...')
    logger.info(f'Calculando las medias y desviaciones estandar moviles')
    # calculando las medias moviles
    ventana_larga = z2_ventanas.T.rolling(window=periodos_ventana_larga).mean().T
    ventana_corta = z2_ventanas.T.rolling(window=periodos_ventana_corta).mean().T
    # calculando las desviaciones moviles
    ventana_larga_std = z2_ventanas.T.rolling(window=periodos_ventana_larga).std().T
    ventana_corta_std = z2_ventanas.T.rolling(window=periodos_ventana_corta).std().T
    # extrayendo el penultimo corte de tiempo o el corte de tiempo anterior al que se monitorea
    logger.info(f'Extrayendo las metricas moviles del corte: {corte_ventana_movil} previo al corte actual: {corte_actual}')
    # metrica en promedio
    try:
        ventana_larga2 = ventana_larga[corte_ventana_movil]
    except:
        ventana_larga2 = ventana_larga[int(corte_ventana_movil)]
    try:
        ventana_corta2 = ventana_corta[corte_ventana_movil]
    except:
        ventana_corta2 = ventana_corta[int(corte_ventana_movil)]
    # metrica en desviacion estandar
    try:
        ventana_larga_std2 = ventana_larga_std[corte_ventana_movil]
    except:
        ventana_larga_std2 = ventana_larga_std[int(corte_ventana_movil)]
    
    try:
        ventana_corta_std2 = ventana_corta_std[corte_ventana_movil]
    except:
        ventana_corta_std2 = ventana_corta_std[int(corte_ventana_movil)]
    logger.info(f'Calculando el nivel de alerta del modelo')
    std_used = pd.concat([ventana_corta_std2,ventana_larga_std2], axis = 1)
    #std_used = std_used + ventana_corta.std(axis = 1).to_frame().rename(columns={0:ventana_corta_std2.name})
    std_used = std_used.max(axis = 1)
    alerta_modelo = ventana_larga2-nivel_aceptabilidad*std_used
    windows = {
        'ventana_media_lp':ventana_larga,
        'ventana_media_cp':ventana_corta,
        'ventana_sd_lp':ventana_larga_std,
        'ventana_sd_cp':ventana_corta_std,
        'ventana_media_lp_t-1':ventana_larga2,
        'ventana_media_cp_t-1':ventana_corta2,
        'ventana_sd_lp_t-1':ventana_larga_std2,
        'ventana_sd_cp_t-1':ventana_corta_std2,
        'alerta_modelo':alerta_modelo
              }
    return windows

# Funcion auxiliar para generacion de imagenes en el pdf
def analisis_grafico_alertas_cambio_datos(windows,metrics,etiquetas,nodos,params):
    """
        Funcion auxiliar que genera y guarda imagenes asociadas a los impactos del drift de datos en el corto plazo
    
    Parameters
    ----------
    windows:
        Dict de feature de las metricas suavizadas en ventanas de tiempo
    metrics:
        Dict de feature de las metricas reales en cada corte
    nodos:
        List con los nodos u observaciones de interes en el ordenamiento
    params:
        Dict de parametros

    Returns
    -------
        Lista con las imagenes relevantes en el pdf para entender los movimientos de las metricas 
        o del desempeño del modelo en el corto plazo
    """
    # inputs o datos
    ventana_corta = windows['ventana_media_cp']
    ventana_larga = windows['ventana_media_lp']
    alerta_modelo = windows['alerta_modelo']
    z_inicial = metrics['z_inicial']
    z_actual = metrics['z_actual']
    # parametros o etiquetas
    product = params['product']
    z_name = params['z_name']
    rezago = params['rezago']
    figsize = params['figsize_drift_cp_datos']
    corte_inicial = etiquetas['corte_produccion']
    corte_ventana_movil = etiquetas['corte_ventana_movil_final']
    # Crear lista para almacenar figuras
    figuras = []
    # Colores para las líneas
    colors = plt.cm.Greys(np.linspace(0.2, 1, ventana_corta.shape[1]))
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    #plt.style.use('ggplot')
    ax1.grid(True, linestyle='dotted', alpha=0.2)
    
    # Graficar datos simulados
    for i, column in enumerate(ventana_corta.columns):
        if ventana_corta[column].dropna().shape[0] == 0:
            pass
        else:
            ax1.plot(ventana_corta.index, ventana_corta[column], label=f"{column} CP", color=colors[i])
            ax1.scatter(nodos,ventana_corta.loc[nodos, column], color=colors[i], s=25)
        if str(corte_ventana_movil) != str(column):
            pass
        else:
            ax1.plot(ventana_larga.index, ventana_larga[column], label=f"{column} LP", color=colors[i], linestyle='--')
            ax1.scatter(nodos,ventana_larga.loc[nodos, column], color=colors[i], s=25)
    
    ax1.plot(alerta_modelo.index, alerta_modelo.values, label = 'Alerta', color='red', linestyle = '--')
    ax1.scatter(nodos,alerta_modelo.loc[nodos].values,color='red', s = 25)
    
    for z_plot,label,col in [[z_inicial,str(corte_inicial),'green'],[z_actual,'Actual','orange']]:
        ax1.plot(z_plot.index,z_plot.values, label = [label], color = col)
        ax1.scatter(nodos,z_plot.loc[nodos],color = col, s = 25)
    ax1.set_ylabel(f'{z_name}')

    fig.suptitle(f'EVOLUCION DE METRICAS: {product} \n {z_name} con {rezago} periodo de retrazo de información durante la predicción')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=10)
    #plt.legend(loc='best', ncol=3)
    plt.show()
    figuras.append(fig)
    return figuras

# Funcion auxiliar para guardar los logs/msj que se reproduciran en el pdf en una lista
def update_msj(mensajes,mensajes_df,msj):
    """
        Funcion auxiliar que genera y guarda imagenes asociadas a los impactos del drift de datos en el corto plazo
    
    Parameters
    ----------
    mensajes:
        List de mensajes en formato de Str
    mensajes_df:
        List  de mensajes en formato pd.DataFrame
    msj:
        Str o pd.Dataframe asociado a lo que se imprimira en el pdf

    Returns
    -------
        List de mensajes en formato de Str y formato pd.DataFrame actualizado
    """
    if isinstance(msj,str):
        mensajes_df.append([None,None])
        logger.info(msj)
    else:
        mensajes_df.append([list(msj.T.round(2).index),msj.T.round(2).columns.name])
        display(msj)
        
    mensajes.append(msj)
    return mensajes,mensajes_df


def generacion_mensajes_cambio_datos_alertas(windows,metrics,nodos_copy, params):
    """
        Funcion auxiliar que genera los mensajes de alertas de drift de datos en el corto plazo
        del modelo durante la ejecucion en terminal y guarda los mensajes relevantes para el pdf
        
    
    Parameters
    ----------
    windows:
        Dict de feature de las metricas suavizadas en ventanas de tiempo
    metrics:
        Dict de feature de las metricas reales en cada corte
    nodos_copy:
        List con los nodos u observaciones de interes en el ordenamiento
    params:
        Dict de parametros

    Returns
    -------
        mensajes:
            List con los mensajes de texto en el pdf
        mensajes_df
            List con los pd.DataFrame de texto en el pdf
        key:
            True implica una activacion de alerta del modelo
    """
    # donde guardare la informacion
    mensajes = []
    mensajes_df = []
    # parametros
    nodos = nodos_copy.copy()
    periodos_ventana_corta = params['n_cortes_ventana_cp']
    periodos_ventana_larga = params['n_cortes_ventana_lp']
    z_name = params['z_name']
    # datos o inputs
    ventana_corta2 = windows['ventana_media_cp_t-1']
    ventana_larga2 = windows['ventana_media_lp_t-1']
    alerta_modelo = windows['alerta_modelo']
    z_actual = metrics['z_actual']
    # cuando la ventana corta es < ventana larga
    logger.info(f'Iniciando el analisis en los cambios de la estructura de los datos...')
    logger.info(f'Identificando los nodos del ordenamiento que tienen una tendencia decreciente (ventana corta de {periodos_ventana_corta} periodos inferior a la ventana larga de {periodos_ventana_larga} periodos).')
    cond = ventana_corta2[(ventana_corta2 < ventana_larga2)]
    msj  = f'Calculando el cambio porcentual entre el ultimo Backtesting generado vs la ventana corta con {periodos_ventana_corta} periodos...'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    # valores negativos implica un deterioro del modelo
    movimiento1 = (z_actual / ventana_corta2 -1)*100
    movimiento_strategico = np.round(movimiento1.loc[movimiento1.index.max()],2)
    msj  = f'El movimiento en el corto plazo corresponde al {movimiento_strategico}% bajo la visual de todo el set de datos, el cual se atribuye a negocio'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    # calculos sobre los nodos de ordenamiento de seguimiento
    movimiento2 = movimiento1.loc[movimiento1.index[movimiento1.index.isin(nodos)]].to_frame().rename(columns={0:z_name})
    msj = f'Y el movimiento de la metrica atribuible al modelo (en los nodos de ordenamiento de interes) es:'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    msj = movimiento2.T.round(2)
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    if (1 in params['nodos_curva'])| (1.0 in params['nodos_curva']):
        # quitamos de la alerta el nodo que comprende todo el dataset de observaciones
        nodos.remove(max(nodos))
        movimiento2 = movimiento1.loc[movimiento1.index[movimiento1.index.isin(nodos)]].to_frame().rename(columns={0:z_name})
    if movimiento1[movimiento1<0].shape[0]==0:
        msj = f'No identificamos deterioro en ningun nodo del ordenamiento en el corto plazo'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        alerta_cp = False
    else:   
        movimiento3 = movimiento2[movimiento2<0]
        if movimiento3.shape[0] == 0:
            msj = f'No identificamos deterioro en los nodos de ordenamiento '
            msj += f'sobre los cuales se hace seguimiento en el corto plazo.'
            msj += '\n'+f'Entonces, el deterioro se encuentra en una parte irrelevante del ordenamiento'
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            alerta_cp = False
        else:
            msj = f'% Deterioro sobre los nodos en los cuales se hace seguimiento: '
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            msj = movimiento3.T.round(2)
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            alerta_cp = True
        msj = f'Evaluando los nodos con tendencia decreciente sobre el comparativo '
        msj += f'del ultimo Backtesting contra la ventana corta de {periodos_ventana_corta} periodos'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        movimiento = movimiento1.loc[cond.index]
        msj = f'Y ubicando los cortes de deterioro..'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        movimiento = movimiento[movimiento<0]
        if movimiento.shape[0]>0:
            msj = f'Caida maxima: {np.round(movimiento.min(),3)}% asociado al nodo (Numero de obervacion: N) = {movimiento.idxmin(axis = 0)}.'
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            msj = f'Caida minima: {np.round(movimiento.max(),3)}% asociado al nodo (Numero de obervacion: N) = {movimiento.idxmax(axis = 0)}.'
        else:
            msj = f'No hubo deterioro sobre los nodos con tendencia decreciente '
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    logger.info(f'--------')
    msj = f'Calculando el cambio porcentual entre el ultimo Backtesting generado '
    msj += f'vs el nivel de la alerta permitido y evaluando alertas de deterioro...'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    alerta_generada = (z_actual/alerta_modelo-1)*100
    alerta_generada2 = alerta_generada[alerta_generada<0]
    alerta_generada2 = alerta_generada2.to_frame().rename(columns= {0:'Actual vs Alerta'})
    if alerta_generada2.shape[0] == 0:
        msj = f'No hubo alerta generada'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        alerta = False
    else:
        msj = f'% Deficiencia del modelo: (Backtesting_t /Alerta -1)*100:'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        msj = alerta_generada2.T.round(2)
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        alerta = True

    ## evaluando el diagnostico final del modelo
    msj = f'Diagnostico del modelo:'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    deterioro_inducido_modelo = movimiento_strategico >movimiento2.round(2)
    deterioro_inducido_modelo = deterioro_inducido_modelo[deterioro_inducido_modelo.values]
    key = True
    if alerta: # Deterioro tras comparar el ultimo backtesting contra la media movil+std
        msj = 'Alerta Grave/Fuerte por cambio estructural de los datos en el corto plazo'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        msj = 'esto es porque el ultimo backtesting esta por debajo que las alerta'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        if deterioro_inducido_modelo.shape[0]:
            msj = 'Sin modelo el deterioro hubiese sido peor. Revisar las condiciones de mercado y estrategia'
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        else:
            msj = 'Modelo descalibrado en los nodos de seguimiento: '
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            msj = deterioro_inducido_modelo.T
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    elif alerta_cp: # Deterioro tras comparar el ultimo backtesting contra la media movil
        msj = 'Alerta Nivel Medio por cambio estructural de los datos en corto plazo. Revisar factores macro o novedades en negocio'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        msj = 'El ultimo Backtesting tiene peores metricas que su nivel medio en el corto plazo aunque el backtesting no esta por debajo de la alerta'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        if deterioro_inducido_modelo.shape[0]:
            msj = 'Sin modelo el deterioro hubiese sido peor. Revisar las condiciones de mercado y estrategia'
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        else:
            msj = 'Modelo descalibrado en los nodos de seguimiento: '
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
            msj = deterioro_inducido_modelo.T
            mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    else:    
        key = False
        msj = 'OK. Modelo Eficiente!'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)

    return mensajes,mensajes_df,key

# pipeline 2: Alertas de cambios estructurales
def eval_alertas_cambio_estructural_de_datos(metrics,etiquetas,params):
    """
        Funcion que calcula las alertas de drift de datos en el corto plazo 
        junto con los mensajes e imagenes evidentes en el pdf
    
    Parameters
    ----------
        metrics:
            Dict de feature de las metricas reales en cada corte
        etiquetas:
            Dict de etiquetas de cortes de tiempo del modelo
        params:
            Dict de parametros

    Returns
    -------
        pdf_info:
            Dict asociado a la informacion requerida para construir un pdf de alertas de drift
        windows:
            Dict de feature de las metricas suavizadas en ventanas de tiempo
        nodos:
            List con los nodos u observaciones de interes en el ordenamiento
        key:
            True implica una activacion de alerta del modelo por drift de datos
    """
    # parametros 
    nodos_curva = params['nodos_curva']
    # informacion asociada al reporte y las alertas 
    parte = 'CAMBIO ESTRUCTURAL EN LOS DATOS O ENTORNO EN EL CORTO PLAZO' # posicion en el pdf de reporte de metricas
    # Espacio donde quedaran las alertas
    pdf_info = {}
    # calculando las alertas y metricas
    windows = calculando_niveles_de_alertas(metrics,etiquetas,params)
    # buscamos los nodos que mas se asocian al monitoreable
    nodos = asociando_nodos(windows['ventana_media_cp'],nodos_curva)
    # generando el analisis grafico
    figuras = analisis_grafico_alertas_cambio_datos(windows,metrics,etiquetas,nodos,params)
    logger.info(f'Añadiendo las graficas sobre los cambios en los datos')
    pdf_info = update_info_pdf(pdf_info,'figuras',figuras,parte)
    # generando mensajes de alertas
    mensajes,mensajes_df,key = generacion_mensajes_cambio_datos_alertas(windows,metrics,nodos, params)
    logger.info(f'Añadiendo los mensajes sobre los cambios en los datos')
    pdf_info = update_info_pdf(pdf_info,'msj',mensajes,parte)
    pdf_info = update_info_pdf(pdf_info,'tickets_df',mensajes_df,parte)
    
    ################ Agregar #####################
    print(f"🚨 Drift alert: {key}")
    logger.info(f"🚨 Resultado drift_alert = {key}")
    
    return pdf_info,windows,nodos,key

# Funcion auxiliar para calcular el drawdown maximo
def drawndown_calc(z_actual,z_inicial,column):
    """
        Funcion auxiliar para calcular el drawdown maximo entre 
            las metricas e un corte inicial: z_inicial
            y las metricas e un corte final: z_actual
            
    """
    drawdown_max = (z_actual/z_inicial.values-1)*100
    validate = drawdown_max[drawdown_max>=0].index.tolist()
    for i in validate:
        drawdown_max.loc[i] = 0
    drawdown_max = drawdown_max.to_frame()
    drawdown_max.columns =  [column]
    return drawdown_max

# Funcion auxiliar
def generacion_mensajes_drawdown(drawdown,nodos_copy,params):
    """
        Funcion auxiliar que genera los mensajes de alertas de drawdown y mantenimiento del modelo
        durante la ejecucion en terminal y guarda los mensajes relevantes para el pdf
    
    Parameters
    ----------
    drawdown:
        pd.DataFrame con las metricas de maxima caida o maximo deterioro
    nodos_copy:
        List con los nodos u observaciones de interes en el ordenamiento
    params:
        Dict de parametros

    Returns
    -------
        mensajes:
            List con los mensajes de texto en el pdf
        mensajes_df
            List con los pd.DataFrame de texto en el pdf
        key:
            True implica una activacion de alerta del modelo
    """
    # donde guardare la informacion
    mensajes = []
    mensajes_df = []
    # parametros:
    alerta_drawdown = params['Alerta_drawdown']
    product = params['product']
    nodos = nodos_copy.copy()
    if (1 in params['nodos_curva'])| (1.0 in params['nodos_curva']):
        nodos.remove(max(nodos))
    # iniciando calculos
    drawdown.columns.name = f'% Δ {product}'
    if any(drawdown<= alerta_drawdown):
        msj  = f'Alerta: Drawdown superior al restringido {alerta_drawdown}% en algunos nodos de la curva de ordenamiento:'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        drawdown2 = drawdown[(drawdown<= alerta_drawdown).apply(any, axis= 1)]
        # filtrando los nodos relevantes
        nodos_view = asociando_nodos(drawdown,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        for j in nodos_view:
            if j not in drawdown.index.tolist():
                msj = f'No se reviso el Nodo (N) asociado al numero de observaciones = {j}'
                mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        nodos_filt = sorted(list(set(nodos_view + nodos)))
    
        msj = drawdown2.loc[drawdown2.index[drawdown2.index.isin(nodos_filt)]]
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)


    msj  = f'Analizando el drawdown en los nodos de interes...'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    drawdown1 = drawdown.loc[nodos]
    drawdown1_cond = drawdown1[(drawdown1<= alerta_drawdown).apply(any, axis= 1)]
    #drawdown1_cond = drawdown1.astype(int).sum(axis  =1)>0
    #drawdown1_cond = drawdown1_cond.values
    if drawdown1_cond.shape[0]>0:
        key = True
        msj = f'Alerta: Drawdown superior al restringido {alerta_drawdown}% en algun nodo de interes'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        msj = f'Analizar el reporte para evaluar la recalibracion del modelo'
    else:
        key = False
        msj = f'Ok modelo sin requerir recalibracion!'
        mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
        msj = f'No hubo alerta de Drawdown en ningun nodo de interes '
        msj += f'ya que no supero el restringido {alerta_drawdown}%'
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    msj = drawdown1
    mensajes,mensajes_df = update_msj(mensajes,mensajes_df,msj)
    return mensajes,mensajes_df,key

def analisis_grafico_alertas_mantenimiento(metrics,etiquetas,params):
    """
        Funcion auxiliar que genera los las graficas de alertas de mantenimiento 
        durante la ejecucion en terminal y las guarda los graficas relevantes para el pdf
    
    Parameters
    ----------
        metrics:
            Dict de feature de las metricas reales en cada corte.
        etiquetas:
            Dict que contiene las etiquetas relevantes de los cortes de tiempo que se procesan
        params:
            Dict de parametros.

    Returns
    -------
        figuras:
            List figuras o imagenes que se presentaran en el pdf
    """
    # extrayendo el set de datos
    z_inicial = metrics['z_inicial']
    z1 = metrics['z1']
    x1 = metrics['x1']
    y1 = metrics['y1']
    # parametros o etiquetas
    product = params['product']
    rezago = params['rezago']
    if rezago >1:
        rezago = f'{rezago} cortes'
    else:
        rezago = f'{rezago} corte'

    z_name = params['z_name']
    x_name = params['x_name']
    y_name = params['y_name']
    z_name = params['z_name']
    figsize = params['figsize_mantenimiento']
    # eliminando el corte de entrenamiento/reentrenamiento
    z2 = z1.drop([z_inicial.name], axis = 1)
    # todos los cortes sin las metricas de backtesting de entrenamineto/reentrenamiento
    cortes_totales2 = etiquetas['cortes_totales2'] 
    # Crear lista para almacenar figuras
    figuras = []
    # generando graficos
    title = ['Backtesting','Δ Marginal/Pendiente del Ordenamiento','Δ Backtesting/ ΔT']
    for t,altura in enumerate([z2,z2.diff(),z2.diff(axis=1)]):
        # Crear subgráficos con 1 fila y 2 columnas
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]), subplot_kw={'projection': '3d'})
        fig.suptitle(f'EVOLUCION DE METRICAS: {product} \n \n {title[t]}: {z_name}. \n Rezago De Informacion: {rezago}')
        #plt.style.use('ggplot')
        axs[0].grid(True, linestyle='dotted', alpha=0.2)
        axs[1].grid(True, linestyle='dotted', alpha=0.2)
        # Primer gráfico: a la izquierda
        ax1 = axs[0]
        ax1.plot_surface(y1, x1, altura, cmap='viridis')
        #ax1.set_title(f'{title[t]}: {z_name}. Corte: {rezago}')
        ax1.set_xlabel(f'{y_name}')
        ax1.set_ylabel(f'{x_name}')
        ax1.set_zlabel(f'{z_name}')
        ax1.set_yticks([int(i) for i in cortes_totales2])
        ax1.set_yticklabels([f'{i}' for i in cortes_totales2])
        
        # Segundo gráfico: a la derecha
        ax2 = axs[1]
        ax2.plot_surface(x1, y1, altura, cmap='viridis')
        #ax2.set_title(f'{title[t]}: {z_name}. Corte: {rezago}')
        ax2.set_xlabel(f'{x_name}')
        ax2.set_ylabel(f'{y_name}')
        ax2.set_zlabel(f'{z_name}')
        ax2.set_xticks([int(i) for i in cortes_totales2])
        ax2.set_xticklabels([f'{i}' for i in cortes_totales2])
        # Mostrar el gráfico
        plt.show()
        figuras.append(fig)
    return figuras

# pipeline 3: Alertas de mantenimiento del modelo
def eval_alertas_drawdown(pdf_info,windows,metrics,etiquetas,nodos, params):
    """
        Funcion principal que genera las graficas de alertas de mantenimiento y los mensajes
        del drawdown durante la ejecucion en terminal y las guarda en el diccionario
        "pdf_info" para reproducirlo en el pdf
    
    Parameters
    ----------
        windows:
            Dict de feature de las metricas suavizadas en ventanas de tiempo
        metrics:
            Dict de feature de las metricas reales en cada corte
        nodos:
            List con los nodos u observaciones de interes en el ordenamiento
        pdf_info:
            dict que contiene las imagenes y mensajes que iran al reporte del pdf
        params:
            Dict de parametros

    Returns
    -------
        pdf_info:
            dict que contiene las imagenes y mensajes actualizados de este nodo
        key:
            True implica una activacion de alerta del modelo 
            por mantenimiento.
    """
    # parametros
    periodos_ventana_corta = params['n_cortes_ventana_cp']
    periodos_ventana_larga = params['n_cortes_ventana_lp']
    # datos o inputs
    ventana_corta2 = windows['ventana_media_cp_t-1']
    ventana_larga2 = windows['ventana_media_lp_t-1']
    # ajuste de mensaje
    n_shape_lp = windows['ventana_media_lp'].shape[1]
    n_shape_cp = windows['ventana_media_cp'].shape[1]
    if n_shape_lp<=periodos_ventana_larga:
        n_t_lp = n_shape_lp
    else:
        n_t_lp = periodos_ventana_larga
    if n_shape_cp<=periodos_ventana_corta:
        n_t_cp = n_shape_cp
    else:
        n_t_cp = periodos_ventana_corta        
        
    z_inicial = metrics['z_inicial']
    z_actual = metrics['z_actual']
    # informacion asociada al reporte y las alertas 
    parte = 'DETERIORO MAXIMO EN LAS METRICAS' # posicion en el pdf de reporte de metricas
    ### añadiendo graficos de mantenimiento
    logger.info(f'Calculando o añadiendo las graficas sobre el mantenimiento del modelo')
    figuras = analisis_grafico_alertas_mantenimiento(metrics,etiquetas,params)
    pdf_info = update_info_pdf(pdf_info,'figuras',figuras,parte)
    logger.info(f'Iniciando el calculo de metricas Drawdown...')
    logger.info(f'Comparando metricas actuales vs metricas desde la fecha de producción...')
    drawdown_actual_ini = drawndown_calc(z_actual,z_inicial,'Actual vs Initial')
    logger.info(f'Calculando metricas actuales vs {periodos_ventana_corta},{periodos_ventana_larga} periodos ...')
    drawdown_actual_ventana_corta = drawndown_calc(z_actual,ventana_corta2,f'Actual vs μ {n_t_cp} periods')
    drawdown_actual_ventana_larga = drawndown_calc(z_actual,ventana_larga2,f'Actual vs μ {n_t_lp} periods')
    drawdown = pd.concat([drawdown_actual_ini,drawdown_actual_ventana_corta,drawdown_actual_ventana_larga], axis = 1)
    ### añadiendo los mensajes
    logger.info(f'Añadiendo los mensajes sobre el mantenimiento del modelo')
    mensajes,mensajes_df,key = generacion_mensajes_drawdown(drawdown,nodos, params)
    pdf_info = update_info_pdf(pdf_info,'msj',mensajes,parte)
    pdf_info = update_info_pdf(pdf_info,'tickets_df',mensajes_df,parte)
    
    ######################## Agregar #######################
    print(f"🌿 Sustainability alert: {key}")
    logger.info(f"🌿 Resultado sustainability_alert = {key}")

    return pdf_info,key

## Funcion Auxiliar para manipular el texto dentro del pdf a construir
def dividir_texto(texto, max_chars=80):
    """
        Funcion que recibe un mensaje de texto en string y retorna el mismo string
        pero ajustado a los guiones de la estructura del formato de pdf
    """
    palabras = texto.split()
    lineas = []
    linea_actual = ""
    for palabra in palabras:
        if len(linea_actual + palabra) <= max_chars:
            linea_actual += palabra + " "
        else:
            lineas.append(linea_actual.strip())
            linea_actual = palabra + " "
    lineas.append(linea_actual.strip())
    return lineas

## Pipeline de generacion de reporte
## Pipeline de generacion de reporte
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os, boto3, pandas as pd

def generar_reporte_pdf(pdf_info, params):
    estructura_pdf = params['estructura_pdf']
    ruta_base_s3 = estructura_pdf['ruta_monitoreo']
    nombre_pdf = estructura_pdf['nombre_pdf']

    # === Ruta temporal local permitida ===
    ruta_local_tmp = "/tmp/monitoring"
    os.makedirs(ruta_local_tmp, exist_ok=True)

    pdf_path_local = os.path.join(ruta_local_tmp, nombre_pdf)
    c = canvas.Canvas(pdf_path_local, pagesize=A4)
    width, height = A4

    # === Parámetros del formato ===
    start_text_cm = estructura_pdf['start_text_cm']
    margen_sup_inf_cm = estructura_pdf['margen_sup_inf_cm']
    renglon_cm = estructura_pdf['renglon_cm']
    margen_izq_cm = estructura_pdf['margen_izq_cm']
    y_graph_cm = estructura_pdf['y_graph_cm']
    width_graph_cm = estructura_pdf['width_graph_cm']
    height_graph_cm = estructura_pdf['height_graph_cm']
    margen_izq_graph_cm = estructura_pdf['margen_izq_graph_cm']

    # Conversión a puntos (cm → puntos PDF)
    from reportlab.lib.units import cm
    start_text = start_text_cm * cm
    margen_inferior = margen_sup_inf_cm * cm
    margen_superior = margen_sup_inf_cm * cm
    renglon = renglon_cm * cm
    margen_izq = margen_izq_cm * cm
    y_graph = y_graph_cm * cm
    margen_izq_graph = margen_izq_graph_cm * cm
    width_graph = width_graph_cm * cm
    height_graph = height_graph_cm * cm

    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.red),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ])

    # === Generar PDF con texto y tablas ===
    for i in list(pdf_info.keys()):
        if i != 'parte':
            y = start_text
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margen_izq, y, "INFORME DE METRICAS")
            y -= renglon
            c.setFont("Helvetica", 13)
            c.drawString(margen_izq, y, f"{i.upper()}")
            y = start_text - margen_superior

            texto = pdf_info[i]['msj']
            tickets_df = pdf_info[i]['tickets_df']

            for t, text in enumerate(texto, start=1):
                if isinstance(text, str):
                    for linea in text.split("\n"):
                        c.setFont("Helvetica", 11)
                        c.drawString(margen_izq, y, linea[:120])
                        y -= renglon
                        if y <= margen_inferior:
                            c.showPage()
                            y = start_text - margen_superior
                elif isinstance(text, pd.DataFrame):
                    y -= renglon
                    text = text.round(2).reset_index()
                    data = [text.columns.to_list()] + text.values.tolist()
                    table = Table(data)
                    table.setStyle(style)
                    table.wrapOn(c, width - 2*margen_izq, height)
                    table_height = table._height
                    if y - table_height <= margen_inferior:
                        c.showPage()
                        y = start_text - margen_superior
                    table.drawOn(c, margen_izq, y - table_height)
                    y -= table_height + renglon

            c.showPage()

            figuras = pdf_info[i]['figuras']
            for t, figura in enumerate(figuras, start=1):
                ruta_img_local = os.path.join(ruta_local_tmp, f"imagen_{i.replace(' ', '_')}{t}.png")
                figura.savefig(ruta_img_local)
                c.drawImage(ruta_img_local, margen_izq_graph, y_graph,
                            width=width_graph, height=height_graph,
                            preserveAspectRatio=True)
                c.showPage()

    # Guardar el PDF localmente
    c.save()
    print(f"✅ PDF local generado: {pdf_path_local}")

    # === Subir a S3 ===
    s3 = boto3.client("s3")
    bucket_name = ruta_base_s3.split("/")[2]
    prefix = "/".join(ruta_base_s3.split("/")[3:])

    # Subir PDF
    s3.upload_file(pdf_path_local, bucket_name, f"{prefix}/{nombre_pdf}")
    print(f"📤 PDF subido a S3: s3://{bucket_name}/{prefix}/{nombre_pdf}")

    # Subir imágenes
    for i in list(pdf_info.keys()):
        if i != "parte":
            figuras = pdf_info[i]['figuras']
            for t, figura in enumerate(figuras, start=1):
                filename = f"imagen_{i.replace(' ', '_')}{t}.png"
                img_local_path = os.path.join(ruta_local_tmp, filename)
                s3.upload_file(img_local_path, bucket_name, f"{prefix}/{filename}")
                print(f"📤 Imagen subida a S3: s3://{bucket_name}/{prefix}/{filename}")

    print("✅ Todas las imágenes y PDF subidas correctamente.")
    return ''
