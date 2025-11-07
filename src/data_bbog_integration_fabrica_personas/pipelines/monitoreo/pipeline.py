"""
Pipeline de la capa monitoreo
"""

## aqui : si al menos una es True se levanta el reentrenamiento

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    extraer_metricas,
    eval_alertas_cambio_estructural_de_datos,
    eval_alertas_drawdown,
    generar_reporte_pdf
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extraer_metricas,
                inputs=[
                    "parameters",
                ],
                outputs=["tickets","metrics"],
                name="extraer_metricas_nodo",
            ),
            node(
                func=eval_alertas_cambio_estructural_de_datos,
                inputs=[
                    "metrics",
                    "tickets",
                    "parameters",
                ],
                outputs=["pdf_msj","windows_metrics","nodes_target","drift_alert"], ## aqui
                name="eval_alertas_drift_node",
            ),

            node(
                func=eval_alertas_drawdown,
                inputs=[
                    "pdf_msj",
                    "windows_metrics",
                    "metrics",
                    "tickets",
                    "nodes_target",
                    "parameters",
                ],
                outputs=["insumos_pdf","sustainability_alert"], #### aqui
                name="eval_alertas_drawdown_node",
            ),
            node(
                func=generar_reporte_pdf,
                inputs=[
                    "insumos_pdf",
                    "parameters",
                ],
                outputs="None", #MODIFICADO JF
                name="creat_pdf_node",
            ),
        ]
    )
