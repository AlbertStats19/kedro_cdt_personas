"""
Pipeline de la capa model_selection
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    generate_modelo_produccion,
    calc_metrics_before_backtesting,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_modelo_produccion,
                inputs=["info_save_all", "parameters"],
                outputs="info_save_temp",
                name="Infraestructura_produccion",
            ),
            node(
                func=calc_metrics_before_backtesting,
                inputs=["info_save_all", "info_save_temp", "parameters"],
                outputs="info_save_select",
                name="Metricas_modelo_produccion",
            ),
        ]
    )
