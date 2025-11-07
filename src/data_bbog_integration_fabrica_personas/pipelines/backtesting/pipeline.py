"""
Pipeline de la capa backtesting
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_data_pd,
    combinar_predicciones_reales,
    generate_metrics_all,
    generate_ks_all,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data_pd,
                inputs=[
                    "feature_selected_list",
                    "homologate_region_model",
                    "info_save_select",
                    "parameters",
                ],
                outputs=["data_process", "nombre_modelo"],
                name="backtesting_data_node",
            ),
            node(
                func=combinar_predicciones_reales,
                inputs=["data_process", "nombre_modelo", "parameters"],
                outputs=["data_process1", "insumo_modelo_360"],
                name="backtesting_combinar_node",
            ),
            node(
                func=generate_metrics_all,
                inputs=["data_process1", "parameters"],
                outputs="data_metrics_backtesting",
                name="backtesting_metrics_node",
            ),
            node(
                func=generate_ks_all,
                inputs=["data_metrics_backtesting", "parameters"],
                outputs="data_backtesting",
                name="backtesting_ks_node",
            ),
        ]
    )
