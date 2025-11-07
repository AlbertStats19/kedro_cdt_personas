"""
Pipeline de la capa modelo_360
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import anexos_modelo_360, cargar_bases, reshape_dataframe, union_frames


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=anexos_modelo_360,
                inputs=["parameters"],
                outputs=["df_inputs", "tenencias_clientes"],
                name="anexos_formato_entrega",
            ),
            node(
                func=cargar_bases,
                inputs=["tenencias_clientes", "parameters"],
                outputs="pronosticos",
                name="cargar_pronosticos",
            ),
            node(
                func=reshape_dataframe,
                inputs=["pronosticos", "parameters"],
                outputs="campana",
                name="modelo_360",
            ),
            node(
                func=union_frames,
                inputs=["campana", "df_inputs", "parameters"],
                outputs="campana_final",
                name="modelo_360_formato",
            ),
        ]
    )
