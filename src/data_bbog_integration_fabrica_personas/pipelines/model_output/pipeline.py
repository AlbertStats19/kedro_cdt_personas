"""
Pipeline de la capa model_output
kedro run --pipeline=model_output
"""

from kedro.pipeline import Pipeline, node

# import data_bbog_integration_fabrica_personas.pipelines.feature.nodes as feature
from .nodes import (
    calificar_base,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create a Kedro pipeline for loading a transform pipeline, applying transformations,
    loading a model, and making predictions.

    Returns:
        Pipeline: A Kedro pipeline object containing the defined nodes.
    """
    return Pipeline(
        [
            node(
                func=calificar_base,
                inputs=[
                    "df_input_mo",
                    "feature_selected_list",
                    "homologate_region_model",
                    "info_save_select",
                    "parameters",
                ],
                outputs="base_calificada_nueva",
                name="calificar_base_node",
            ),
        ]
    )
