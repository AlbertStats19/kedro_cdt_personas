"""
Pipeline de la capa feature
kedro run --pipeline=feature
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    calculate_new_variables_pd,
    modelo_homologacion_regiones,
    homologate_region,
    eliminar_columnas,
    preprocesar_feature_df,
    separar_características,
    calcular_importancia,
    seleccionar_características,
    filtrar_columnas_df,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_new_variables_pd,
                inputs=["master_primary_fp", "parameters"],
                outputs="feature_df",
                name="feature_new_node",
            ),
            node(
                func=modelo_homologacion_regiones,
                inputs=["parameters"],
                outputs="homologate_region_model",
                name="model_homologate_region_node",
            ),
            node(
                func=homologate_region,
                inputs=["feature_df", "homologate_region_model"],
                outputs="homologate_region_df",
                name="feature_homologate_region_node",
            ),
            node(
                func=eliminar_columnas,
                inputs=["homologate_region_df", "parameters"],
                outputs="feature_df_filt",
                name="feature_eliminar_columnas_node",
            ),
            node(
                func=preprocesar_feature_df,
                inputs=["feature_df_filt", "parameters"],
                outputs="feature_preprocesado_df",
                name="preprocesar_feature_df_node",
            ),
            node(
                func=separar_características,
                inputs=["feature_preprocesado_df", "parameters"],
                outputs=["X_feature", "y_feature"],
                name="feature_separar_características_node",
            ),
            node(
                func=calcular_importancia,
                inputs=["X_feature", "y_feature", "parameters"],
                outputs="importance_df",
                name="feature_calcular_importancia_node",
            ),
            node(
                func=seleccionar_características,
                inputs=["importance_df", "feature_df", "parameters"],
                outputs="feature_selected_list",
                name="feature_seleccionar_características_node",
            ),
            node(
                func=filtrar_columnas_df,
                inputs=["homologate_region_df", "feature_selected_list", "parameters"],
                outputs="master_feature_fp",
                name="feature_filtrar_columnas_node",
            ),
        ]
    )
