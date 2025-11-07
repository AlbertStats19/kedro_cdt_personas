"""
Pipeline de la capa raw
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    validar_columnas,
    convertir_a_minusculas,
    standardize_strings,
    values_to_null,
    change_dtypes,
    validate_unique_id_period_pd,
    create_targets_pd,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validar_columnas,
                inputs=["master_concat_fp", "parameters"],
                outputs="master_concat_fp_validation",
                name="raw_validar_columnas_node",
            ),
            node(
                func=convertir_a_minusculas,
                inputs=["master_concat_fp_validation", "parameters"],
                outputs="master_minusculas",
                name="raw_minusculas_columnas_node",
            ),
            node(
                func=standardize_strings,
                inputs=["master_minusculas", "parameters"],
                outputs="master_standard",
                name="raw_standardize_strings_node",
            ),
            node(
                func=values_to_null,
                inputs=["master_standard"],
                outputs="master_null",
                name="raw_values_to_nulls_node",
            ),
            node(
                func=change_dtypes,
                inputs=["master_null", "parameters"],
                outputs="master_change_dtypes",
                name="raw_change_dtypes_node",
            ),
            node(
                func=validate_unique_id_period_pd,
                inputs=["master_change_dtypes", "parameters"],
                outputs="master_validate_unique_fp",
                name="raw_validate_unique_id_period_pd_node",
            ),
            node(
                func=create_targets_pd,
                inputs=["master_validate_unique_fp", "parameters"],
                outputs="master_raw_fp",
                name="raw_create_targets_pd_node",
            ),
        ]
    )
