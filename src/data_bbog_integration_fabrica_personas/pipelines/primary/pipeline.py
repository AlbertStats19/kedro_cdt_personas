"""
Pipeline de la capa primary
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_business_data_pd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_business_data_pd,
                inputs=["master_intermediate_fp", "parameters"],
                outputs="master_primary_fp",
                name="primary_filter_business_data_pd_node",
            ),
        ]
    )
