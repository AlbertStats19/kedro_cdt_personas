"""
Pipeline de la capa intermediate
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_data_segment_pd, filter_data_prod_pd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_data_segment_pd,
                inputs=["master_raw_fp", "parameters"],
                outputs="master_filter_data_segment_fp",
                name="intermediate_filter_data_segment_pd_node",
            ),
            node(
                func=filter_data_prod_pd,
                inputs=["master_filter_data_segment_fp", "parameters"],
                outputs="master_intermediate_fp",
                name="intermediate_filter_data_prod_pd_node",
            ),
        ]
    )
