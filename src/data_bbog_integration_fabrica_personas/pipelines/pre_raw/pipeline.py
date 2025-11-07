"""
This is a boilerplate pipeline 'pre_raw'
generated using Kedro 0.18.14
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    concat_dataframes_pl_pd,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=concat_dataframes_pl_pd,
                inputs=["parameters"],
                outputs="master_concat_fp",
                name="raw_concat_dataframes_pl_pd_node",
            ),
        ]
    )
