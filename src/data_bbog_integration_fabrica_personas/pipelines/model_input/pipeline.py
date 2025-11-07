"""
Pipeline de la capa model_input
kedro run --pipeline=model_input
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    # feature_selec_pd,
    train_test_split_pd,
    treatment_null_numbers,
    treatment_outliers,
    run_numeric_values,
    treatment_null_categorical,
    run_categorical_values,
    One_Hot_encoding_keys,
    One_Hot_encoding_func,
    scale,
    save_transformer,
    min_max_scaler_pd,
    # balance_target_variable_pd_oscar
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_pd,
                inputs=["master_feature_fp", "parameters"],
                outputs=[
                    "X_train_feature",
                    "Y_train_feature",
                    "X_test_feature",
                    "Y_test_feature",
                ],
                name="model_input_split_data_node",
            ),
            node(
                func=treatment_null_numbers,
                inputs=["X_train_feature", "parameters"],
                outputs=["imputer_model_numeric", "order_col_numeric"],
                name="model_imputer_numeric_null_node",
            ),
            node(
                func=treatment_outliers,
                inputs=["X_train_feature", "parameters"],
                outputs="outliers_adj",
                name="variable_imputer_outliers",
            ),
            node(
                func=run_numeric_values,
                inputs=[
                    "X_train_feature",
                    "order_col_numeric",
                    "imputer_model_numeric",
                    "outliers_adj",
                    "parameters",
                ],
                outputs="train_feature1",
                name="model_input_numeric_data_values",
            ),
            node(
                func=treatment_null_categorical,
                inputs=["train_feature1", "parameters"],
                outputs="model_imputer_model_categorical",
                name="model_imputer_categorical_null_node",
            ),
            node(
                func=run_categorical_values,
                inputs=[
                    "train_feature1",
                    "model_imputer_model_categorical",
                    "parameters",
                ],
                outputs="train_feature2",
                name="model_input_categorical_data_values",
            ),
            node(
                func=One_Hot_encoding_keys,
                inputs=["train_feature2", "parameters"],
                outputs="reindex_OneHotEncoding",
                name="model_imputer_drop_first_node",
            ),
            node(
                func=One_Hot_encoding_func,
                inputs=["train_feature2", "reindex_OneHotEncoding", "parameters"],
                outputs="train_feature3",
                name="model_input_data_values",
            ),
            node(
                func=scale,
                inputs=["train_feature3", "parameters"],
                outputs=["model_imputer_scale", "col_names_all"],
                name="model_imputer_Scale",
            ),
            node(
                func=min_max_scaler_pd,
                inputs=[
                    "train_feature3",
                    "parameters",
                    "model_imputer_scale",
                    "col_names_all",
                ],
                outputs="X_train_scaler_master",
                name="model_input_scaler_node",
            ),
            node(
                func=save_transformer,
                inputs=[
                    "X_train_feature",  # guardar el nombre de todas las
                    # columnas antes del procesamiento
                    "order_col_numeric",
                    "col_names_all",
                    "imputer_model_numeric",
                    "model_imputer_model_categorical",
                    "outliers_adj",
                    "reindex_OneHotEncoding",
                    "model_imputer_scale",
                    "parameters",
                ],
                outputs="scaler_transform",
                name="model_imputer_transform",
            ),
        ]
    )
