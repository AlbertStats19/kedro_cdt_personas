"""
Pipeline de la capa models
kedro run --pipeline=models
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    Experimentacion_balanceos,
    red_neuronal,
    train_xgboost_with_cv,
    train_random_forest_with_cv,
    evaluate_models_for_all,
    compare_metrics_models,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # bases
            node(
                func=Experimentacion_balanceos,
                inputs=[
                    "X_train_scaler_master",
                    "Y_train_feature",
                    "X_test_feature",
                    "Y_test_feature",
                    "scaler_transform",
                    "parameters",
                ],
                outputs="info_save",
                name="model_input_balance_target_node",
            ),
            # Modelo red neuronal
            node(
                func=red_neuronal,
                inputs=["info_save", "parameters"],
                outputs="info_save1",
                name="run1_node",
            ),
            # Modelo xgboots
            node(
                func=train_xgboost_with_cv,
                inputs=["info_save1", "parameters"],
                outputs="info_save2",
                name="run2_node",
            ),
            # Modelo  Random Forest
            node(
                func=train_random_forest_with_cv,
                inputs=["info_save2", "parameters"],
                outputs="info_save3",
                name="run3_node",
            ),
            # Modelo red neuronal
            node(
                func=red_neuronal,
                inputs=["info_save3", "parameters"],
                outputs="info_save4",
                name="run4_node",
            ),
            # Modelo xgboots
            node(
                func=train_xgboost_with_cv,
                inputs=["info_save4", "parameters"],
                outputs="info_save5",
                name="run5_node",
            ),
            # Modelo random forest
            node(
                func=train_random_forest_with_cv,
                inputs=["info_save5", "parameters"],
                outputs="info_save6",
                name="run6_node",
            ),
            # Modelo red neuronal
            node(
                func=red_neuronal,
                inputs=["info_save6", "parameters"],
                outputs="info_save7",
                name="run7_node",
            ),
            # Modelo xgboots
            node(
                func=train_xgboost_with_cv,
                inputs=["info_save7", "parameters"],
                outputs="info_save8",
                name="run8_node",
            ),
            # Modelo Random Forest
            node(
                func=train_random_forest_with_cv,
                inputs=["info_save8", "parameters"],
                outputs="info_save9",
                name="run9_node",
            ),
            # Modelo red neuronal
            node(
                func=red_neuronal,
                inputs=["info_save9", "parameters"],
                outputs="info_save10",
                name="run10_node",
            ),
            # Modelo xgboots
            node(
                func=train_xgboost_with_cv,
                inputs=["info_save10", "parameters"],
                outputs="info_save11",
                name="run11_node",
            ),
            # Modelo random forest
            node(
                func=train_random_forest_with_cv,
                inputs=["info_save11", "parameters"],
                outputs="info_save12",
                name="run12_node",
            ),
            # todas las metricas
            node(
                func=evaluate_models_for_all,
                inputs=[
                    "info_save12",  # recibe lo ultimo que entrega el ultimo nodo
                    "parameters",
                ],
                outputs="info_save_last_all",  # no tocar
                name="evaluate_models_for_all_node",
            ),
            # compare_metrics_models
            node(
                func=compare_metrics_models,
                inputs=["info_save_last_all", "parameters"],  # no tocar
                outputs="info_save_all",  # ultimo pronostico
                name="compare_metrics_models_node",
            ),
        ]
    )
