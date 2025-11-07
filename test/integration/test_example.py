# -*- coding: utf-8 -*-
"""
module: Integration tests
"""


import pandas as pd
from typing import Dict, List


def process(input_paths: Dict[str, List[str]]) -> pd.DataFrame:

    # Check that all dfs have unique indices
    check_no_duplicate_idx([])

    # Create pre-master data table
    pre_mdt = create_mdt(["a", "b", "c"])

    # Return pre-mdt as df
    return pre_mdt


def check_no_duplicate_idx(df_list: List[pd.DataFrame]) -> None:
    """
    This functions checks that all dfs in list have unique indices.

    :param df_list: list of dfs to be checked
    """

    for df in df_list:
        assert (
            df.index.duplicated().sum() == 0
        ), "One of the dfs has repeated indices. Cannot concatenate."


def create_mdt(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    This function receives a list of dataframes and concatenates them
    """

    return df_list


class TestProcessPreparation:

    S3_TEST_DATA_PATH = {"cifin": ["s3://adl-landing-test/sample/DATA_EXAMPLE.csv"]}

    def test_process_preparation(self):
        pre_mdt = process(self.S3_TEST_DATA_PATH)

        necessary_columns = ["a", "b", "c"]

        for column in necessary_columns:
            assert column in pre_mdt
