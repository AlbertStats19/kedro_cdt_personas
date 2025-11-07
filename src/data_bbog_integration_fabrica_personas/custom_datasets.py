from typing import Any, Dict
from kedro.io import AbstractDataset
from kedro.io.core import get_protocol_and_path
import fsspec
import polars as pl
import pandas as pd
import awswrangler as wr
import numpy as np


class PolarsDataSet(AbstractDataset):
    def __init__(self, filepath: str, load_args: dict = None, save_args: dict = None):
        self._filepath = filepath
        self._load_args = load_args or {}
        self._save_args = save_args or {}

    def _load(self) -> pl.DataFrame:
        return pl.read_parquet(self._filepath, **self._filter_load_args())

    def _save(self, data: pl.DataFrame) -> None:
        data.write_parquet(self._filepath, **self._save_args)

    def _describe(self) -> dict:
        return dict(filepath=self._filepath)

    def _filter_load_args(self) -> dict:
        # Aquí puedes filtrar los argumentos incompatibles
        compatible_args = {"columns"}
        return {k: v for k, v in self._load_args.items() if k in compatible_args}


class CSVDataSet(AbstractDataset):
    def __init__(
        self,
        filepath: str,
        library: str = "pandas",
        load_args: dict = None,
        save_args: dict = None,
    ):
        """
        Clase para leer y escribir archivos CSV usando pandas o polars.

        :param filepath: Ruta del archivo CSV.
        :param library: Biblioteca a usar ("pandas" o "polars").
        :param load_args: Argumentos adicionales para la función de carga.
        :param save_args: Argumentos adicionales para la función de guardado.
        """
        self._filepath = filepath
        self._library = library
        self._load_args = load_args or {}
        self._save_args = save_args or {}

        if self._library not in ["pandas", "polars"]:
            raise ValueError("Library must be either 'pandas' or 'polars'.")

    def _load(self):
        if self._library == "pandas":
            return pd.read_csv(self._filepath, **self._load_args)
        elif self._library == "polars":
            return pl.read_csv(self._filepath, **self._load_args)

    def _save(self, data):
        if self._library == "pandas":
            data.to_csv(self._filepath, **self._save_args)
        elif self._library == "polars":
            data.write_csv(self._filepath, **self._save_args)

    def _describe(self) -> dict:
        return dict(filepath=self._filepath, library=self._library)


class AwsParquetDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
    ):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = filepath
        self._fs = fsspec.filesystem(self._protocol)
        self._load_args = {**self.DEFAULT_LOAD_ARGS, **(load_args or {})}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(save_args or {})}

    def load(self) -> pd.DataFrame:
        # load_path = str(self._get_load_path())
        load_path = f"{self._filepath}"
        df = wr.s3.read_parquet(path=load_path, **self._load_args)
        # Convert all float columns to int64
        float_columns = df.select_dtypes(include=["float64", "Int64"]).columns
        df[float_columns] = df[float_columns].astype(np.int64)

        # Convert all object columns to string
        object_columns = df.select_dtypes(include=["object"]).columns
        df[object_columns] = df[object_columns].astype(np.str_)

        return df

    def save(self, data: pd.DataFrame) -> None:
        wr.s3.to_parquet(df=data, **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
