import re
from abc import ABC, abstractmethod

import pandas as pd


class DataFrameHandler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next_handler(self, handler: "DataFrameHandler") -> "DataFrameHandler":
        self._next_handler = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._process(df)
        if self._next_handler:
            return self._next_handler.handle(df)
        return df

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class CleanDataHandler(DataFrameHandler):
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        for col in res.columns:
            if res[col].dtype == 'object':
                res[col] = res[col].apply(
                    lambda x: re.sub(r"[\u00a0\u200b\ufeff]", " ", str(x)).strip()
                    if pd.notna(x) else x
                )
        return res


class DropUselessDataHandler(DataFrameHandler):
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_na().reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        return df


class DropEmptySalaryHandler(DataFrameHandler):
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["ЗП"].notna()]
