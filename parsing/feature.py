import numpy as np
import pandas as pd


def _extract_salary(series: pd.Series) -> pd.Series:
    salary_regex = r"[^\d]"
    return (
        series.astype(str)
        .str.replace(salary_regex, "", regex=True)
        .replace("nan", pd.NA)
        .replace("", pd.NA)
        .astype("Int64")
    )


def _extract_age(series: pd.Series) -> pd.Series:
    age_regex = r"(\d+)\s*год"
    return series.str.extract(age_regex)[0].astype("Int64")


def _extract_time(series: pd.Series) -> pd.Series:
    return series.str.contains("полная", case=False, na=False).astype(int)


def _extract_gender(series: pd.Series) -> pd.Series:
    return series.str.contains("Мужчина", na=False).astype(int)


def _extract_town(series: pd.Series) -> pd.Series:
    return series.str.split(",").str[0].astype("category").cat.codes


def _extract_car(series: pd.Series) -> pd.Series:
    return series.str.contains("авто", case=False, na=False).astype(int)


def _extract_education(series: pd.Series) -> pd.Series:
    return series.str.contains("Высшее", case=False, na=False).astype(int)


def _extract_remote(series: pd.Series) -> pd.Series:
    return series.str.contains("удал", case=False, na=False).astype(int)


def _extract_experience(series: pd.Series) -> pd.Series:
    years = series.str.extract(r"(\d+)\s*лет")[0]
    return years.astype("Int64")


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = _extract_salary(df["ЗП"])

    X = pd.DataFrame({
        "is_male": _extract_gender(df["Пол, возраст"]),
        "age": _extract_age(df["Пол, возраст"]),
        "town": _extract_town(df["Город"]),
        "full_time": _extract_time(df["Занятость"]),
        "has_car": _extract_car(df["Авто"]),
        "higher_education": _extract_education(df["Образование и ВУЗ"]),
        "remote_work": _extract_remote(df["График"]),
        "experience": _extract_experience(df["Опыт (двойное нажатие для полной версии)"]),
    })

    mask = y.notna()

    X = X[mask].fillna(0)
    y = y[mask]

    return (
        X.astype("int64").to_numpy(),
        y.astype("int64").to_numpy(),
    )
