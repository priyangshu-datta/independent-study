import enum
from pathlib import Path
import pickle
import typing

import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn import (
    base,
    ensemble,
    metrics,
    model_selection,
    preprocessing,
    svm,
    feature_selection,
    neighbors,
)
import strenum
import tqdm
import xgboost

OVERWRITE = {
    "best_params": False,
    "train_model": False,
    "img_generate": True,
    "shap_calculation": False,
}
# overwrite, ignoring and overwriting all cache.


class Models(strenum.StrEnum):
    SVM = enum.auto()
    XGB = enum.auto()
    RF = enum.auto()
    GB = enum.auto()


class Parameters(strenum.StrEnum):
    T = "Temperature"
    DO = "Dissolved Oxygen"
    PH = "pH"
    COND = "Conductivity"
    BOD = "Biochemical Oxygen Demand"
    N = "Nitrate Nitrite Conc"
    FC = "Fecal Coliform"
    TC = "Total Coliform"


long_keys = [lk.value for lk in Parameters]
short_keys = [lk.name for lk in Parameters]

short2long = {k: v for k, v in zip(short_keys, long_keys)}
long2short = {v: k for k, v in short2long.items()}

model_params: dict[
    Models,
    dict[
        typing.Literal["fixed", "tuning", "other"],
        dict[str, str | int | float | bool | list],
    ],
] = {
    Models.GB: {
        "fixed": {
            "loss": "squared_error",
            "random_state": 0,
            "max_depth": 10,
            "subsample": 0.5,
        },
        "tuning": {
            "n_estimators": [1000, 3000, 4500],
            "learning_rate": [0.01, 0.001, 0.1],
        },
        "other": {},
    },
    Models.RF: {
        "fixed": {
            "max_depth": 10,
            "random_state": 0,
            "max_samples": 0.5,
            "criterion": "squared_error",
        },
        "tuning": {
            "n_estimators": [1000, 3000, 5000],
        },
        "other": {},
    },
    Models.XGB: {
        "fixed": {
            "random_state": 0,
            "subsample": 0.5,
        },
        "tuning": {
            "n_estimators": [5000, 7500, 10000, 15000, 20000],
            "learning_rate": [0.01, 0.1, 0.001],
        },
        "other": {
            "early_stopping_rounds": 50,
            "eval_metric": metrics.mean_squared_error,
        },
    },
    Models.SVM: {"fixed": {}, "tuning": {}, "other": {}},
}

estimators: dict[
    Models,
    type[ensemble.GradientBoostingRegressor]
    | type[ensemble.RandomForestRegressor]
    | type[svm.SVR]
    | type[xgboost.XGBRegressor],
] = {
    Models.GB: ensemble.GradientBoostingRegressor,
    Models.RF: ensemble.RandomForestRegressor,
    Models.SVM: svm.SVR,
    Models.XGB: xgboost.XGBRegressor,
}

root_path = Path("./")
all_paths = {
    "model": root_path.joinpath("models"),
    "image": root_path.joinpath("images"),
    "best_param": root_path.joinpath("best_params"),
    "shap_value": root_path.joinpath("shap_values"),
    "data": root_path.joinpath("."),
}
for path in all_paths.values():
    path.mkdir(exist_ok=True, parents=True)

T = typing.TypeVar("T")


def read_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as pkl:
        return pickle.load(pkl)


def write_pkl(python_obj: any, pkl_path: Path):
    assert pkl_path.suffix == ".pkl", "Path does not point to a .pkl file."
    pkl_path.parent.mkdir(exist_ok=True, parents=True)

    with open(pkl_path, "wb") as pkl:
        return pickle.dump(python_obj, pkl, protocol=pickle.HIGHEST_PROTOCOL)


def file_exists(
    file_path_s: Path | list[Path] | typing.Iterator[Path],
) -> dict[typing.Literal["exists", "not_exists"], typing.List[Path]]:
    exists = []
    not_exists = []
    if isinstance(file_path_s, Path):
        return (
            {"exists": [file_path_s], "not_exists": []}
            if file_path_s.exists()
            else {"exists": [], "not_exists": [file_path_s]}
        )
        # return ([file_path_s], []) if file_path_s.exists() else ([], [file_path_s])
    for f_path in file_path_s:
        if f_path.exists():
            exists.append(f_path)
        else:
            not_exists.append(f_path)
    return {"exists": exists, "not_exists": not_exists}


def literal_to_list(abc: T) -> typing.List[T]:
    if abc is None:
        abc = []
    if not (isinstance(abc, enum.EnumType) or isinstance(abc, list)):
        return [abc]
    return abc


def remove_outliers(data: pd.Series) -> list[int]:
    mu = data.mean()
    sigma = data.std()
    outliers_index = [
        i for i, x in enumerate(data) if x > (mu + 4 * sigma) or x < (mu - 4 * sigma)
    ]

    return outliers_index


def load_models(
    model_name: Models, targets: Parameters | list[Parameters], path_prefix: Path
):
    models: dict[Parameters, base.BaseEstimator] = {}
    targets = literal_to_list(targets)
    for target in targets:
        models[target] = read_pkl(
            pkl_path=all_paths["model"]
            .joinpath(path_prefix, model_name, target)
            .with_suffix(".pkl")
        )

    return models


data_path = all_paths["data"].joinpath("WQ 2012_2021.csv")

train_df = (
    pd.read_csv(data_path)
    .query("YEAR != 2021")
    .dropna()
    .iloc[:, 7::3]
    .reset_index(drop=True)
)
train_df.columns = long_keys

outliers = []
for col in train_df.columns:
    outliers.extend(remove_outliers(train_df[col]))
train_df = train_df.drop(outliers).reset_index(drop=True)

train_df[long_keys] = preprocessing.MinMaxScaler().fit_transform(train_df)


def get_train_data(
    df: pd.DataFrame,
    target: Parameters,
    validation_split: None | float | int = None,
    exclude_features: None | Parameters | list[Parameters] = None,
):
    exclude_features = literal_to_list(exclude_features)

    X = df.drop(columns=[target, *exclude_features])
    y = df[target]

    if validation_split:
        assert 0 < validation_split < 1 or (
            isinstance(validation_split, int) and 0 < validation_split < 100
        ), "validation split must be between 0..1 or 0..100."
        return model_selection.train_test_split(
            X, y, test_size=validation_split, random_state=0
        )
    return X, y


def shap_func(config, train_df, only_values=False, save_plot=False, show_plot=True):
    model_name = config["model_name"]
    path_prefix = config["path_prefix"]
    exclude_features = config["exclude_features"]
    target = config["target"]
    assert (
        all_paths["model"].joinpath(path_prefix, model_name).exists()
    ), "model not found. First train models."

    shap_PATH = (
        all_paths["shap_value"]
        .joinpath(path_prefix, model_name, target)
        .with_suffix(".pkl")
    )

    shap_EXIST, _ = file_exists(shap_PATH).values()
    if shap_EXIST.__len__() > 0:
        if shap_EXIST[0].is_dir():
            shap_EXIST[0].rmdir()
            shap_EXIST = []

    shap_value = None
    if not OVERWRITE["shap_calculation"] and shap_EXIST.__len__() > 0:
        shap_value = read_pkl(shap_PATH)

    shap_PATH.parent.mkdir(exist_ok=True, parents=True)

    X_train, *_ = get_train_data(
        df=train_df,
        target=target,
        validation_split=0.2 if model_name == Models.XGB else None,
        exclude_features=exclude_features,
    )
    model = load_models(model_name, target, path_prefix)[target]

    explainer = shap.Explainer(model, X_train)
    if shap_value is None:
        shap_value = explainer(X_train)
        write_pkl(
            shap_value,
            all_paths["shap_value"]
            .joinpath(path_prefix, model_name, target)
            .with_suffix(".pkl"),
        )

    def save_plot(type: typing.Literal["bar", "summary", "beeswarm"]):
        image_path = (
            all_paths["image"]
            .joinpath(type, path_prefix, model_name, target)
            .with_suffix(".jpg")
        )
        image_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(image_path, bbox_inches="tight")

    if not only_values and OVERWRITE["img_generate"]:
        plt.title(target)
        shap.summary_plot(shap_value, X_train, alpha=0.4, show=not save_plot)
        if save_plot:
            save_plot("summary")
        if show_plot:
            plt.show()

        plt.title(target)
        shap.summary_plot(shap_value, X_train, plot_type="violin", show=not save_plot)
        if save_plot:
            save_plot("violin")
        if show_plot:
            plt.show()

        plt.title(target)
        shap.plots.bar(shap_value, show=not save_plot)
        if save_plot:
            save_plot("bar")
        if show_plot:
            plt.show()

        force_plot = shap.force_plot(
            explainer.expected_value, shap_value.values, X_train
        )
        html_path = (
            all_paths["image"]
            .joinpath("force", path_prefix, model_name, target)
            .with_suffix(".html")
        )
        html_path.parent.mkdir(exist_ok=True, parents=True)
        shap.save_html(html_path.__str__(), force_plot)

    return shap_value


exclude_features = [Parameters.FC]
path_prefix = f"{'exclude_features/' if exclude_features.__len__() > 0 else 'optimized/'}{', '.join([long2short[omits] for omits in exclude_features])}"

for target_name in Parameters:
    if target_name is not Parameters.FC:
        shap_func(
            config={
                "model_name": Models.RF,
                "target": target_name,
                "exclude_features": exclude_features,
                "path_prefix": path_prefix,
            },
            train_df=train_df,
            show_plot=False,
            save_plot=True,
        )
