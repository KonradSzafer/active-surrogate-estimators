import openml
import numpy as np
import pandas as pd


def transform_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms all not floating columns to numeric.
    """
    continuous_cols = df.select_dtypes(include=[np.floating]).columns
    for col in df.columns:
        if col not in continuous_cols:
            df[col] = pd.factorize(df[col])[0]
    return df


def replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        median_value = df[col].dropna().median()
        df[col].fillna(median_value, inplace=True)
    return df


def replace_labels_with_integers(Y: pd.Series) -> pd.Series:
    unique_labels = Y.unique()
    assert len(unique_labels) == 2
    if unique_labels[0] != 0 and unique_labels[1] != 1:
        print(f'Replacing labels {unique_labels} with 0 and 1')
        label_to_binary = {unique_labels[0]: 0, unique_labels[1]: 1}
        Y = Y.replace(label_to_binary)
    return Y.astype('int')


def load_dataset(dataset_id: int):
    dataset = openml.datasets.get_dataset(
        dataset_id=dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    df, *_ = dataset.get_data()
    df = transform_columns(df)
    df = replace_missing(df)
    target_col = dataset.default_target_attribute
    X = df.drop([target_col], axis=1, inplace=False)
    Y = df[target_col]
    Y = replace_labels_with_integers(Y)
    assert set(Y.unique()) == {0, 1}
    return X.to_numpy(), Y.to_numpy()
