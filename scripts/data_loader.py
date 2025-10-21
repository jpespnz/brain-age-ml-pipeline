import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path):
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. The CSV must have a target column named 'brain_age'.

    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector containing brain age.
    """
    df = pd.read_csv(csv_path)
    if 'brain_age' not in df.columns:
        raise ValueError("CSV must contain a 'brain_age' column as the target.")
    X = df.drop(columns=['brain_age']).values
    y = df['brain_age'].values
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train and test sets.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    test_size : float, optional
        Fraction of samples to use as test set (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(X_train, X_test):
    """
    Scale features using standard scaling.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    X_test : array-like
        Test feature matrix.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler : tuple
        The scaled training and test matrices and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
