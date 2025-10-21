from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train_baseline(X_train, y_train):
    """
    Train a baseline linear regression model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.

    Returns
    -------
    model : LinearRegression
        Fitted regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using mean absolute error.

    Parameters
    ----------
    model : estimator
        Fitted regression model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        Test target vector.

    Returns
    -------
    mae : float
        Mean absolute error of predictions.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae
