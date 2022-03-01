import random
# from tkinter.tix import Y_REGION
# from pytest import skip
import torch
import inspect
import pickle
import numpy as np
import pandas as pd
import part1_nn_lib as nn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from collections import defaultdict


class Regressor():

    def __init__(self, x, nb_epoch=500,
                 neurons=[150, 150, 150, 1],
                 activations=["relu", "relu", "relu", "linear"], batch_size=8000, dropout_rate=0.00, learning_rate=0.1, loss_fun="mse"):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """
        # NOT VERY SMOOTH, ACCOUNT FOR ONE_HOTS
        self.input_size = x.shape[1] + 4
        self.output_size = 1
        self.x = x
        self.neurons = neurons
        self.activations = activations
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss_fun = "mse"
        self.net = nn.MultiLayerNetwork(self.input_size, neurons, activations)
        return

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """
        # SET UP ONE_HOT MAKER
        if training:
            self._lb = LabelBinarizer()
            self._lb.fit(['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])

        # FIX TEXT ENTRIES AND
        X = x.copy()  # Copy the dataframe
        # Not sure if this is correct default
        X.fillna(random.uniform(0, 1), inplace=True)
        one_hots = self._lb.transform(
            X["ocean_proximity"])  # Form one-hot vectors
        X = X.drop(labels="ocean_proximity", axis=1)

        # NO-LOOP NORMALISATION METHOD
        X_norm = (X-X.min(skipna=True))/(X.max(skipna=True)-X.min(skipna=True))
        X_numpy = X_norm.copy().to_numpy().astype(float)
        X_numpy = np.concatenate((X_numpy, one_hots), axis=1)
        

        Y_numpy = None
        if isinstance(y, pd.DataFrame):
            Y_numpy = y.copy().to_numpy().astype(float)
            if training:
                self.min_y = np.amin(Y_numpy)
                self.max_y = np.amax(Y_numpy)
            Y_numpy = (Y_numpy-self.min_y)/(self.max_y -
                                            self.min_y)  # DO WE NORMALISE Y?

        return X_numpy, Y_numpy

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        trainer = nn.Trainer(
            network=self.net,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            learning_rate=self.learning_rate,
            loss_fun=self.loss_fun,
            shuffle_flag=False
        )
        trainer.train(X, Y)
        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """
        X, _ = self._preprocessor(x, training=False)  # Do not forget
        return self.net(X)

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        _, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        predictions = self.predict(x)
        print('preds:', predictions, '\ntruths:', Y)
        return mean_squared_error(Y, predictions)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train, x_test, y_test):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # x = [x_train]
    # neurons = [[5, 20, 20], [5, 10, 20], [10, 50, 50]]
    # learning_rate = [0.001, 0.01, 0.1]
    # nb_epoch = [50, 500, 2500]
    # batch_size = [5, 20, 50]
    # dropout_rate = [0.0, 0.3, 0.4, 0.5]
    
    x = [x_train]
    neurons = [[5, 20, 20, 1]]
    learning_rate = [0.01, 0.1]
    nb_epoch = [5, 25, 100]
    batch_size = [5]
    dropout_rate = [0.0]

    regressor = Regressor(x_train)

    grid = dict(x=x, neurons=neurons,
                nb_epoch=nb_epoch, batch_size=batch_size, dropout_rate=dropout_rate, learning_rate=learning_rate)

    grid_search = GridSearchCV(estimator=regressor, param_grid=grid,
                               scoring=["neg_mean_squared_error"], refit="neg_mean_squared_error", cv=2, verbose=4, error_score="raise")

    result = grid_search.fit(x_train, y_train)

    return result.best_params_
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    split_idx = int(0.8 * len(x))

    x_train = x.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    x_val = x.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    # regressor = Regressor(x_train)
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)
    # regressor = load_regressor()

    print(RegressorHyperParameterSearch(x_train, y_train, x_val, y_val))

    # Error
    # error = regressor.score(x_val, y_val)
    # print("\nRegressor error: {}\n".format(error))
    # rhs_zero_point_zero_one = abs(-56016414055.705 -55899521116.989)
    # lhs_zero_point_one = abs(-56016429457.618 -55899518780.060)

    # print(rhs_zero_point_zero_one < lhs_zero_point_one)

if __name__ == "__main__":
    example_main()
