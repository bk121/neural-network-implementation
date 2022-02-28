import random
from tkinter.tix import Y_REGION
from pytest import skip
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import part1_nn_lib as nn
from sklearn.metrics import mean_squared_error


class Regressor():

    def __init__(self, x, nb_epoch=30000,
                 neurons=[150, 150, 150, 1],
                 activations=["relu", "relu", "relu", "relu"]):
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
        self.nb_epoch = nb_epoch
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
            self._lb.fit(x["ocean_proximity"])

        # FIX TEXT ENTRIES AND
        X = x.copy()  # Copy the dataframe
        # Not sure if this is correct default
        X.fillna(random.uniform(0, 1), inplace=True)
        one_hots = self._lb.transform(
            X["ocean_proximity"])  # Form one-hot vectors
        X = X.drop(labels="ocean_proximity", axis=1)

        # # STORE PARAMS FOR NORMALISATION
        # if training:
        #     for column in X:
        #         self.min_x = X.min(skipna=True)
        #         self.max_x = X.max(skipna=True)        

        # # NORMALISATION - MAYBE WE CAN USE PART 1 HERE - THIS MUST BE SPED UP
        # for i in X.index:
        #     for column, mi, ma in zip(X, self.min_x, self.max_x):
        #         # Min/max normalisation
        #         X.at[i, column] = (X.at[i, column]-mi) / (ma-mi)

        # FASTER NORMALISATION METHOD
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

    def fit(self, x, y,
            batch_size=8000,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=False):
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
            batch_size=batch_size,
            nb_epoch=self.nb_epoch,
            learning_rate=learning_rate,
            loss_fun=loss_fun,
            shuffle_flag=shuffle_flag
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


def RegressorHyperParameterSearch():
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

    return  # Return the chosen hyper parameters

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
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_val, y_val)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
