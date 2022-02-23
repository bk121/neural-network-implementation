import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import part1_nn_lib as nn


class Regressor():

    def __init__(self, x, nb_epoch=1000,
                 neurons=[16, 1],
                 activations=["relu", "relu"]):
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
        #X, _ = self._preprocessor(x, training=True)
        self.input_size = x.shape[1]
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

        if training:
            self._lb = LabelBinarizer()
            self._lb.fit(x["ocean_proximity"])
            for column in x:
                self.min_x = x.min(skipna=True)
                self.max_x = x.max(skipna=True)
                self.mean_x = x.mean(skipna=True, numeric_only=True)

        X = x.copy()  # Copy the dataframe
        X.fillna(self.mean_x)  # Not sure if this is correct default

        one_hots = self._lb.transform(
            X["ocean_proximity"])  # Form one-hot vectors
        # Make it so the column can take lists
        X.astype({"ocean_proximity": "object"})
        for i, one_hot in zip(X.index, one_hots):
            # one_hot  # needs to take lists in NN model
            X.at[i, "ocean_proximity"] = one_hot # Replace words with vectors FIX

            for column, mi, ma in zip(X, self.min_x, self.max_x):
                if column != "ocean_proximity":  # Don't normalise word one
                    X.at[i, column] = (X.at[i, column]-mi) / \
                        (ma-mi)  # Min/max normalisation

        return X, (y if isinstance(y, pd.DataFrame) else None)

    def fit(self, x, y,
            batch_size=8,
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
        X_numpy = X.copy().to_numpy()
        Y_numpy = Y.copy().to_numpy()
        trainer.train(X_numpy, Y_numpy)
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
        X_numpy = X.copy().to_numpy().astype(float)
        return self.net(X_numpy).argmax(axis=1).squeeze()

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
        preds = self.predict(x)
        targets = y.to_numpy().astype(float).argmax(axis=1).squeeze()
        accuracy = (preds == targets).mean()
        print("Validation accuracy: {}".format(accuracy))
        return 0


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
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_val, y_val)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
