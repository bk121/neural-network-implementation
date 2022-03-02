# from matplotlib.pyplot import step
import numpy as np
import pickle
from pyrsistent import b
from part2_house_value_regression import *
import sys


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """

    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))

    return np.random.uniform(low=low, high=high, size=size)

class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs
        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self._cache_current = 1/(1+np.exp(-x))
        return self._cache_current

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:self._cache_current
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        return grad_z * self._cache_current*(1-self._cache_current)


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # Standard ReLU implementation
        self._cache_current = x.clip(min=0)

        # Leaky ReLU implementation -- 0.01 is an arbitrary "very"
        # self._cache_current = np.where(x > 0, x, x * 0.00)

        return self._cache_current

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        # Standard RELU implementation
        result = self._cache_current[self._cache_current > 0] = 1

        return grad_z * result


class LinearActivationLayer(Layer):
    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        self._cache_current = x
        return self._cache_current

    def backward(self, grad_z):
        return grad_z


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = xavier_init((n_in, n_out), gain=1)
        # self._W = he_init((n_in, n_out))
        self._b = np.zeros((1, n_out))

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        self._cache_current = x
        return ((x @ self._W) + self._b)

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments: 
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        self._grad_W_current = self._cache_current.T@grad_z
        self._grad_b_current = np.ones((len(grad_z), 1)).T@grad_z

        return grad_z@self._W.T

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W = np.clip(self._W - learning_rate*self._grad_W_current, -1000, 1000)
        self._b = np.clip(self._b - learning_rate*self._grad_b_current, -1000, 1000)


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations, dropout_rate=0.00):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding
                the batch dimension).
                        - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self._layers = []
        self._dropout_rate = dropout_rate
        n_ins = [input_dim] + neurons[:-1]
        for i, activation in enumerate(activations):
            self._layers.append(LinearLayer(n_ins[i], self.neurons[i]))
            if activation.lower() == "relu":
                self._layers.append(ReluLayer())
            elif activation.lower() == "sigmoid":
                self._layers.append(SigmoidLayer())
            elif activation.lower() == "linear":
                self._layers.append(LinearActivationLayer())

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output ar0.54643265yer)
        """
        a = x
        for i, layer in enumerate(self._layers):
            # Implementing dropout
            # if i % 2 == 1:
            #     binary_values = np.random.rand(a.shape[0], a.shape[1]) < (1 - self._dropout_rate)
            # #     a = layer.forward(a) * binary_values
            # #     a /= (1 - self._dropout_rate)
            # else:
            a = layer.forward(a)
        return a

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                # _neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        d = grad_z
        for layer in self._layers[::-1]:
            d = layer.backward(d)
        return d

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """

        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = MSELossLayer() if self.loss_fun == 'mse' else CrossEntropyLossLayer()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns:
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        shuffled_indices = np.random.default_rng(
        ).permutation(np.shape(input_dataset)[0])
        shuffled_inputs = input_dataset[shuffled_indices]
        shuffled_targets = target_dataset[shuffled_indices]

        return shuffled_inputs, shuffled_targets

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for epoch in range(self.nb_epoch):
            input_data, target_data = self.shuffle(
                input_dataset, target_dataset) if self.shuffle_flag else (input_dataset, target_dataset)
            number_of_splits = max(np.shape(input_data)[0] / self.batch_size, 1)
            split_input_dataset = np.array_split(input_data, number_of_splits)
            split_target_dataset = np.array_split(
                target_data, number_of_splits)

            for i in range(int(number_of_splits)):
                predictions = self.network.forward(split_input_dataset[i])
                error = self._loss_layer.forward(
                    predictions, split_target_dataset[i])

                grad_z = self._loss_layer.backward()
                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)

            if(epoch % 10 == 0):
                print("Epoch: ", epoch, "Error: ", error)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).

        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        predictions = self.network.forward(input_dataset)
        return self._loss_layer.forward(predictions, target_dataset)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Scale smallest value to a and largest value to b: for example [a,b] = [0,1]
        self._lower_range = 0
        self._upper_range = 1
        self._X_min = np.amin(data, axis=0)
        self._X_max = np.amax(data, axis=0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        normalised_data = self._lower_range + ((data - self._X_min) * (
            self._upper_range - self._lower_range)) / (self._X_max - self._X_min)

        return normalised_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        reverted_data = (data * (self._X_max - self._X_min) -
                         self._lower_range) / (self._upper_range - self._lower_range) + self._X_min

        return reverted_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4

    neurons = [16, 3]
    # activations = ["sigmoid", "sigmoid"]
    activations = ["relu", "relu"]

    # neurons = [16, 30, 15, 3]
    # activations = ["relu", "relu", "relu", "relu"]
    # activations = ["sigmoid", "sigmoid", "sigmoid", "sigmoid"]

    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    # print(x_train)
    # print(x_train_pre)
    # print(prep_input.revert(x_train_pre) - x_train)

    # predictions = net.forward(x_train_pre)
    # cll = CrossEntropyLossLayer()
    # cll.forward(predictions, y_train)
    # grad_z = cll.backward()
    # error = net.backward(grad_z)
    # net.update_params(0.1)
    # print("predictions", predictions)
    # print("gradz", grad_z)
    # print("error", error)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=10000,
        learning_rate=0.01,
        loss_fun="bce",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
