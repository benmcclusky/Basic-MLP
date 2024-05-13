import numpy as np

from loss_functions import MSELossLayer, CrossEntropyLossLayer

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
        
        # Initialise mean and std for data
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """

        return (data - self.mean) / self.std


    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """

        return (data * self.std) + self.mean



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
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        # Initialize the loss layer based on loss_fun
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            raise ValueError(
                f"Unsupported loss function: {loss_fun}\nAvailable Loss Functions: 'mse', 'cross_entropy'"
            )


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
        # Assert that input and target datasets have the same number of data points

        assert len(input_dataset) == len(
            target_dataset
        ), "Input and target datasets must have the same length"

        # Generate random permutation indices
        perm = np.random.permutation(len(input_dataset))

        # Apply permutation to input and target datasets
        shuffled_inputs = input_dataset[perm]
        shuffled_targets = target_dataset[perm]

        return shuffled_inputs, shuffled_targets


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

        for epoch in range(self.nb_epoch):
            epoch_loss = 0

            # Shuffle dataset if shuffle_flag = True
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(
                    input_dataset, target_dataset
                )

            # Iterate over batches
            for i in range(0, len(input_dataset), self.batch_size):
                batch_input = input_dataset[i : i + self.batch_size]
                batch_target = target_dataset[i : i + self.batch_size]

                # Forward pass and compute loss computation
                predictions = self.network.forward(batch_input)
                loss = self._loss_layer.forward(predictions, batch_target)
                epoch_loss += loss

                # Backward pass and update parameters
                self.network.backward(self._loss_layer.backward())
                self.network.update_params(self.learning_rate)

            # Average loss for the epoch
            epoch_loss /= len(input_dataset) / self.batch_size

            # Report training progress
            print(f"Epoch {epoch+1}/{self.nb_epoch}, Loss: {epoch_loss}")
            

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

        # Forward pass
        predictions = self.network.forward(input_dataset)

        # Compute loss
        loss = self._loss_layer.forward(predictions, target_dataset)

        return loss


