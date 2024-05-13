# Basic-MLP

Manually built a working Neural Network using only NumPy, includes basic core functionality for training and inference

| Script                       | Contains                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `run.py`                     | Main script, preprocesses and trains a neural network using `iris.dat` for 1000 epochs                                         |
| `neural_network.py`          | `LinearLayer`: Performs affine transformation of input.                                                                        |
|                              | `MultiLayerNetwork`: A network consisting of stacked linear layers and activation functions.                                   |
| `train.py`                   | `Preprocessor`: Object used to apply "preprocessing" operation to datasets. The object can also be used to revert the changes. |
|                              | `Trainer`: Object that manages the training of a neural network.                                                               |
| `activation_functions.py`    | `SigmoidLayer`: Applies sigmoid function elementwise.                                                                          |
|                              | `ReluLayer`: Applies Relu function elementwise.                                                                                |
| `loss_functions.py`          | `MSELossLayer`: Computes mean-squared error between y_pred and y_target.                                                       |
|                              | `CrossEntropyLossLayer`: Computes the softmax followed by the negative log-likelihood loss.                                    |
| `initilisation_functions.py` | `xavier_init`: Xavier initialization of network weights.                                                                       |

```bash
# Install all required Python packages as specified in the requirements.txt file
pip install -r requirements.txt

# Run the Python script named run.py
python run.py
```
