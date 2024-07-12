# Execution Instructions 
Command-line argument for main.py: path to folder containing the split from A2 (this is the one that includes the labels and is useful for training and evaluation): /DATA1/shristov/assignments/a2/data

# Classifier Performance for each Activation Function

| Acitvation Function | Test Accuracy |
|------------|---------------|
| ReLU       | 78.198%       |
| Sigmoid    | 77.289%       |
| Tanh       | 77.546%       |

# Effect of Activation Function
It is worth noting that each of the functions performed very similarly. However, ReLU was clearly superior than Sigmoid and Tanh, which achieved almost identical performance. I believe this may be because Sigmoid and Tanh are both smooth functions, whereas ReLU is piecewise linear. Perhaps, when coupled with softmax, introducing non-linearity in a piecewise fashion results in a more robust network (indeed, when trying to classify an ambiguous sentence such as 'i dont love it, but i like it', the ReLU model was the only one that correctly identified it as positive). 

# Effect of L2-norm Regularization 
L2-norm regularization adds a penalty as the model complexity increases, which forces weights to be small instead of being zero. This prevents overfitting (increases bias but decreases variance). Indeed, when I trained a model without L2-norm regularization, it performed worse on the validation set, indicating that it overfit to the training set.

# Effect of Dropout Rate
Once the model learns the optimial weights for a given iteration, dropout forces a subset of them to randomly be set to 0 (unlike regularization, which incentivizes the model to learn weights closer to 0). This forces the model to diversify the network such that it is not overly dependent on any single neuron, leading to more robust predictions. However, dropout is a hyperparameter; setting it too low will result in potential overfitting, whereas setting it too high will result in the network not properly learning the relationships.
