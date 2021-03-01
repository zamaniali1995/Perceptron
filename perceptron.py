import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def plot(weights, datapoints, labels):
  """
  Plots weights/bias in a 2-D grid. The specifics of this are something y'all can kind of ignore
  """
  plt.figure(figsize=(10,10))
  plt.grid(True)

  for input, target in zip(datapoints, labels):
    plt.plot(input[1],input[2],'ro' if (target == 1.0) else 'go')

  #ax = plt.axes()
  #ax.arrow(0, 0, weights[1], weights[2], head_width=0.5, head_length=0.7, fc='lightblue', ec='black')
  x_min = np.amin(datapoints[:,1])
  x_max = np.amax(datapoints[:,1])
  get_y = lambda x: (-weights[0]-weights[1]*x)/weights[2]
  
  plt.axline([0, get_y(0)], [1, get_y(1)], color="black")

# Initializing training/testing data
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[2, 2], [8, 8]], cluster_std=1.05, random_state=6)

## Inserting a column of ones (to provide offset to your otherwise-linear predictor)
X=np.insert(X, 0, 1, axis=1)
## Making the outputs 1 and -1
y[y==0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

# The perceptron learning algorith
def predict(x, weights):
  """
  Returns a single prediction given a sample x and the predictor's weights
  TODO
  """
  return np.sign(np.matmul(np.transpose(weights), x))

def train(X, y, lr=0.1):
  """
  Initializes, then updates weights based on the provided training data, then
  finally returns the weights of the resulting model
  TODO
  """
  # Init weights
  weights = np.random.rand(X.shape[1])
  print('weights', weights)
  
  # Train
  for Xi, yi in zip(X, y):
    if np.sign(np.matmul(np.transpose(weights), Xi)) != yi:
      weights  += (lr * yi * Xi)
  return weights 
  

# Performing training, predictions, and analyses
weights = train(X_train, y_train)
predictions = [predict(xi, weights) for xi in X_test]
print("The accuracy is", accuracy_score(y_true=y_test, y_pred = predictions))
print("The weights w=[b, w_x, w_y] are", weights)
plot(weights, X, y)
