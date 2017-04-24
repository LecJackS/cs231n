import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # adding softmax regularization to prevent exploding
    scores += -np.max(scores)
    # softmax layer
    scores = np.exp(scores) / np.sum(np.exp(scores))

    correct_class_score = scores[y[i]]
    loss += -np.log(correct_class_score)

    for j in xrange(num_classes):
        dW[:, j] += X[i] * (scores[j]-(j==y[i]))
 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  
  # Adding regularization gradient
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax(X):
    # adding softmax regularization to prevent exploding
    X += -np.max(X)
    # get unnormalized probabilities
    exp_X = np.exp(X)
    # normalize them for each example and return
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W) #(NxD, DxC) > NxC

  probs =  softmax(scores)
  #array of correct examples probabilities
  num_examples = X.shape[0]
  correct_log_probs = -np.log(probs[range(num_examples), y])

  #compute loss: average cross-entropy loss and regulareization
  data_loss = np.sum(correct_log_probs) / num_examples
  reg_loss = 0.5 * reg * np.sum(W*W)
  loss = data_loss + reg_loss

  #computing analytic gradient
  dscores = probs
  dscores[range(num_examples), y] += -1
  dscores /= num_examples

  dW = np.dot(X.T, dscores)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

