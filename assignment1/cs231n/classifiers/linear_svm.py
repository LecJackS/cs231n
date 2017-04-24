import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print(dW.shape)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        ## my code
        #count_j_dif_y += 1
        dW[:, y[i]] += -X[i]
        dW[:, j] += X[i]
        ## cuando y[i] le resta X[i] (num_classes - 1) de veces
        ## cuando es otro, le suma, ese ejemplo una sola vez
 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # we scale dW by the number of training examples
  dW /= num_train
  
  # Adding regularization gradient
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  ##pass
  # mycode
  num_train = X.shape[0]
  #- W: A numpy array of shape (D, C) containing weights.
  #- X: A numpy array of shape (N, D) containing a minibatch of data.
  #- y: A numpy array of shape (N,) containing training labels;
  # matrix of scores:
  # - each row is a list of scores for that training example
  # - each column a category
  scores = X.dot(W) # (N, C)
  #print("scores y ", len(tuple(y)))
  #y = tuple(y)
  #correct_class_scores = scores[y] # (N, 1) a vector of correct scores
  #correct_class_scores = np.asarray([row[col] for row,col in zip(scores, y)])
  correct_class_scores = scores[np.arange(num_train), y]
  #print("correct_class_scores", correct_class_scores.shape)
  margins = scores.T - correct_class_scores + 1
  #margins = np.sum((scores, -correct_class_scores), axis=0) + 1
  # to use always the same shape
  margins = margins.T
  # we need to remove the margins of the correct label y
  margins[np.arange(num_train), y] = 0 
  # we need to sum over all positive values of margins
  loss = np.sum(margins[margins>0])
  
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #loss -= 1
  ##
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ##pass
  # mycode
  # we don't need margins < 0
  margins[margins < 0] = 0

  margins[margins > 0] = 1

  #print(margins.shape)
  #margins = margins.T
  # we need to remove the margins of the correct label y
  #margins[np.arange(num_train), y] = 0
  #print(np.sum(margins, axis=1).shape)
  margins[np.arange(num_train), y] = margins[np.arange(num_train), y].T -1*np.sum(margins, axis=1)
  #margins[np.arange(num_train), y].T += -1*np.sum(margins, axis=1)

  dW = np.dot(margins.T, X)
  # we scale dW by the number of training examples
  dW /= num_train
  
  # Adding regularization gradient
  #dW += reg * W
  dW = dW.T
  ##
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
