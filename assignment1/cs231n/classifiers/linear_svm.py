from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue # loss += 0, dW += 0
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
#                 print(X[i].shape)
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i] # why? why couldn't be =X[i].

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # L = 1/N * ∑Li, dL/dW = dL/dLi * dLi / dW = 1/N * dLi/dW

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # "Implementation is above."

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # X: (500, 3073)
    # W: (3073, 10)
    # y: (500,)
    scores = np.dot(X, W) # 500,10
    correct_scores = scores[np.arange(scores.shape[0]), y] # !trick, arr[(a,b),(c,d)] produce value of (a,c) (b,d) ..
    correct_scores = np.reshape(correct_scores, (scores.shape[0], -1))
#     print(scores.shape)
#     print(correct_scores.shape)
    margins = scores - correct_scores + 1.0
    L = np.maximum(0, margins)
#     print(L.shape)
    L[np.arange(X.shape[0]), y] = 0.0 # 将j==y_i这些位置赋值为0
    loss += np.sum(L) / X.shape[0]
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    L[L > 0] = 1.0 # 设置为1，这样的话这一行的1加起来就可以成为n,即 -nXi的n
#     print(L) # 对于500个样本计算
    row_sum = np.sum(L, axis=1) # 对列求和得到每一样本的n值, 进而方便求解 -nXi
#     print(row_sum)
    L[np.arange(L.shape[0]), y] = -row_sum # 得到示性函数的数值，对于j != y_i 示性函数为1，对于 j==y_i 示性函数为 该行的其它非0值个数的和
    # dW 500,10  X_T (3073, 500) L (500,10) W (3073, 10)
    # 首先梯度公式：dL/dW_j = 1(scores - correct_scores + 1.0)xi dL/dW_yi = -∑(scores - correct_scores + 1.0)xi 
    # 这个可以看成 一个示性矩阵 * x_i 示性矩阵在j != y_i,值为1， j==y_i 值为-n
    dW = X.T@L / L.shape[0] + 2*reg*W # (1)X.T @ L (矩阵乘法)=(3073, 10) (2)除以L.shape[0](根据L=1/N∑Li) (3) 增加正则化项
#     dW += 2*reg*W # (3073,10)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
