from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # softmax P(Y=k|X=x_i) = e^{s_k}/∑e^{s_j} softmax loss = -log(softmax)
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W) # get scores
        max_score = np.max(scores)
        scores -= max_score # 考虑数值计算稳定性 softmax = (e^s_c - max)/∑(e^s_j - max)
        correct_score = scores[y[i]] # score_correct
        P_ic = np.exp(correct_score)/np.sum(np.exp(scores))
        loss += -np.log(P_ic)
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += (P_ic - 1) * X[i].T
            else:
                P_ij = np.exp(scores[j])/np.sum(np.exp(scores))
                dW[:, j] += P_ij * X[i].T
                
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X@W # 500,10
#     print(scores.shape)
    max_scores = np.max(scores, axis=1).reshape(-1,1) # 500, numeric instablity
#     print(max_scores.shape)
    scores -= max_scores # numeric instablity
#     print(scores.shape)
    correct_scores = scores[np.arange(scores.shape[0]), y] # 500,
    P_ic = np.exp(correct_scores)/np.sum(np.exp(scores), axis=1)
#     print(P)
    loss += np.sum(-np.log(P_ic))/scores.shape[0] # L = ∑L_i/N
    loss += reg * np.sum(W * W) # regularization
    # 向量化梯度：用scores构建一个P [500, 10]，首先取exp(scores)得到每一个位置的exp,然后对每个位置除以这一行的exp和
    # 上面的操作会得到500，10的矩阵，每个位置都是softmax之后的结果
    # !重点：对于[i,y[i]]位置，根据P_ic - 1, 要减1 
    P = np.exp(scores) # 正确分类的梯度, 位于梯度矩阵所有c的行
    P /= np.sum(np.exp(scores),axis=1).reshape(-1, 1)
    P[np.arange(scores.shape[0]), y] -= 1 # 将 i, y[i] -= 1
    
    # 得到这个矩阵之后，与X.T相乘即可得到dL/dW P(500,10) X(500,3073) X.T (3073, 500) W(3073, 10)
    dW += X.T@P
    dW /= scores.shape[0] # *1/N
    dW += 2*reg*W # 正则化梯度
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
