# cs231n: assignments

<a href='http://www.scu.edu.cn'><img src="https://img.shields.io/badge/University-SCU-blue?style=for-the-badge&logo=Julia" alt='www.scu.edu.cn'/></a>   <img src="https://img.shields.io/badge/Status-In%20Progress-important?style=for-the-badge&logo=Apple%20Podcasts">

## To-Do Lists

- [x] Assignment1
  - [x] KNN
    - [x] Answer all inline questions. (If there are some problems, please submit issue and let me know!)
  - [x] SVM
    - [x] Answer all inline questions. (Inline question 1)
  - [x] Softmax Classifier
    - [x] Answer all inline questions.
  - [x] Two-layer Neural Network
    - [x] Answer all inline questions.
    - [x] hyperparameters tuning: `54.7%` val acc, `54.3%` test acc.
  - [x] High Level Representation of Image Features
    - [x] Answer all inline questions.
- [ ] Assignment2
- [ ] Assignment3



## What do I learn from assignments

- kNN workflow
- Numpy
- Vectorization implementation.
- SVM workflow (s=f(x;w) => multi-svm loss => gradients descent)
- SVM loss and gradients computation(I think it Hard)
- [bias trick](https://hetpinvn.wordpress.com/2016/10/26/bias-trick/): y = Wx + b => y = Wx', x' = x.stack(np.ones(x.shape[0], 1)), as shown in the picture below.

![Bias trick](https://tva1.sinaimg.cn/large/008eGmZEly1gn3gm4cp3bj318w0heaii.jpg)

* Handwritten 2 layer network.
* Hyperparameters tuning process.
* Weight visulization.