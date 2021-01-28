# cs231n: assignments

<a href='http://www.scu.edu.cn'><img src="https://img.shields.io/badge/University-SCU-blue?style=for-the-badge&logo=Julia" alt='www.scu.edu.cn'/></a>

## To-Do Lists

- [ ] Assignment1
  - [x] KNN
    - [x] Answer all inline questions. (If there are some problems, please submit issue and let me know!)
  - [x] SVM
    - [ ] Answer all inline questions.
  - [x] Softmax Classifier
    - [x] Answer all inline questions.
  - [ ] Two-layer Neural Network
  - [ ] High Level Representation of Image Features
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