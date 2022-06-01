r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
A. The shape of this tensor is **(64, 512, 64, 1024)**
B. **The Jacobian is sparse**. Every cell i,j in Y is affected only by the i row in X.
Thus, only cells who's first and third entries are the same (Their index has the form of (i,j,i,k)) could be non-zero.
C. **No.** we can calculate the vector-jacobian product directly:
$\delta\mat{X} = \delta\mat{Y} * ~\mat{W}$.

A2. **$N\times D_{\mathrm{out}}\times D_{\mathrm{out}}\times D_{\mathrm{in}}$.**
B2. **The Jacobian is sparse.** Every cell i,j in Y is affected only by the the j-th row in W.
Thus, only cells who's second and third entries are the same (Their index has the form of (i,j,j,k)) could be non-zero.
C2. **No.** we can calculate the vector-jacobian product directly:
$\delta\mat{W} = \delta\mattr{Y} * ~\mat{X}$.
"""

part1_q2 = r"""
**NO.**
It is possible to calculate the gradient of the loss w.t.r each of the parameters directly, but the use
of B.P makes it feasible to use gradient methods for training large multilayer networks.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.048
    lr = 0.019
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.085
    lr_momentum = 0.01
    lr_rmsprop = 0.0005
    reg = 0
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**The graphs match what we expected.** Dropout is a regularization method used 
to reduce overfitting and improve generalization. 

We can see that for the cases where dropout=0 and dropout=0.4, there seems to be over-fitting to the training data since the training accuracy is
significantly higher than the test accuracy.
By looking at the case where dropout=0.8, we can also see, that the training accuracy and the test accuracy are close, which could indicate low over-fitting
to the training data.

It is also evident that in the case of no dropout, the difference between training and test, both in loss and accuracy is extreme.

"""

part2_q2 = r"""
**It is possible**

the cross-entropy loss is influenced by the resulting ditribution over the class scores,
while the accuracy depends only on the ratio of correct predictions to all predictions.
that is why, for a few epochs, if the prediction is correct while the distribution is pretty
close to uniform for example, it is possible for the loss and accuracy to increase simultaneously.

"""

part2_q3 = r"""
1. Backpropogation is an algorithm for efficiently calculating the gradient
of the loss function w.r.t its parameters. Gradient decent is the algorithm used to find the local
minimum of the loss function by making small steps in the negative direction of the gradient (that could be claculated
using B.P).

2. Gradient Descent uses all samples in each iteration of the algorithm to calculate the gradient of
the loss while stochastic G.D uses a single sample each iteration.

3. One justification is that it is usually unfeasible to store the entire dataset in memory and save all the gradients.
Another justification is that in a lot of cases the functions that we are attempting to optimize a non-convex
so by stochastically selecting samples to train on SGD decreases the likelihood of the optimizer getting 
stuck in a local minimum.

4. **A. The two approaches are equivalent.** In GD we calculate the loss for the entire dataset and then use B.P
    to update the weights. If $\mathcal{L}(x)$ is the loss function, regular GD, would sum the loss of all
    samples ($n$ samples) and get $total\_loss = \mathcal{L}(X)=\sum_{i=1}^{n}\mathcal{L}(x_i)$.
    With the suggested batches approach, we divide our $n$ samples to $m$ batches, therefore, the size of each batch is
    $\sim \frac{n}{m}$. Hence, in this approach, the total loss for each batch $i$ is
    $batch_j\_loss = \mathcal{L}(X_j)=\sum_{i=1}^{\frac{n}{m}}\mathcal{L}(x_i\in X_j)$.
    Summing the loss from all batches we get $total\_loss = \sum_{j}^{m}\mathcal{L}(X_j)
    =\sum_{j}^{m}\sum_{i=1}^{\frac{n}{m}}\mathcal{L}(x_i\in X_j)=\sum_{i}^{n}\mathcal{L}(x_i\in X)$.
    We can see that the loss is equivalent to G.D.

    B. After each step we need to save the results in memory for the backward step, summing the losses. 
    That is, for each forward step, we need to save the gradient. The result is that after a number of forward steps, we fill up the memory
    and get an out of memory error.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 100
    activation = "relu"
    out_activation = "softmax"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.085
    weight_decay = 1e-4
    momentum = 0.28
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
1. It seems that the results a pretty good in terms of accuracy and loss, it might be the case that we haven't find the optimal minimizer,
and that the solution is sub-optimal, but the error is not large.

2. It seems that the generalization error of our model is not so high either since the results on the
test set are about the same for all epochs (with some ups and downs) so it doesn't seem that we over-fitted
the training data.

3. From looking at the boundry plot, it seems that there are areas that our model missed, large area of examples
that were labeld in correctly. that could indicate that the approximation error is large and we should consider
changing our model to better approximate the tru function.

 


"""

part3_q2 = r"""
**We would expect the FNR to be higher. From looking at the plot of the decision
boundry it seems that there are more red outliers in the blue zone
than blue outliers in the red zone, which would cause the FNR to be higher in the
training set (obviously we can also see that from the confusion matrix).
Because the training and validation are not sampled I.I.D, we can excpect the same result
on the validation set.**

"""

part3_q3 = r"""
**1. In this scenario we would choose a HIGHER threshold than the one calculated by the ROC curve.
In this scenario, we would like to minimize the number of false positives in order to
minimize cost and loss of life, because the cost of false negative is much lower in this scenario than
the cost of false positive both in terms of cost and risk.

2. In this scenario we would like to minimize false negatives because the cost of a false positive is
lower than  the cost of a false negative in this scenario. thus, we would want to lower the threshold.

**

"""


part3_q4 = r"""
1. It seems that when the depth is fixed and the width varies, the wider the model, the closer the decision boundary is to the true boundary between the two sets.
By increasing the width we add more features to the model,
and allow it to approximate more complex functions from a richer hypothesis class that results
in a more complex decision boundary.

2. The greater the depth of the model, the closer the decision boundary is to true boundary between the two sets.
Each layer provides additional linear classifier and a non-linear activation, thus 
allowing the model to emulate more complex functions, resulting in a decision boundary which provides
better accuracy on the test set.

3.1  The results of the deeper network (4,8) are better than those of the wider network.

 
3.2 The results of the deeper network (4,32) are better than those of the wider network.

It seems that the complexity of the boundry that is achieved by adding depth to the model, had more effect on
the results than having a very wide but shallow model, even though the number of parameters was the same.
This fact shows that you can achieve better results with a smaller model if you add comlexity with additional layers.


4. ** Yes** , The threshold selection improved the final result. in all network configurations the model performed 
better or at least not much worse on the test set compared to the validation set
This fact indicates that by tuning the threshold on a randomly selected validation set,
allows the model predict in a way that attempts to minimize FNR and FPR.


"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.1
    weight_decay = 1e-4
    momentum = 0.28
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
