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
1. The model has a high Optimization error. This is evident by the increase in the loss function 
towards the end of the testing process.A plausible reason could be that the learning rate is too large

Furthermore, The Training set's loss function seems to have plateaued, this could also indicate that there
is a need for a smaller learning rate or perhaps a different loss function.


The last possible culprit responsible for the optimization error could be the depth of the network, 
as it introduces more parameters which complicates the optimization problem, which could possibly result 
a subpar optimization of the model.

2. The model has a high generalization error, the potential cause for this could be that the weight decay 
parameter is too small.

Additionally, we can see that the epoch in which the model scored the best on test set is not the 
last one but the 11th instead (88.5 vs 90.8) this could be the result of high learning rate and momentum


3. The model does not have a high approximation error. The model scored 95% on the training set and seemed
to have converged. If the 5% error rate is still unacceptable, a possible remedy could, a smaller learning 
rate, a different loss function which could have landscape features that are easier to optimize on. The 
activation functions and the final activation function could also be tuned to further reduce the error.



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
**1. It seems that when the depth is fixed and the width varies, the wider the model, the closer the decision boundary is to the true boundary between the two sets.
By increasing the width we add more features to the model,
and allow it to approximate more complex functions from a richer hypothesis class that results
in a more complex decision boundary.

2. The greater the depth of the model, the closer the decision boundary is to true boundary between the two sets.
Each layer provides additional linear classifier and a non-linear activation, thus 
allowing the model to emulate more complex functions, resulting in a decision boundary which provides
better accuracy on the test set.

3.1  The results of the deeper network are better than those of the wider network.

 
3.2 The results of the deeper network are better than those of the wider network.

It is possible that the deeper models are able to create a good decision boundary by creating 
"linear segments" that are each able to separate a part of plane and ultimately combining them all with
non-linear functions as the "segment delimiters" is what gives this network configuration a better accuracy.


4. Yes, The threshold selection improved the final result. in all network configurations the model performed 
better on the test set compared to the validation set
or slightly better on the validation set with no significant drop-off in the test set accuracy.
This performance indicates that by tuning the threshold with threshold selection,
the model finds a threshold the more accurately reflects the data,
thus allowing it to perform well on the test set.**


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

(1)  For a convolutional filter consisting of $F\times F$ sized kernels, we'll get $C_{in}\cdot C_{out}\cdot F^{2}$ parameters.
A layer with $K$ kernels will have $K\cdot(C_{in}\cdot F^{2}+1)$ parameters.
Thus, the number of parameters in the regular block will be: $256\cdot (64\cdot 3^{2}+1)+64\cdot(256\cdot 3^{2}+1)=295232$.
The number of parameters in the bottleneck block will be: $64\cdot(256 \cdot 1^{2}+1)+64\cdot(64\cdot 3^{2}+1)+256\cdot(64\cdot 1^{2}+1)=70016$.
We clearly see that the the bottleneck block has much less parameters than the regular block.

(2) The shape of the input should be $(C_{in},H,W)$ and the shape of the kernel should be $F\times F$.
We have the following formula: $H\cdot W\cdot(C_{in}\cdot F^{2}+(C_{in}\cdot F^{2}-1)+1)=2 C_{in} F^{2}\cdot H W$.
To calculate the total number of floating point operations, we need to insert the correct shapes and number of kernels, use the layers as described before, set the padding and stride so that they do not affect the shape, and add the number of floating point operations required for the skip-connections.
Thus we get: \sum_{i}(K\cdot 2 C_{in} F^{2}\cdot H W) + C_{sc}\cdot HW$, for all layers $i$ in the block.
Finally, the number of floating point operations for the regular block is: $64\cdot2\cdot256\cdot3^{2}\cdot H W+256\cdot2\cdot64\cdot3^{2}\cdot H W +256 H W=590080 H W$, while the number of floating point operations for the bottleneck block is: $64\cdot2\cdot256\cdot1^{2}\cdot H W+64\cdot2\cdot64\cdot3^{2}\cdot H W+256\cdot2\cdot64\cdot1^{2}\cdot H W+256 H W=139520 H W$.

(3) The bottleneck lowers the dimension of the feature space, and we used it here to reduce the dimension from $256d$ to $64d$. Looking in the feature maps, we see that the output of a bottleneck block and a regular block differ by the shape of the subtensor they depend on (which are $256\times3\times3$ and $256\times5\times5$ respectively).
The bottleneck block thus loses some ability to spatially combine the input. Looking across the feature maps, we see that a bottleneck block with a reduced dimension of $64d$, every feature is a combination of the original 256 features.
This causes its ability to combine the input across feature maps to diminish, while the regular block still considers all 256 of the original features.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

(1) Examining the effect of depth on accuracy, we see that for both $K = 32$ and $K = 64$, depth $L = 4$ yields the best results.
We see in the graphs that $L = 2$ and $L = 4$ produce decent results, while the accuracy gained by $L = 4$ is slightly higher.
A possible reason for this is that $L = 4$ produces a deeper network, and thus it can extract better features than $L = 2$.

(2) We saw that there are indeed values of $L$ for which the network is not trainable.
They are $L=8$ and $L=16$, for which the training and test accuracies were very low (around $10\%$).
This is similar to the accuracy that would be achieved by a random classifier, and so we see that the network wasn't really trained well.
This is probably caused by the fact that the network's depth ($L\geq 8$), which lead to vanishing gradients.
Two things which we could do to fix that is to use batch normalization (to help make sure that the signal is not diminished by shifting distributions across the network during backpropagation) or to use a residual network (such networks can still learn even if the depth is large, allowing gradient information to pass through the layers).
"""

part5_q2 = r"""
**Your answer:**

From our graphs, we can see that we obtained good results for $L=2$ and $L=4$ across all $K$ values, while the networks with $L=8$ did not fare so well.
The best network was with $L=4$ (once again!) and $K=256$, for which the test accuracy was the highest (greater than $70\%$).
In general, the higher $K$ values (128, 256) produced the highest accuracies and lowest losses.
Comparing between experiments 1.1 and 1.2, we see that our results in both of them line up with each other, thought we still found better results in 1.2 by increasing $K$ further.
"""

part5_q3 = r"""
**Your answer:**

In this part, our results weren't so great. The only depth that seemed to be trainable was $L=1$, which is a bit odd.
This is possibly due to a poor choice of parameters, rather than actually being a direct result of the depth. 
Depth can still be a factor though, while greater depths like $L=3$ (which actually leads to the depth being $3L=9$ since $K$ is a list of 3 values) could cause vanishing gradients like we saw before.
For $L=1$, our results were better than those of the previous experiments, when we consider both the accuracies and the losses.
This is likely due to the higher values $K$ used here, since the network can use a larger number of features during the learning process.
"""

part5_q4 = r"""
**Your answer:**

Unlike the previous experiments, here we used a residual network in our architecture.
The use of skip-connections should grant better performance from the model, and looking at our results that's exactly what we got.
We got good accuracy values here on the training and test sets, and the network was even able to reach a test accuracy of over $70\%$ for $L=32$!
We note that this type of network was trainable even for high depth values, which was not so much the case previously.
This is eaxctly the advantage of using a residual network, which can help us train deep networks while still preventing any vanishing gradients from occurring.
In this experiment, we see that we've obtained some of our best results so far.
"""

part5_q5 = r"""
**Your answer:**

(1) We utilized a straightforward ResNet model, but this time with batch normalization, ReLU activation, drouput of 0.1, and max. pooling.

(2) The test loss and accuracy for this model, especially for $L=12$, was fantastic. It outperformed the models from experiment one, with a test accuracy of almost $80\%$!
"""
# ==============
