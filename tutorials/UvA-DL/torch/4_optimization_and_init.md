# Tutorial 4: Optimization and Initialization

- Review of techniques for optimizing and initializing NNs.
  - Large issue with increasing model depth is ensuring a stable gradient flow throughout all the layers, otherwise, can encounter issues such as vanishing and exploding gradients.
- Normalization of data to values between -1 and 1 or mean of 0 and stddev 1 both works well.
  - Calculate normalization parameters based off the mean and stddev of the dataset.
  - Validate normalization worked by checking the statistics on a single batch.

## Initialization 

- Good [source](https://pouannes.github.io/blog/initialization/#mjx-eqn-eqfwd_K) for math on initialization.
- Properties we want for a neural network:
  1. Variance of the input should be equivalent to the variance of the output.
    - Variance decreasing as we go deeper would result in a harder time optimizing the model as inputs between layers are nearly constant.
    - Variance increasing would result in it heading to infinity as we go deeper.
    - We want the variance of the input to a layer to be equal to the variance of the output of the layer.
  2. The variance of the gradients of each layer should be equal to each other.
    - If the variance of the gradients is not equal across layers, there would be differences in the scale of the gradients across layers. This makes it impossible to select a good learning rate since any value would be either too slow or too fast depending on the scale of the gradients of that layer.

### Constant Initialization 

- Initialize all weights to the same constant value, a value close to 0 (setting it to 0 would cause gradients to be 0).
- Results in the first and last layers having a diverse gradient distribution, but with the hidden layers having the same gradients for all weights.
  - This results in there being a single parameter for those layers, since the weights are all the same.

### Constant Variance

- Randomly sample from a distribution with a fixed variance for all layers.
- With a small variance, results in the variance of the activation decreasing across layers. With a bigger variance, the variance of the activations increase across layers.
  - Possible to find a good variance, but it'll be dependent on the size and type of the network.

### Finding an Appropriate Initialization Value

- We have two requirements from the perspective of the activation distribution:
  1. Mean of activations should be zero.
  2. Variance of the activations should stay the same across layers.

- Say we want to design an initialization for $y = Wx+b, y \in \mathbb{R}^{d_{y}}, x \in \mathbb{R}^{d_{x}}$
  - We want $Var(y_i) = Var(x_i) = \sigma^2_x$ and mean is 0.
  - Assume $x$ to have a mean of 0.
  - $b$ is a single element constant across inputs, so we set it to 0.
- We can calculate the variance for initialization with:

$$
\begin{align}
y_i &= \sum_j w_{ij}x_{j} \\
Var(y_i) &= \sigma^2_x = Var(\sum_j w_{ij}x_{j}) \\ 
&= \sum_j Var(w_{ij}x_{j}) \\ 
&= \sum_j Var(w_{ij}) \cdot Var(x_j) \text{Variance rule with expectation being 0} \\
&= d_x \cdot Var(w_{ij}) \cdot Var(x_j) \text{Variance equal for all elements} \\
&= \sigma^2_x \cdot d_x \cdot Var(w_{ij}) \\ 
Var(w_{ij}) &= \sigma^2_{W} = \frac{1}{d} \\
\end{align}
$$

- So we should initialize the weights with variance equal to the inverse of the input dimension $d_x$.
  - We can use any distribution with mean of 0 and variance of $\frac{1}{d}$. Uniform distribution is preferred so we can exclude the chance of initializing with very large or small values.
- We apply the same process to the gradients starting from $\Delta x = W\Delta y$, which brings us to the conclusion that the weights should be initialize with the inverse of the output dimensions, $\frac{1}{d_y}$.
- Using the harmonic mean of the two values gives us the **Xavier initialization**.

$$
W \sim \mathcal{N}(0, \frac{2}{d_x + d_y})
$$

- This initialization works for linear activation functions, but this also works with tanh as we can assume that for small values, tanh works as a linear function.
- This initialization however **does not** work for activation functions such as ReLU because it sets half of the inputs to 0 so the expectation of the input is not zero.
  - This results in desired weight variance becoming $\frac{2}{d_x}$, giving us the **Kaiming initialization**.
    - Note that this initialization does not using the harmonic mean between input and output size, argued that using either will lead to stable gradients.

## Optimization

```python
class OptimizerTemplate:
  def __init__(self, params, lr):
    self.params = list(params)
    self.lr = lr 

  def zero_grad(self):
    for p in self.params:
      if p.grad is not None:
        p.grad.detach_() # important for 2nd order optimizer
        p.grad.zero_()
  
  @torch.no_grad()
  def step(self):
    for p in self.params:
      if p.grad is None: # skip params without grad
        continue
      self.update_param(p)
  
  def update_param(self, p):
    raise NotImplementedError
```


- An optimizer updates the network's weights given the gradients. It is essentially $w^t=f(w^{t-1}, g^t)$ where $w$ is the parameters and $g^t=\Delta_{w^{t-1}}\mathbb{L}^t$ is the gradients at time $t$. 
  - We also commonly include $\lambda$, the learning rate, as a parameter. It is essentially the step size of the update.

- The first optimizer is stochastic gradient descent, $w^t=w^{t-1}-\lambda \cdot g^t$
  - Can include momentum by replacing the gradient with it. Momentum is the exponential average of all previous gradients including the current.

```python 
class SGD(OptimizerTemplate):
  def __init__(self, params, lr):
    super().__init__(params, lr)

  def update_param(self, p):
    p_update = -self.lr * p.grad
    p.add_(p_update) # inplace op saves mem

class SGDMomentum(OptimizerTemplate):
  def __init__(self, params, lr, momentum=0.0):
    super().__init__(params, lr)
    self.momentum = momentum
    self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}

  def update_param(self, p):
    self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
    p_update = -self.lr * self.param_momentum[p]
    p.add_(p_update) # inplace op saves mem
```

- Adam combines momentum with an adaptive learning rate based on an exponential average of the gradients.

```python
class Adam(OptimizerTemplate):

  def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    super().__init__(self, params, lr)
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.param_step = {p: 0 for p in self.params}
    self.param_momenum = {p: torch.zeros_like(p.data) for p in self.params}
    self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}

  def update_param(self, p):
    self.param_step[p] += 1 

    self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
    self.param_2nd_momentum[p] = (1 - self.beta2) * p.grad ** 2 + self.beta2 * self.param_2nd_momentum[p]

    bias_corr_1 = 1 - self.beta1 ** self.param_step[p]
    bias_corr_2 = 1 - self.beta2 ** self.param_step[p]

    p_mom = self.param_momentum[p] / bias_corr_1
    p_2nd_mom = self.param_2nd_momentum[p] / bias_corr_2
    p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
    p_update = -p_lr * p_mom 
    
    p.add_(p_update)
```

### Comparing Optimization Techniques

- All three optimization techniques perform similarly well, but this can be due to the initialization used.
  - Adam is usually more robust due to the adaptive learning rate.

#### Pathological Curvatures 

- These are a type of surface similar to ravines and is particularly tricky for plain SGD.
  - Contains steep gradients in one direction with a (local) optimum, while a second direction has a slower gradient towards a global optimum.

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial4_Optimization_and_Initialization_59_0.svg)

- Ideally, the optimization algorithm finds the center of the ravine and focuses on optimizing parameters towards $w_2$.
  - However, if we encounter a point along the ridge, the gradient is much greater in $w_1$ than $w_2$, can lead us to jump from side to side of the ravine.
  - We would have to reduce learning rate.

- Comparing the three optimization techniques on this loss surface, we can see that plain SGD is unable to find the global optimum. Adam and SGD with momentum are able to converge while plain SGD fails to do so.

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial4_Optimization_and_Initialization_65_0.svg)

#### Steep Optima

- Second type of challenging loss surface are steep optima. These surfaces have small gradients for most of the surface while around the optimum there's very large gradients.

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial4_Optimization_and_Initialization_68_0.svg)
- Can expect adaptive learning rate to be crucial

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial4_Optimization_and_Initialization_70_0.svg)

- Only Adam converges.

#### What Optimizer to Use

- While Adam can be seen to be superior to SGD, papers show that SGD with momentum generalizes better while Adam tends to overfit.
  - Related to the idea of finding *wider* optima

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/flat_vs_sharp_minima.svg)

- Black line is training loss surface, red line is test.
- Finding sharp, narrow minima minimizes train loss but not test loss since a small change can have a significant impact for sharp minima, while a flat minima is more robust to such changes.








