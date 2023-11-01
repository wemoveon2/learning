# Notes

- [Reference](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html)

- Should always seed `torch` via `torch.manual_seed(1)`, CPU and GPU must both be seeded.
  - Seed GPU via `torch.cuda.manual_seed(1)` and `torch.cuda.manual_seed_all(1)`
  - Must also ensure operations use the deterministic implementation via `torch.backends.cudnn.deterministic = True`
- Torch tensors are very similar to numpy arrays
  - Tensor operations ending with `_` are **inplace** operations.

## Backpropagation

- Torch has auto diff, when the `requires_grad` attribute of a tensor is set to True, calling `backward()` on the final output will populate the gradients `grad` attribute.

## Continuous XOR

- `torch.nn` is the API for different NN layers.
- `torch.nn.functional` contains functions used in layers.

- NNs are composed of modules, each module must define `__init__` and `forward`.

```python
import torch.nn as nn 
import torch.nn.functional as F 

class Layer(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    pass
```
- `forward` is essentially `__call__` but with additional functionality such as automatically calling backward.

### Simple Classifier

- Following code represents a dense NN with a single hidden layer.

```python
class SimpleClassifier(nn.Module):
  def __init__(self, dim_in, dim_hidden, dim_out):
    super().__init__()
    self.linear1 = nn.Linear(dim_in, dim_hidden)
    self.activation = nn.Tanh()
    self.linear2 = nn.Linear(dim_hidden, dim_out)
  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x
```

- We apply the classifier to binary classification.
  - Loss is more accurate when calculated on the original output, so we will not apply sigmoid.

```python
model = SimpleClassifier(dim_in=2, dim_hidden=4, dim_out=1)
```

- Printing the model lists the submodules.
  - We can see the different parameters by calling `parameters()` or `named_parameters()` on the model.
    - Parameters are only registered for direct object attributes of the module.

### Data

- `torch.utils.data` contains functionality for loading data.
  - `data.Dataset` is an interface for accessing data, `data.DataLoader` is for efficiently loading the data during training.

#### Dataset

- Must implement three methods for `Dataset`:
  - `__init__`
  - `__len__(self) -> int` - For returning the length of the dataset.
  - `__getitem__(self, idx) -> Tuple` - For returning the sample and label at `idx`.

```python
```python
import torch.utils.data as data
class XORDataset(data.Dataset):

  def __init__(self, size, std=0.1):
    super().__init__()
    self.size = size
    self.std = std
    self.generate_continuous_xor()

  def generate_continuous_xor(self):
    # generate self.size rows of floats that are 0 or 1
    data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
    # create labels 
    labels = (data.sum(dim=1) == 1).int()
    # add gaussian noise to data
    data += self.std * torch.randn(data.shape)
    self.data = data
    self.labels = labels

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
```

#### DataLoader

- `torch.utils.data.DataLoader` is an iterable over a `Dataset`, supports many features such as batching, multi-process data loading, etc.
- Can create one by passing in an instance of a `Dataset` to the class along with additional arguments.

```python
dataset = XORDataset(500)
data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
```

- The initialized data loader will return tensors of size `[8, 2]`, since `shuffle=True` the order will be randomized.

### Loss, Backpropagation, and SGD

- We will use binary cross entropy as the loss for binary classification.
  - There are two loss functions, `nn.BCELoss` and `nn.BCEWithLogitsLoss`. We will use the latter because its more numerically stable.
```python
loss_module = nn.BCEWithLogitsLoss()
```
- `torch.optim` has the most popular optimizers. We will use SGD, which will take the product of the gradient and a learning rate before subtracting the product from the parameter value.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

- `optimizer.step()` updates the parameteres and `optimizer.zero_grad()` clears the previous gradients.

## Training

```python
dataset = XORDataset(size=2500)
data_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
model.to(device)

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
  model.train()

  for epoch in tqdm(range(num_epochs)):
    for inputs, labels in data_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      preds = model(inputs)
      preds = preds.squeeze(dim=1) # [b, 1] -> [b]

      loss = loss_module(preds, labels.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

train_model(model, optimizer, data_loader, loss_module)
```

## Saving a model

- To save a model, we extract the `state_dict` which contains all the learnable parameters.

```python
state_dict = model.state_dict()
torch.save(state_dict, "model.tar")
```

- We can load the trained model back by loading the `state_dict`.

```python
state_dict = torch.load("model.tar")
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
model.load_state_dict(state_dict)
```










