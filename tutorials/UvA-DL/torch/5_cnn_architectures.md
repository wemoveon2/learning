# Tutorial 5: Inception, Resnet, and DenseNet

- Discussion on modern CNN architectures

## PyTorch Lightning

- NOTE: `pytorch-lightning` was renamed to `lightning`

```python
import lightning as L
```

- Framework for eliminating boilerplate code for training, evaluation, and testing.

  - Omits the need for having own seed function, can call `L.seed_everthing(42)`.

- PL has `L.LightningModules`, which inherit from `nn.Module`, this organizes code into 5 sections:
  1. `__init__`: For initializing parameters (sub-modules in practice) of the model.
  2. `configure_optimizers`: For creating the optimizer.
  3. `training_step`: Where the loss calculation for a single batch is defined.
  4. `validation_step`: Similar to the training, defines what happens in each step.
  5. `test_step`: Ditto.
- PL reorganizes torch code instead of abstracting them away.

```python
class CIFARModule(L.LightningModule):

  def __init__(self, model_name: str, model_hparams: dict, optimizer_name: str, optimizer_hparams: dict):
    super().__init__()
    self.save_hyperparameters() # this saves __init__ params to the checkpoint
    # can get init params from self.hparams now
    self.model = create_model(model_name, model_hparams)
    self.loss_module = nn.CrossEntropyLoss()
    self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

  def forward(self, imgs):
    return self.model(imgs)

  def configure_optimizers(self):
    if self.hparams.optimizer_name == "Adam":
      optimizer = optim.AdamW(
        self.parameters(), **self.hparams.optimizer_hparams
      )
    elif self.hparams.optimizer_name == "SGD":
      optimizer = optim.SGD(self.parameters, **self.hparams.optimizer_hparams)
    else:
      raise ValueError(f"Invalid optimizer name {self.hparams.optimizer_name}")
    # reduce LR by 0.1 after 100 and 150 epochs
    scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=[100, 150], gamma=0.1
    )
    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    loss = self.loss_module(preds, labels)
    # get index of max value in last dim representing the predicted label
    acc = (preds.argmax(dim=-1) == labels).float().mean()
    # Logs the accuracy per epoch to tensorboard (weighted average over batches)
    self.log('train_acc', acc, on_step=False, on_epoch=True)
    self.log('train_loss', loss)
    return loss # for calling .backwards() on

  def validation_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs).argmax(dim=-1)
    acc = (preds == labels).float().mean()
    self.log("val_acc", acc)

  def test_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs).argmax(dim=-1)
    acc = (preds == labels).float().mean()
    # default is to log per epoch, avgd over batches
    self.log("test_acc", acc)
```

- Callbacks are a big part of PL, contains non-essential logic used to provide utilities.
  - `LearningRateMonitor` logs the LR to TensorBoard and `ModelCheckpoint` allows configuration of how checkpoints are saved.
- We define mappings of `map[str]object` so we can pass in strings to our PL module. This allows them to be saved in the hyperparameters.
- `L.Trainer` is the second most important module (behind `L.LightningModule`), it's used to execute the training steps defined in the module.
  - `trainer.fit` takes a lightning module, a dataset, and an optional validation dataset and trains the module on the dataset.
  - `trainer.test` takes a model and dataset and returns the test metric.

```python
def train_model(model_name: str, save_name=None, **kwargs):
  save_name = model_name if save_name is None else save_name

  trainer = L.Trainer(
    default_root="./",
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=180,
    callbacks=[
      ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
      LearningRateMonitor("epoch"),
    ],
    enable_progress_bar=True,
  )
  trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
  trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

  # Check whether pretrained model exists. If yes, load it and skip training
  pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
  if os.path.isfile(pretrained_filename):
      print(f"Found pretrained model at {pretrained_filename}, loading...")
      model = CIFARModule.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
  else:
      pl.seed_everything(42) # To be reproducable
      model = CIFARModule(model_name=model_name, **kwargs)
      trainer.fit(model, train_loader, val_loader)
      model = CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

  # Test best model on validation and test set
  val_result = trainer.test(model, val_loader, verbose=False)
  test_result = trainer.test(model, test_loader, verbose=False)
  result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

  return model, result
```

## Architectures

### Inception

- Created by Google in 2014, won ImageNet.
- Inception blocks applies four convolution blocks separately on the same feature map.
  - 1x1, 3x3, and 5x5

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/inception_block.svg)

- Additional 1x1 convolutions before 3x3 and 5x5 are for dimensionality reduction.

```python
class InceptionBlock(nn.Module):
  def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
    """
    Inputs:
        c_in - Number of input feature maps from the previous layers
        c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
        c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
        act_fn - Activation class constructor (e.g. nn.ReLU)
    """
    super().__init__()
    self.conv_1x1 = nn.Sequential(
      nn.Conv2d(c_in, c_out['1x1'], kernel_size=1),
      nn.BatchNorm2d(c_out['1x1']),
      act_fn()
    )

    self.conv_3x3 = nn.Sequential(
      nn.Conv2d(c_in, c_red['3x3'], kernel_size=1),
      nn.BatchNorm2d(c_red['3x3']),
      act_fn(),
      nn.Conv2d(c_in, c_out['3x3'], kernel_size=3, padding=1),
      nn.BatchNorm2d(c_out['3x3']),
      act_fn(),
    )

    self.conv_5x5 = nn.Sequential(
      nn.Conv2d(c_in, c_red['5x5'], kernel_size=1),
      nn.BatchNorm2d(c_red['5x5']),
      act_fn(),
      nn.Conv2d(c_in, c_out['5x5'], kernel_size=5, padding=2),
      nn.BatchNorm2d(c_out['5x5']),
      act_fn(),
    )

    self.max_pool = nn.Sequential(
      nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
      nn.Conv2d(c_in, c_out["max"], kernel_size=1),
      nn.BatchNorm2d(c_out["max"]),
      act_fn()
    )
  def forward(self, x):
    x_1x1 = self.conv_1x1(x)
    x_3x3 = self.conv_3x3(x)
    x_5x5 = self.conv_5x5(x)
    x_max = self.max_pool(x)
    x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
    return x_out
```

- The overall architecture consists of multiple Inception blocks with occasional pooling operations to reduce dimensions.

```python
class GoogleNet(nn.Module):

    def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn()
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, act_fn=self.hparams.act_fn),
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn)
        )
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
```

### ResNet

- One of the most cited AI papers, the architecture introduced residual connections, allowing for deeper networks.
- Based on the idea of modeling $x_{l+1}=x_l+F(x_l)$ instead of $x_l=F(x_l)$, which allows for stable gradient propagation deeper into the network as there's a bias for the identity matrix in the derivative.

![](https://uvadlc-notebooks.readthedocs.io/en/latest/_images/resnet_block.svg)

- Original ResNet applies non-linear activation function after the skip connection, pre activation ResNet applies it at the start of each $F$.
  - Pre activation ResNet is better for deeper networks since the gradient flow is guaranteed to have the identity matrix since its not affected by the activation function.
