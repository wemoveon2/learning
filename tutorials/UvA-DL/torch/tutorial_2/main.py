import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 


class SimpleClassifier(nn.Module):


    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)


class XORDataset(data.Dataset): 
    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std
        self._generate_dataset()
    
    def _generate_dataset(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)
        self.data = data
        self.labels = label

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


model = SimpleClassifier(2, 4, 1)
dataset = XORDataset(300)
dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1*10**-3)

def train_model(model, optimizer, data_loader, loss_module, epochs=100):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    model.train()
    model.to(device)

    for epoch in tqdm(range(epochs)):
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs).squeeze(dim=1)

            loss = loss_module(preds, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model




model = train_model(model, optimizer, dataloader, loss)

def save_model(model, dir):
    state_dict = model.state_dict()
    save_path = os.path.join(dir, "model.tar")
    torch.save(state_dict, save_path)
    return save_path

save_path = save_model(model, "./")
print(f"Model saved to {save_path}")

def eval_model(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    model.eval()
    model.to(device)
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).squeeze(dim=1)
            preds = torch.sigmoid(preds)
            pred_labels = (preds >= 0.5).long()

            true_preds += (pred_labels == labels).sum()
            num_preds += labels.shape[0]
        acc = true_preds / num_preds
        print(f"Accuracy of model = {100 * acc:4.2f}%")

eval_dataset = XORDataset(500)
eval_dataloader = data.DataLoader(eval_dataset, batch_size=128)
eval_model(model, eval_dataloader)








