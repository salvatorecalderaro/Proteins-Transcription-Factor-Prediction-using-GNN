from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch
from torch_geometric.profile import timeit
from tqdm import tqdm
import torch.nn as nn

class GraphTFactor(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, dropout_rate=0.5):
        super(GraphTFactor, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout_rate)  # Dropout layer

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the first convolution

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the second convolution

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout after the third convolution

        # 2. Readout layer (alternative pooling options)
        x = global_mean_pool(x, batch)  # Alternative: global_max_pool, global_add_pool
        
        # 3. Apply a final linear layer (logits)
        x = self.lin(x)
        return x  # Raw logits returned


def train_net(device, net, trainloader, epochs, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    class_weights = torch.tensor([3.0]).to(device)  # Weighting class 1 more heavily
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[0])
    print("Training....")
    with timeit() as time_counter:
        for epoch in tqdm(range(epochs)):
            net.train()
            epoch_loss = 0
            for i, data in enumerate(trainloader):
                # Iterate in batches over the training dataset
                data.x = data.x.float().to(device)
                data.y = data.y.float().to(device)
                data.edge_index = data.edge_index.to(device)
                data.batch = data.batch.to(device)
                
                # Forward pass
                out = net(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.view(-1, 1))  # Compute loss (logits to criterion)
                loss.backward()  # Backpropagation
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.detach().item()
            epoch_loss /= (i + 1)
    elapsed_time = time_counter.duration
    return net, elapsed_time


def predict(device, net, testloader):
    y_true = []
    y_pred = []
    proba = []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            data.x = data.x.float().to(device)
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            
            # Forward pass
            out = net(data.x, data.edge_index, data.batch)
            out = torch.sigmoid(out)  # Apply sigmoid since raw logits are returned
            
            proba.extend(out.tolist())
            pred = torch.round(out).tolist()  # Round to get binary predictions
            y_pred += pred
            y_true += data.y.tolist()  # True labels
    
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred).reshape(-1)
    proba = torch.tensor(proba).reshape(-1)
    return y_true, y_pred, proba
