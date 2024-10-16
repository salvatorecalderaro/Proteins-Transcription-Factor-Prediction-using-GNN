from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch
from torch_geometric.profile import timeit
from tqdm import tqdm
import torch.nn as nn

def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Performs a forward pass through the GraphTFactor model.

    Args:
        x (torch.Tensor): The input node features of shape (num_nodes, in_features).
        edge_index (torch.Tensor): The edge indices of shape (2, num_edges).
        batch (torch.Tensor): The batch tensor of shape (num_nodes,).

    Returns:
        torch.Tensor: The raw logits of shape (num_nodes, num_classes).

    This method first applies a series of convolutional layers using the Graph Convolutional Network (GCN)
    architecture, followed by a readout layer and a final linear layer. The output is a tensor of raw
    logits, which can be used for classification tasks.
    """
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
    """
    Trains the given neural network (`net`) on the provided training dataset (`trainloader`) for a specified number of epochs (`epochs`) and learning rate (`lr`).

    Args:
        device (torch.device): The device to run the training on.
        net (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): The training dataset loader.
        epochs (int): The number of epochs to train the model for.
        lr (float): The learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the trained neural network model (`net`) and the elapsed time for training (`elapsed_time`).

    This function initializes an Adam optimizer, sets a class weight for class 1, and defines a binary cross-entropy loss function with a positive weight for class 1. It then iterates over the specified number of epochs, training the model on each batch of the training dataset. The loss for each batch is computed and backpropagated, and the model parameters are updated using the optimizer. The average loss for each epoch is calculated and printed. Finally, the elapsed time for training is returned along with the trained model.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    class_weights = torch.tensor([3.0]).to(device)  # Weighting class 1 more heavily
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[0])
    print("Training....")
    with timeit.timeit as time_counter:
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
    """
    Predicts the binary classification labels for the given test dataset using the provided neural network model.

    Args:
        device (torch.device): The device to run the prediction on.
        net (torch.nn.Module): The neural network model to be used for prediction.
        testloader (torch.utils.data.DataLoader): The test dataset loader.

    Returns:
        tuple: A tuple containing the true labels (y_true), predicted labels (y_pred), and probabilities (proba) for the test dataset.

    This function evaluates the provided neural network model on the given test dataset. It iterates over the test dataset, performing a forward pass through the model for each batch of data. The predicted probabilities are obtained using the sigmoid function, and the binary predictions are obtained by rounding the probabilities. The true labels, predicted labels, and probabilities are collected and returned as a tuple.
    """
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