"""
This file contains the implementation of the different classes considered.
"""
import torch
from torch.nn import Linear, Parameter, PairwiseDistance, CosineSimilarity
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv

class GCN_k_m(torch.nn.Module):
    """
    Takes as input a pair of graphs which are both fed through k graph convolutional layers and m linear layers.
    conv1 -> dropout -> ...         -> convk -> meanpool -> relu -> compute dist(x1, x2)
    If apply_relu_conv is True:
    conv1 -> dropout -> relu -> ... -> convk -> meanpool -> relu -> compute dist(x1, x2)
    If a number of linear layers m >= 1 is specified:
    conv1 -> dropout -> ...         -> convk -> meanpool -> relu -> linear1 -> relu -> ... -> linearm -> compute dist(x1, x2)
    
    If mlp_dist is set to True:
    The distance between the two embedding vectors is obtained by applying a linear transformation on the difference between the two.
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, n_conv_layers, n_linear_layers, p, name, apply_relu_conv = False, dist = 'L1', mlp_dist = False):
        super(GCN_k_m, self).__init__()

        if n_conv_layers < 1:
            raise ValueError("Invalid value for n_conv_layers. n_conv_layers should be an integer larger than 0")
        if dist not in ['cosine', 'L1', 'L2']:
            raise ValueError("Invalid value for 'dist'. Expected 'cosine', 'L1', or 'L2'.")
        
        # Details for the architecture
        self.dist = dist
        self.mlp_dist = mlp_dist
        self.input_features = input_features
        self.output_embeddings = output_embeddings 
        self.name = name

        # Hyper-parameters.
        self.apply_relu_conv = apply_relu_conv # If True applies relu after each convolutional layer
        self.p = p                             # Sets dropout probability. If 0, no dropout is allowed
        self.hidden_channels = hidden_channels # Sets dimension of hidden channels
        self.n_conv_layers = n_conv_layers     # Sets number of convolutional layers.
        self.n_linear_layers = n_linear_layers # Sets number of linear layers.


        # GCN and Linear layers employed by the model.
        self.GCN_layers = torch.nn.ModuleList()
        self.Linear_layers = torch.nn.ModuleList()

        # If no linear layers
        if self.n_linear_layers == 0:
            for i in range(self.n_conv_layers-1): 
                self.GCN_layers.append(GCNConv(input_features, hidden_channels))
                input_features = hidden_channels # From second layer we'll need this.and
            # Final GCN layer
            self.final_GCN = GCNConv(hidden_channels, output_embeddings)
        
        # If there are some linear layers
        else:
            for i in range(self.n_conv_layers): 
                self.GCN_layers.append(GCNConv(input_features, hidden_channels))
                input_features = hidden_channels # From second layer we'll need this.and
            # Linear layers
            for i in range(self.n_linear_layers - 1):
                self.Linear_layers.append(Linear(hidden_channels, hidden_channels))
            # Final linear layer
            self.final_linear = Linear(hidden_channels, output_embeddings)

        # Additional layers required later
        self.dropout = torch.nn.Dropout(self.p)
        self.relu = torch.nn.ReLU()

        if self.mlp_dist:
            self.linear_dist = Linear(output_embeddings, 1)
        
        # Define the operation to be performed on the distances based on the the specification.
        if self.dist == 'L1':
            self.pdist = PairwiseDistance(p=1)
        elif self.dist == 'L2':
            self.pdist = PairwiseDistance(p=2)
        elif self.dist == 'cosine':
            self.pdist = CosineSimilarity()

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2): # Need a way to extract these from dataloader

        # 1. Obtain node embeddings for graph 1 and 2
        for layer in self.GCN_layers: 
            x1 = layer(x1, edge_index1)
            x1 = self.dropout(x1)
            x2 = layer(x2, edge_index2)
            x2 = self.dropout(x2)
            if self.apply_relu_conv:
                x1 = self.relu(x1)
                x2 = self.relu(x2)
        
        if self.n_linear_layers == 0:
            x1 = self.final_GCN(x1, edge_index1)
            x2 = self.final_GCN(x2, edge_index2)

        # 2. Readout layer followed by RELU (and Linear layers).
        x1 = global_mean_pool(x1, batch1)
        x1 = self.relu(x1)
        x2 = global_mean_pool(x2, batch2)
        x2 = self.relu(x2)

        if self.Linear_layers:
            for layer in self.Linear_layers:
                x1 = layer(x1)
                x1 = self.relu(x1)
                x2 = layer(x2)
                x2 = self.relu(x2)
            
            x1 = self.final_linear(x1)
            x2 = self.final_linear(x2)

        if self.mlp_dist:
            # Maybe try Euclidean distance
            x = torch.abs(x1 - x2)
            vdist = self.linear_dist(x).reshape(-1)
            return vdist

        # Compute the corresponding distance between the embeddings.
        if self.dist == 'cosine':
            vdist = 1 - self.pdist(x1, x2)
        else:
            vdist = self.pdist(x1, x2)
        return vdist
    
    def save(self):
        """
        Saves the model state dictionary in models folder
        """
        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)


class GCN3(torch.nn.Module):
    """
    Takes as input a pair of graphs which are both fed through 3 convolutional layers each followed by an activation function.
    Here 3 graph convolutional layers were used, without any form of parameter sharing.
    Specify the distance matrix as one of cosine, L1, L2
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, name, dist = 'L2'):
        super(GCN3, self).__init__()

        self.dist = dist

        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_embeddings)
        self.name = name
        
        if self.dist not in ['cosine', 'L1', 'L2']:
            raise ValueError("Invalid value for 'dist'. Expected 'cosine', 'L1', or 'L2'.")
        elif self.dist == 'L1':
            self.pdist = PairwiseDistance(p=1)
        elif self.dist == 'L2':
            self.pdist = PairwiseDistance(p=2)
        elif self.dist == 'cosine':
            self.pdist = CosineSimilarity()


    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2): # Need a way to extract these from dataloader
        # 1. Obtain node embeddings for graph 1
        x1 = self.conv1(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        # 2. Readout layer
        x1 = global_mean_pool(x1, batch1)  # [batch_size, hidden_channels]
        # 3. Apply a final linear transformation on the aggregated embedding
        x1 = torch.nn.functional.dropout(x1, p=0.5)
        x1 = self.lin(x1)

        # 1. Obtain node embeddings for graph 2
        x2 = self.conv1(x2, edge_index2)
        x2 = x2.relu()
        x2 = self.conv2(x2, edge_index2)
        x2 = x2.relu()
        x2 = self.conv3(x2, edge_index2)
        # 2. Readout layer
        x2 = global_mean_pool(x2, batch2)  # [batch_size, hidden_channels]
        # 3. Apply a final linear transformation on the aggregated embedding
        x2 = torch.nn.functional.dropout(x2, p=0.5)
        x2 = self.lin(x2)
        if self.dist == 'cosine':
            vdist = 1 - self.pdist(x1, x2)
        else:
            vdist = self.pdist(x1, x2)
        return vdist
    
    def save(self):
        """
        Saves the model state dictionary in models folder
        """
        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)


class GCN3_MLP(torch.nn.Module):
    """
    Takes as input a pair of graphs which are both fed through 3 convolutional layers each followed by an activation function
    3 graph convolutional layers (Welling) that share parameters.
    It computes the distance using a linear function of the final embedding instead of the distance between two graph embeddings
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, name):
        super(GCN3_MLP, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, output_embeddings)
        self.lin2 = Linear(2*output_embeddings, 1)
        self.name = name
        
    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2): # Need a way to extract these from dataloader
        # 1. Obtain node embeddings for graph 1
        x1 = self.conv1(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        # 2. Readout layer
        x1 = global_mean_pool(x1, batch1)  # [batch_size, hidden_channels]
        # 3. Apply a final linear transformation on the aggregated embedding
        x1 = torch.nn.functional.dropout(x1, p=0.5)
        x1 = self.lin1(x1)

        # 1. Obtain node embeddings for graph 2
        x2 = self.conv1(x2, edge_index2)
        x2 = x2.relu()
        x2 = self.conv2(x2, edge_index2)
        x2 = x2.relu()
        x2 = self.conv3(x2, edge_index2)
        # 2. Readout layer
        x2 = global_mean_pool(x2, batch2)  # [batch_size, hidden_channels]
        # 3. Apply a final linear transformation on the aggregated embedding
        x2 = torch.nn.functional.dropout(x2, p=0.5)
        x2 = self.lin1(x2)

        # Compute the distance on the concatenation of the two using an MLP
        x = torch.cat((x1, x2), 1)
        x = self.lin2(x)
        return x

    
    def save(self):
        """
        Saves the model state dictionary in models folder
        """
        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)