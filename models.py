"""
This file contains the implementation of the different classes considered.
"""
import torch
from torch.nn import Linear, Parameter, PairwiseDistance, CosineSimilarity
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.nn.norm import BatchNorm

class GCN_k_m(torch.nn.Module):
    """
    Takes as input a pair of graphs which are both fed through k graph convolutional layers and m linear layers (with k >= 1 and m >= 0).
    conv1 -> relu -> ... -> convk -> meanpool -> compute dist(x1, x2)
    If a number of linear layers m >= 1 is specified:
    conv1 -> ...         -> convk -> meanpool -> linear1 -> relu -> ... -> linearm -> compute dist(x1, x2)
    
    If mlp_dist is set to True:
    The distance between the two embedding vectors is obtained by applying a linear transformation on the difference between the two.
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, n_conv_layers, n_linear_layers, name, dist = 'L1'):
        super(GCN_k_m, self).__init__()

        if n_conv_layers < 1:
            raise ValueError("Invalid value for n_conv_layers. n_conv_layers should be an integer larger than 0")
        if dist not in ['cosine', 'L1', 'L2']:
            raise ValueError("Invalid value for 'dist'. Expected 'cosine', 'L1', or 'L2'.")
        
        # Details for the architecture
        self.dist = dist
        self.input_features = input_features
        self.output_embeddings = output_embeddings 
        self.name = name

        # Hyper-parameters.
        # self.apply_relu_conv = apply_relu_conv # If True applies relu after each convolutional layer
        self.hidden_channels = hidden_channels # Sets dimension of hidden channels
        self.n_conv_layers = n_conv_layers     # Sets number of convolutional layers.
        self.n_linear_layers = n_linear_layers # Sets number of linear layers.

        # GCN and Linear layers employed by the model.
        self.GCN_layers = torch.nn.ModuleList()
        self.Linear_layers = torch.nn.ModuleList()
        self.BN_layers = torch.nn.ModuleList()

        # If no linear layers
        if self.n_linear_layers == 0:
            for i in range(self.n_conv_layers-1): 
                self.GCN_layers.append(GCNConv(input_features, hidden_channels))
                input_features = hidden_channels # From second layer we'll need this.
            # Final GCN layer
            self.GCN_layers.append(GCNConv(input_features, output_embeddings))

        # If there are some linear layers
        else:
            for i in range(self.n_conv_layers): 
                self.GCN_layers.append(GCNConv(input_features, hidden_channels))
                input_features = hidden_channels # From second layer we'll need this.
            # Linear layers
            for i in range(self.n_linear_layers - 1):
                self.Linear_layers.append(Linear(hidden_channels, hidden_channels))
            # Final linear layer
            self.Linear_layers.append(Linear(hidden_channels, output_embeddings))

        # Prepare batch normalization layers to apply after each convolutonal layer.
        for i in range(self.n_conv_layers-1):
            self.BN_layers.append(BatchNorm(hidden_channels))

        # Additional layers
        self.relu = torch.nn.ReLU()
        
        # Define the operation to be performed on the distances based on the the specification.
        if self.dist == 'L1':
            self.pdist = PairwiseDistance(p=1)
        elif self.dist == 'L2':
            self.pdist = PairwiseDistance(p=2)
        elif self.dist == 'cosine':
            self.pdist = CosineSimilarity()

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2): # Need a way to extract these from dataloader

        # 1. Obtain node embeddings for graph 1 and 2
        for i, gcn_layer in enumerate(self.GCN_layers):
            x1 = gcn_layer(x1, edge_index1)
            # x1 = self.conv_dropout(x1)
            x2 = gcn_layer(x2, edge_index2)
            # x2 = self.conv_dropout(x2)

            if i != (self.n_conv_layers-1): # Do not apply batch normalization or ReLu after the last convolutional layer.
                x1 = self.relu(x1)
                x2 = self.relu(x2)
                x1 = self.BN_layers[i](x1) 
                x2 = self.BN_layers[i](x2)
        
        # 2. Readout layer followed by Linear layers.
        x1 = global_mean_pool(x1, batch1)
        x1 = torch.nn.functional.dropout(x1, p=0.5)
        x2 = global_mean_pool(x2, batch2)
        x2 = torch.nn.functional.dropout(x2, p=0.5)

        for i, layer in enumerate(self.Linear_layers):
            x1 = layer(x1)
            x2 = layer(x2)
            if i != self.n_linear_layers: # Do not apply ReLu after the final linear layer
                x1 = self.relu(x1)
                x2 = self.relu(x2)

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