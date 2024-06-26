"""
This file contains the main model used for training.
"""
import torch
from torch.nn import Linear, Parameter, PairwiseDistance, CosineSimilarity
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.nn.norm import BatchNorm

class GCN_k_m(torch.nn.Module):
    """
    The base layer of the network consists of k graph convolutional layers followed by m linear layers (with k >= 1 and m >= 0) to obtain graph embeddings.
    conv1 -> ...         -> convk -> meanpool -> linear1 -> relu -> ... -> linearm -> embedding 

    For training the network considers two approaches:
    - contrastive approach: Given two graphs, the network computes the distance between their embeddings and penalizes according to some pre-specified distance.
    - triplet approach: Given three graphs (a, p, n) the network computed the distance between the embeddings of a and p, and the distance between the embeddings of a and n. It then penalizes if this difference is smaller than some pre-specified one. 
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, n_conv_layers, n_linear_layers, name, dist = 'L1'):
        """
        Input: 
            - input_features: The number of input features of the dataset
            - hidden_channels: The dimension of the hidden representation of the nodes/graphs.
            - output_embeddings: The dimension of the final embedding.
            - n_conv_layers: The number of GCN layers to be employed.
            - n_linear_layers: The number of linear layers to be employed.
            - name: The name of the model, used for saving.
            - dist: The distance to be computed on the output embedding when working with CL/TL approaches.
        """
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

    def forward_base_network(self, x, edge_index, batch):  # This is the forward pass for the base network. 
        # 1. Obtain node embeddings for graph 1 and 2
        for i, gcn_layer in enumerate(self.GCN_layers):
            x = gcn_layer(x, edge_index)
    
            if i != (self.n_conv_layers-1): # Do not apply batch normalization or ReLu after the last convolutional layer.
                x = self.relu(x)
                x = self.BN_layers[i](x) 
        
        # 2. Readout layer followed by Linear layers.
        x = global_mean_pool(x, batch)
        x = torch.nn.functional.dropout(x, p=0.5)

        for i, layer in enumerate(self.Linear_layers):
            x = layer(x)
            if i != self.n_linear_layers: # Do not apply ReLu after the final linear layer
                x = self.relu(x)
        return x

    def forward_contrastive_loss(self, x1, edge_index1, batch1, x2, edge_index2, batch2): # Forward pass when two inputs are given to the network
        x1 = self.forward_base_network(x1, edge_index1, batch1)
        x2 = self.forward_base_network(x2, edge_index2, batch2)

        # Compute the corresponding distance between the embeddings.
        if self.dist == 'cosine':
            vdist = 1 - self.pdist(x1, x2)
        else:
            vdist = self.pdist(x1, x2)
        return vdist

    def forward_triplet_loss(self, x1, edge_index1, batch1, x2, edge_index2, batch2, x3, edge_index3, batch3): # Forward pass when three inputs are given to the network
        x1 = self.forward_base_network(x1, edge_index1, batch1)
        x2 = self.forward_base_network(x2, edge_index2, batch2)
        x3 = self.forward_base_network(x3, edge_index3, batch3)

        return x1, x2, x3

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2, x3=None, edge_index3=None, batch3=None):
        """
        If the input has three entries it calls forward_triplet_loss, otherwise forward_contrastive_loss is used.
        """
        if x3 is None:
            # Call Contrastive loss forward.
            return self.forward_contrastive_loss(x1, edge_index1, batch1, x2, edge_index2, batch2)  
        else:
            # Call triplet loss forward.
            return self.forward_triplet_loss(x1, edge_index1, batch1, x2, edge_index2, batch2, x3, edge_index3, batch3)

    def save(self):
        """
        Saves the model state dictionary in models folder.
        """
        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)