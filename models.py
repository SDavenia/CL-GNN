"""
This file contains the implementation of the different classes considered.
"""
import torch
from torch.nn import Linear, Parameter, PairwiseDistance, CosineSimilarity
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv

"""class GCN_pairs(torch.nn.Module):

    Takes as input a pair of graphs which are both fed through 3 convolutional layers each followed by an activation function
    3 graph convolutional layers (Welling) that share parameters

    def __init__(self, input_features, hidden_channels, output_embeddings, name):
        super(GCN_pairs, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_embeddings)
        self.pdist = PairwiseDistance(p=1)
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
        x1 = torch.nn.functional.dropout(x1, p=0.5, training=self.training)
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
        x2 = torch.nn.functional.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin(x2)

        dist = self.pdist(x1, x2) # p = 1 is the manhattan distance
        return dist
    
    def save(self):
        Saves the model state dictionary in models folder

        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)"""



class GCN_pairs_distance(torch.nn.Module):
    """
    Takes as input a pair of graphs which are both fed through 3 convolutional layers each followed by an activation function
    3 graph convolutional layers (Welling) that share parameters.
    Specify the distance matrix as one of cosine, L1, L2
    """
    def __init__(self, input_features, hidden_channels, output_embeddings, name, dist = 'L2'):
        super(GCN_pairs_distance, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_embeddings)
        self.name = name
        
        if dist not in ['cosine', 'L1', 'L2']:
            raise ValueError("Invalid value for 'dist'. Expected 'cosine', 'L1', or 'L2'.")
        elif dist == 'L1':
            self.pdist = PairwiseDistance(p=1)
        elif dist == 'L2':
            self.pdist = PairwiseDistance(p=2)
        elif dist == 'cosine':
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
        x1 = torch.nn.functional.dropout(x1, p=0.5, training=self.training)
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
        x2 = torch.nn.functional.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin(x2)

        dist = self.pdist(x1, x2)
        return dist
    
    def save(self):
        """
        Saves the model state dictionary in models folder
        """
        path = 'models/' + self.name + '.pt'
        torch.save(self.state_dict(), path)