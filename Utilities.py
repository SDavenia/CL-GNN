"""
This file contains the necessary Utilities functions for preparing the DataLoaders and plotting results.
"""
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

import torch
from torch import Tensor
from torch.nn import MSELoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import random
import json
import numpy as np

########################### Plotting functions ###########################
def plot_matrix_runs(matrix_run1, matrix_run2, num_elements):
    """
    This function takes as input two matrices and plots them next to each other.
    The same colour scale is used in both
    """

    cmap = plt.cm.get_cmap('viridis')

    # Set up subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot matrix_run1
    plt1 = axs[0].imshow(matrix_run1[0:num_elements, 0:num_elements], cmap=cmap)
    axs[0].set_title('Run 1')

    # Plot matrix_run2
    plt2 = axs[1].imshow(matrix_run2[0:num_elements, 0:num_elements], cmap=cmap)
    axs[1].set_title('Run 2')

    # Create a single color bar for both subplots
    cbar = fig.colorbar(plt1, ax=axs, shrink=0.6, label='Color scale')

    # Update the color limits based on the data in both matrices
    plt1.set_clim(vmin=min(matrix_run1.min(), matrix_run2.min()), vmax=max(matrix_run1.max(), matrix_run2.max()))
    plt2.set_clim(vmin=min(matrix_run1.min(), matrix_run2.min()), vmax=max(matrix_run1.max(), matrix_run2.max()))

    plt.show()
    plt.close()


def save_plot_losses(train_losses, validation_losses, save_path):
    """
    Saves the train and validation losses along with a plot containing both in the directory specified by save_path
    """
    # Save plot of train and validation loss, exclude the first one otherwise loss unreadable
    save_img = save_path + '.png'
    plot_losses(train_losses[1:], validation_losses[1:], save_path=save_img)

    # Save train and validation losses
    save_train_loss = save_path + 'train_loss.txt'
    with open(save_train_loss, 'w') as file:
        for loss in train_losses:
            file.write(f'{loss}\n')

    save_validation_loss = save_path + 'validation_loss.txt'
    with open(save_validation_loss, 'w') as file:
        for loss in validation_losses:
            file.write(f'{loss}\n')

def plot_losses(train_losses, validation_losses, save_path=None):
    """
    Saves a plot of train and validation losses in the directory specified by save_path.
    """
    plt.plot(train_losses, label='train losses')
    plt.plot(validation_losses, label='validation losses')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title('Line Plot of train and validation loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_results(y, predictions, subset = None, save_path=None):
    """
    Plot of predicted vs actual results and saves the plot in the directory specified by save_path.
    If subset is specified only subset observations at random will be plotted.
    """
    y_array = y.cpu().numpy()
    predictions_array = predictions.cpu().numpy()

    y_array = y_array
    predictions_array = predictions_array

    if subset:
        random_ids = [random.randint(0, len(y)-1) for _ in range(subset)]
        y_array = y_array[random_ids]
        predictions_array = predictions_array[random_ids]

    # Plot the actual vs predicted values
    plt.scatter(y_array, predictions_array, label='Actual vs Predicted', s=0.5)
    plt.plot([min(y_array), max(y_array)], [min(y_array), max(y_array)], '--', color='red', label='Perfect Prediction')

    # Customize the plot
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

########################### DataLoader functions ###########################
class Add_ID_Count_Neighbours:
    """
    Adds a unique ID to each graph and retains only one feature which is the number of neighbours of each node.
    """
    def __init__(self):
        self.graph_index = 0

    def __call__(self, data):
        # Assign a unique ID (graph index) as an attribute to each graph
        # If a node is disconnected from the other 
        node_indices, node_neighbours = data.edge_index[0].unique(return_counts=True)
        counts = torch.zeros(data.x.shape[0], dtype=torch.int64)
        counts[node_indices] = node_neighbours
        counts = counts.reshape(-1, 1)

        data.x = counts
        data.id = torch.tensor([self.graph_index], dtype=torch.long)
        self.graph_index += 1
        return data

class PairData(Data):
    """
    Defines a base Data Object that takes as input two graphs.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class TripletData(Data):
    """
    Defines a base Data Object that takes as input three graphs.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        if key == 'edge_index_3':
            return self.x_3.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class CustomTripletMarginLoss(torch.nn.Module):
    """
    Custom Triplet Loss where margin is included.
    """
    p: float

    def __init__(self, p: float = 2.):
        super(CustomTripletMarginLoss, self).__init__()
        self.p = p

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, margin: Tensor) -> Tensor:
        d_ap = torch.norm(anchor - positive, dim=1, p=self.p)
        d_an = torch.norm(anchor - negative, dim=1, p=self.p)
        losses = torch.nn.functional.relu(d_ap - d_an + margin, inplace=False) # Remove all entries smaller than 0.
        return losses.mean()


def prepare_dataloader_contrastive(file_path, dataset, device, batch_size = 32, dist = 'L1', scaling = 'counts', scale_y=True):
    """
    Input:
        - path to .homson file as the output of homcount.
        - dataset corresponding to the specified homson file.
        - device: cpu or cuda depending on whether cuda is available.
        - batch_size: batch size for the torch loaders.
        - dist: metric to use to compute the distance between two vectors.
        - scaling: specifies whether it should use absolute counts, counts densities.
        - scale_y: specifies whether the distances should be scaled by the maximum distance in the training set.
    Output:
        - train_loader, val_loader, test_loader, test_dataset (the last needed for evaluation purposes).
    """
    # Compute the distance matrix
    if dist not in ['cosine', 'L1', 'L2']:
        raise ValueError("Invalid value for dist. Expected one of: cosine, L1, or L2.")
    if scaling not in ['counts', 'counts_density']:
        raise ValueError("Invalid value for scaling. Expected one of: counts, counts_density")
    
    # Read file
    with open(file_path) as f:
        data = json.load(f)
    
    if scaling == 'counts':
        hom_counts = [element['counts'] for element in data['data']]
    else:
        # Extract number of vertices of each graph in the dataset
        vertices = [data['data'][i]['vertices'] for i in range(len(data['data']))]

        # Extract number of vertices for the patterns used and number of patterns used
        pattern_sizes = data['pattern_sizes']
        p = len(pattern_sizes)

        # Compute the counts densities
        for i in range(len(data['data'])):
            n_vertices = vertices[i]
            den = [n_vertices**pattern_sizes[j] for j in range(p)]
            data['data'][i]['counts_densities'] = [data['data'][i]['counts'][j] / den[j] for j in range(p)]
        
        hom_counts = [element['counts_densities'] for element in data['data']]
        #assert all(entry <= 1 for list_ in hom_counts for entry in list_), "Densities should be <= 1"
        #assert all(entry >= 0 for list_ in hom_counts for entry in list_), "Densities should be <= 1"


    # Compute the distance matrix
    if dist == 'L1':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cityblock')
    elif dist == 'L2':
        dist_matrix = cdist(hom_counts, hom_counts, metric='euclidean')
    elif dist == 'cosine':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cosine')

    dist_matrix = dist_matrix.astype('float32')
    
    if scale_y:
        dist_matrix = np.sqrt(dist_matrix)
        
   
    # Split with 60, 20, 20 split
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.6*len(dataset) + 1)]
    val_dataset = dataset[int(0.6*len(dataset) + 1):int(0.8*len(dataset) + 1)]
    test_dataset = dataset[int(0.8*len(dataset) + 1):]

    max_train_dist=0
    train_data_list = []
    for ind1, graph1 in enumerate(train_dataset):
        for ind2, graph2 in enumerate(train_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = train_dataset[ind1].id.item()
            id2 = train_dataset[ind2].id.item()
            entry_dist = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))
            train_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index, id_1 = graph1.id,
                                            x_2=graph2.x, edge_index_2=graph2.edge_index, id_2 = graph2.id,
                                            distance = entry_dist).to(device)) 
            if entry_dist > max_train_dist:
                max_train_dist = entry_dist

    # Rescale the distances by dividing by the maximum distance in the training set.
    if scale_y:
        for pair in train_data_list:
            pair.distance = pair.distance / max_train_dist

    val_data_list = []
    for ind1, graph1 in enumerate(val_dataset):
        for ind2, graph2 in enumerate(val_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = val_dataset[ind1].id.item()
            id2 = val_dataset[ind2].id.item()
            val_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index, id_1 = graph1.id,
                                          x_2=graph2.x, edge_index_2=graph2.edge_index, id_2 = graph2.id,
                                          distance = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))).to(device)) 

    # Rescale the distances by dividing by the maximum distance in the training set.
    if scale_y:
        for pair in val_data_list:
            pair.distance = pair.distance / max_train_dist

    
    test_data_list = []
    for ind1, graph1 in enumerate(test_dataset):
        for ind2, graph2 in enumerate(test_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = test_dataset[ind1].id.item()
            id2 = test_dataset[ind2].id.item()
            test_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index, id_1 = graph1.id,
                                           x_2=graph2.x, edge_index_2=graph2.edge_index, id_2 = graph2.id,
                                           distance = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))).to(device)) 
                
    # Rescale the distances by dividing by the maximum distance in the training set.
    if scale_y:
        for pair in test_data_list:
            pair.distance = pair.distance / max_train_dist

    train_loader = DataLoader(train_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=False)

    return train_loader, val_loader, test_loader, test_dataset


def prepare_dataloader_triplet(dataset, dist_matrix, batch_size, k=10, device='cpu'):
    """
    Prepares dataloaders used for training/validation/test with the triplet loss. Note that the first two are triplet loaders, while for the 
      test_loader, we use a pair loader.
    Input:
        - dataset corresponding to the specified homson file.
        - device: cpu or cuda depending on whether cuda is available.
        - batch_size: batch size for the torch loaders.
        - dist_matrix: distance matrix for all pairs of graphs in the dataset.
    Output:
        - train_loader, val_loader, test_loader, test_dataset (the last needed for evaluation purposes).
    """
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.7*len(dataset))]
    val_dataset = dataset[int(0.7*len(dataset)):int(0.9*len(dataset))]
    test_dataset = dataset[int(0.9*len(dataset)):]

    train_loader = prepare_triplet_fold_loader(train_dataset, dist_matrix, k=k, batch_size=32, shuffle=True, device=device)
    val_loader = prepare_triplet_fold_loader(val_dataset, dist_matrix, k=k, batch_size=32, shuffle=False, device=device)
    test_loader = prepare_pair_loader(test_dataset, dist_matrix, batch_size=32, shuffle=False, device=device)
    return train_loader, val_loader, test_loader, test_dataset

def prepare_triplet_fold_loader(dataset_split, dist_matrix, batch_size, k=5, shuffle=True, device = 'cpu'):
    """
    Used to prepare the triplets that are used for training with the triplet loss.
    """
    data_list = []
    # IDs of graphs in the split
    ids = [int(graph['id']) for graph in dataset_split]
    for a_index, a_graph in enumerate(dataset_split):
        a_id = a_graph['id']

        fold_graphs = np.argsort(dist_matrix[a_id])
        fold_graphs = np.delete(fold_graphs, np.where(fold_graphs == int(a_id)))   # Remove the graph itself.
        fold_graphs = np.delete(fold_graphs, np.where(~np.isin(fold_graphs, ids))) # Remove the graphs not in the fold.

        closest = fold_graphs[0:k]  # Retain k closest graphs
        farthest = fold_graphs[-k:] # Retain k farthest away graphs

        # Create all triplets with anchor, one positive and one negative
        for p_id in closest:
            p_graph = dataset_split[ids.index(p_id)]
            for n_id in farthest:
                n_graph = dataset_split[ids.index(n_id)]
                # Obtain the margin to assign between the two 
                margin = torch.from_numpy(np.asarray(dist_matrix[a_id, n_id] - dist_matrix[a_id, p_id]))
                data_list.append(TripletData(x_1=a_graph.x, edge_index_1=a_graph.edge_index,
                                            x_2=p_graph.x, edge_index_2=p_graph.edge_index,
                                            x_3=n_graph.x, edge_index_3=n_graph.edge_index,
                                            margin=margin))

    loader = DataLoader(data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2', 'x_3'], shuffle=shuffle)
    return loader

def prepare_pair_loader(dataset_split, dist_matrix, batch_size, shuffle=False, device = 'cpu'):
    """
    Used to prepare the pairs for evaluation when working with triplet loss
    """
    data_list = []
    for ind1, graph1 in enumerate(dataset_split):
        for ind2, graph2 in enumerate(dataset_split[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = dataset_split[ind1].id.item()
            id2 = dataset_split[ind2].id.item()
            entry_dist = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))
            data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index, id_1 = graph1.id,
                                x_2=graph2.x, edge_index_2=graph2.edge_index, id_2 = graph2.id,
                                distance = entry_dist).to(device))
    loader = DataLoader(data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=shuffle)
    return loader   


def compute_distance_matrix(homomorphism_path, distance, scaling, scale_y=True):
    with open(homomorphism_path) as f:
        data = json.load(f)
    if scaling == 'counts':
        hom_counts = [element['counts'] for element in data['data']]
    else:
        raise NotImplemented('Scaling not implemented')
    
    if distance == 'L1':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cityblock')
    elif distance == 'L2':
        dist_matrix = cdist(hom_counts, hom_counts, metric='euclidean')
    elif distance == 'cosine':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cosine')

    dist_matrix = dist_matrix.astype('float32')

    if scale_y:
        dist_matrix = np.sqrt(dist_matrix)
        dist_matrix = dist_matrix / np.max(dist_matrix)
    return dist_matrix


########################### Evaluation functions ###########################
def score(model, loader, device = 'cpu'):
    """
    Given a (pre-trained) model and a dataloader, 
    It returns:
        - y: the true values of the regressor
        - predict: the predicted values according to the model
    """
     
    y = Tensor().to(device)
    predictions = Tensor().to(device)

    model.eval()
    with torch.no_grad():
        # The remaining part is the same with the difference of not using the optimizer to backpropagation
        for batch in loader:
            y = torch.cat((y, batch['distance']))
            preds = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                    batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
            predictions = torch.cat((predictions, preds))
    mse = MSELoss()
    print(f"MSE Loss: {mse(y, predictions)}")
    return y, predictions

def extract_k_closest_homdist(dataset_split, dist_matrix, k = 5):
    """
    Given a model and a dataloader, it returns for each graph in the dataset the k closest graphs according to the vectors of homomorphism counts.
    Input:
        - dataset_split: split of the dataset to be used.
        - dist_matrix: distance matrix between the vectors of homomorphism counts.
        - k: number of closest graphs to be extracted.
    Output:
        - closest_ids: dictionary with the closest k graphs for each graph in the dataset, based on the vectors of homomorphism counts.
    """
    closest_graphs = {}
    farthest_graphs = {}

    # IDs of graphs in the split
    ids = [int(graph['id']) for graph in dataset_split]
    for a_index, a_graph in enumerate(dataset_split):
        a_id = a_graph['id']

        fold_graphs = np.argsort(dist_matrix[a_id])
        fold_graphs = np.delete(fold_graphs, np.where(fold_graphs == int(a_id)))   # Remove the graph itself.
        fold_graphs = np.delete(fold_graphs, np.where(~np.isin(fold_graphs, ids))) # Remove the graphs not in the fold.

        closest = fold_graphs[0:k].tolist()  # Retain k closest graphs
        # farthest = fold_graphs[-k:].tolist() # Retain k farthest away graphs

        closest_graphs.update({int(a_id) : closest})
        # farthest_graphs.update({int(a_id) : farthest})

    return closest_graphs

def extract_k_closest_embedding(model, loader, k):
    """
    Given a model and a dataloader, it returns for each graph in the dataset the k closest graphs according to the embeddings obtained by the model.
    Input:
        - model: model to be evaluated.
        - loader: test dataloader to be used for evaluation.
        - k: number of closest graphs to be extracted.
    Output:
        - closest_ids: dictionary with the closest k graphs for each graph in the dataset, based on the embeddings obtained by the model.
    """

    l = sum([len(b) for b in loader])
    predicted_distances = np.array([], dtype=float)
    id1s = np.array([], dtype=int)
    id2s = np.array([], dtype=int)

    # Obtain model predicted distances between pairs.
    model.eval()
    with torch.no_grad():
        for batch in loader: 
            id1 = np.array([int(x) for x in batch.id_1])
            id2 = np.array([int(x) for x in batch.id_2])

            id1s = np.concatenate((id1s, id1))
            id2s = np.concatenate((id2s, id2))

            preds = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch,
                        batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
            preds = np.array([float(x) for x in preds])
            predicted_distances = np.concatenate((predicted_distances, preds))
    
    # Extract closest ones for each graph
    all_ids = np.concatenate((id1s, id2s))
    unique_ids = np.unique(all_ids)
    # Dictionary to store closest ids for each id
    closest_ids = {}
    k = 10

    for id_ in unique_ids:
        # Find indices where id appears in id1s or id2s and extract distances
        indices = np.where((id1s == id_) | (id2s == id_))[0]
        distances = predicted_distances[indices]
        combined = list(zip(distances, id1s[indices], id2s[indices]))
        combined.sort(key=lambda x: x[0])
        
        closest = combined[:k]
        closest_ids[id_] = [pair[1] if pair[1] != id_ else pair[2] for pair in closest]


    return closest_ids

def compute_jaccard(closest_graphs_original, closest_graphs_embedding):
    """
    Given the closest graphs based on the homomorphism counts and on the embeddings obtained, it returns the average Jaccard similarity between these sets.
    Input:
        - closest_graphs_original: dictionary with the closest graphs for each graph in the dataset, based on the vectors of homomorphism counts.
        - closest_graphs_embedding: dictionary with the closest graphs for each graph in the dataset, based on the embeddings obtained by the model.
    Output:
        - jaccard: average Jaccard similarity between the sets of closest graphs.
    """
    jaccard_similarities = []
    for x in closest_graphs_original.keys():
        original = set(closest_graphs_original[x])
        embedding = set(closest_graphs_embedding[x])
        jaccard_similarities.append(len(original.intersection(embedding)) / len(original.union(embedding)))
    return sum(jaccard_similarities)/len(jaccard_similarities)
