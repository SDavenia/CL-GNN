"""
This file contains the necessary Utilities functions
"""
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import torch
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

def plot_matrix_runs_different_scale(matrix_run1, matrix_run2, num_elements):
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
    cbar1 = plt.colorbar(plt1, ax=axs[0])  # Add colorbar for matrix_run1

    # Plot matrix_run2
    plt2 = axs[1].imshow(matrix_run2[0:num_elements, 0:num_elements], cmap=cmap)
    axs[1].set_title('Run 2')
    cbar2 = plt.colorbar(plt2, ax=axs[1])  # Add colorbar for matrix_run2

    # Adjust layout to make room for colorbars
    plt.tight_layout()
    plt.show()
    plt.close()


def save_plot_losses(train_losses, validation_losses, save_path):
    # Save plot of train and validation loss
    save_img = save_path + '.png'
    plot_losses(train_losses, validation_losses, save_path=save_img)

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
    Plot of predicted vs actual results.
    If subset is specified only subset observations at random will be plotted.
    """
    y_array = y.numpy()
    predictions_array = predictions.numpy()

    y_array = y_array
    predictions_array = predictions_array

    if subset:
        random_ids = [random.randint(0, len(y)-1) for _ in range(subset)]
        y_array = y_array[random_ids]
        predictions_array = predictions_array[random_ids]

    # Plot the actual vs predicted values
    plt.scatter(y_array, predictions_array, label='Actual vs Predicted')
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
    def __init__(self):
        self.graph_index = 0

    def __call__(self, data):
        # Assign a unique ID (graph index) as an attribute to each graph
        counts = data.edge_index[0].unique(return_counts=True)[1].reshape(-1, 1)
        if len(counts) == data.x.shape[0]:
            data.x = counts
            data.id = torch.tensor([self.graph_index], dtype=torch.long)
            self.graph_index += 1
            return data
        print(f"The input graph contains some unconnected node, this pre-transform can't work with it")
        raise TypeError()


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def prepare_dataloader_distance_scale(file_path, dataset, device, batch_size = 32, dist = 'L1', scaling = 'counts'):
    """
    Input:
        - path to .homson file as the output of homcount.
        - dataset corresponding to the specified homson file.
        - device: cpu or cuda depending on whether cuda is available.
        - batch_size: batch size for the torch loaders.
        - dist: metric to use to compute the distance between two vectors.
        - scaling: specifies whether it should use absolute counts, counts densities, or rescaled counts densities.
    """
    # Compute the distance matrix
    if dist not in ['cosine', 'L1', 'L2']:
        raise ValueError("Invalid value for dist. Expected one of: cosine, L1, or L2.")
    if scaling not in ['counts', 'counts_density', 'counts_density_rescaled']:
        raise ValueError("Invalid value for scaling. Expected one of: counts, counts_density or counts_density_rescaled")
    
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

        # Compute the counts densitied
        for i in range(len(data['data'])):
            n_vertices = vertices[i]
            den = [n_vertices**pattern_sizes[j] for j in range(p)]
            data['data'][i]['counts_densities'] = [data['data'][i]['counts'][j] / den[j] for j in range(p)]
        
        hom_counts = [element['counts_densities'] for element in data['data']]
        assert all(entry <= 1 for list_ in hom_counts for entry in list_), "Densities should be <= 1"
        assert all(entry >= 0 for list_ in hom_counts for entry in list_), "Densities should be <= 1"


    # Compute the distance matrix
    if dist == 'L1':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cityblock')
    elif dist == 'L2':
        dist_matrix = cdist(hom_counts, hom_counts, metric='euclidean')
    elif dist == 'cosine':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cosine')

    dist_matrix = dist_matrix.astype('float32')
    
    # Rescale dividing by the number of homomorphism counts extracted if necessary
    if scaling == 'counts_density_rescaled' and dist != 'cosine':  # since for cosine it makes no sense to rescale since always in [0,2]
        if dist == 'L1':
            dist_matrix = dist_matrix / p
        else:
            dist_matrix = dist_matrix / np.sqrt(p)
        assert all(entry <= 1  and entry >= 0 for row in dist_matrix for entry in row), f"Not all entries in dist_matrix are valid"

   
    # Split with 60, 20, 20 split
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.6*len(dataset) + 1)]
    val_dataset = dataset[int(0.6*len(dataset) + 1):int(0.8*len(dataset) + 1)]
    test_dataset = dataset[int(0.8*len(dataset) + 1):]

    train_data_list = []
    for ind1, graph1 in enumerate(train_dataset):
        for ind2, graph2 in enumerate(train_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = train_dataset[ind1].id.item()
            id2 = train_dataset[ind2].id.item()
            train_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))).to(device)) 

    val_data_list = []
    for ind1, graph1 in enumerate(val_dataset):
        for ind2, graph2 in enumerate(val_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = val_dataset[ind1].id.item()
            id2 = val_dataset[ind2].id.item()
            val_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))).to(device)) 

    test_data_list = []
    for ind1, graph1 in enumerate(test_dataset):
        for ind2, graph2 in enumerate(test_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            id1 = test_dataset[ind1].id.item()
            id2 = test_dataset[ind2].id.item()
            test_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = torch.from_numpy(np.asarray(dist_matrix[id1, id2]))).to(device)) 

    train_loader = DataLoader(train_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, follow_batch=['x_1', 'x_2'], shuffle=False)

    return train_loader, val_loader, test_loader



########################### Evaluation functions ###########################
def score(model, loader):
    """
    Given a (pre-trained) model and a dataloader, 
    It returns:
        - y: the true values of the regressor
        - predict: the predicted values according to the model
    """
        
    y = torch.Tensor()
    predictions = torch.Tensor()

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