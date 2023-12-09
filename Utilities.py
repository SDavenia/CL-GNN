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


########################### Plotting functions ###########################
def plot_matrix_runs(matrix_run1, matrix_run2, num_elements):
    """
    This function takes as input two matrices and plots them next to each other.
    """

    cmap = plt.cm.get_cmap('viridis')

    # Set up subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot matrix_run1
    plt1 = axs[0].imshow(matrix_run1[0:num_elements, 0:num_elements], cmap=cmap)
    axs[0].set_title('Matrix Run 1')

    # Plot matrix_run2
    plt2 = axs[1].imshow(matrix_run2[0:num_elements, 0:num_elements], cmap=cmap)
    axs[1].set_title('Matrix Run 2')

    # Create a single color bar for both subplots
    cbar = fig.colorbar(plt1, ax=axs, shrink=0.6, label='Color scale')

    # Update the color limits based on the data in both matrices
    plt1.set_clim(vmin=min(matrix_run1.min(), matrix_run2.min()), vmax=max(matrix_run1.max(), matrix_run2.max()))
    plt2.set_clim(vmin=min(matrix_run1.min(), matrix_run2.min()), vmax=max(matrix_run1.max(), matrix_run2.max()))

    plt.show()


def plot_results(y, predictions, subset = None):
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

    # Show the plot
    plt.show()

########################### DataLoader functions ###########################
def number_of_neighbours(graph):
    """
    Assuming that there are no disconnected nodes which do not appear in edge_index, this modifies
     the features of each node to only be one vector containing its number of neighbours
    """
    counts = graph.edge_index[0].unique(return_counts=True)[1].reshape(-1, 1)
    if len(counts) == graph.x.shape[0]:
        graph.x = counts
        return graph
    print(f"The input graph contains some unconnected node, this pre-transform can't work with it")
    raise TypeError()


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def prepare_dataloader_distance(hom_counts, dataset, device, batch_size = 32, dist = 'L1'):
    """
    Given a list of the homomorphism counts for the dataset and a specified distance,
    it returns the train validation and test dataloaders.
    """
    # Compute the distance matrix
    if dist not in ['cosine', 'L1', 'L2']:
        raise ValueError("Invalid value for 'dist'. Expected 'cosine', 'L1', or 'L2'.")
    elif dist == 'L1':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cityblock')
    elif dist == 'L2':
        dist_matrix = cdist(hom_counts, hom_counts, metric='euclidean')
    elif dist == 'cosine':
        dist_matrix = cdist(hom_counts, hom_counts, metric='cosine')

    # Split with 60, 20, 20 split (for now not randomized, to implement)
    train_dataset = dataset[:int(0.6*len(dataset) + 1)]
    val_dataset = dataset[int(0.6*len(dataset) + 1):int(0.8*len(dataset) + 1)]
    test_dataset = dataset[int(0.8*len(dataset) + 1):]

    train_data_list = []
    for ind1, graph1 in enumerate(train_dataset):
        for ind2, graph2 in enumerate(train_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            train_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = float(dist_matrix[ind1, ind2])).to(device))  
    
    val_data_list = []
    for ind1, graph1 in enumerate(val_dataset):
        for ind2, graph2 in enumerate(val_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            val_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = float(dist_matrix[ind1 + len(train_dataset), ind2 + len(train_dataset)])).to(device))    

    test_data_list = []
    for ind1, graph1 in enumerate(test_dataset):
        for ind2, graph2 in enumerate(test_dataset[ind1+1:]):
            ind2 += (ind1 + 1)
            test_data_list.append(PairData(x_1=graph1.x, edge_index_1=graph1.edge_index,
                                x_2=graph2.x, edge_index_2=graph2.edge_index,
                                distance = float(dist_matrix[ind1 + len(train_dataset) + len(test_dataset), ind2 + len(train_dataset) + len(test_dataset)])).to(device))    

    
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