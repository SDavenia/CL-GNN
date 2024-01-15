import json
import os
import numpy as np
import pyhopper
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


from Utilities import score
from Utilities import plot_matrix_runs, plot_results, save_plot_losses
from Utilities import Add_ID_Count_Neighbours, PairData, prepare_dataloader_distance_scale

from training import training_loop

from models import GCN3, GCN3_MLP, GCN_k_m


def train_GCN_k_m(params, for_testing=False):
    #print(f"Trying out one")
    #print(f"Number of GCN_layers: {params['n_conv_layers']}")
    #print(f"Number of Linear layers: {params['n_linear_layers']}")
    #print(f"Relu After: {params['apply_relu_conv']}")
    #print(f"MLP Distance: {params['mlp_distance']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if params['dataset'] == 'MUTAG':
        dataset = TUDataset(root='/tmp/MUTAG_transformed', name='MUTAG', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)
    if params['dataset'] == 'ENZYMES':
        dataset = TUDataset(root='/tmp/ENZYMES_transformed', name='ENZYMES', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)

    hom_counts_path = 'data/homomorphism_counts/' + params['dataset'] + "_" + str(params['nhoms']) + ".homson"
    if not os.path.exists(hom_counts_path):
        raise FileNotFoundError(f"The file '{hom_counts_path}' was not found.")

    # Prepare dataloaders, where each element of the batch contains a pair of graphs and the specified distance obtained with homomorphism counts.
    torch.manual_seed(1231)
    train_loader, val_loader, test_loader = prepare_dataloader_distance_scale(hom_counts_path, dataset, batch_size=params['batch_size'], dist=params['distance'], device = device, scaling = params['hom_types'])

    model_details = ''
    if params['apply_relu_conv']:
        model_details += '_RELU'
    if params['mlp_distance']:
        model_details += '_mlp'
    
    name = params['dataset'] + "_" + str(params['nhoms']) + "_" + params['model_name'] + model_details + "_" + params['distance'] + "_" + params['hom_types'] + "_" + str(params['hidden_size']) + "_" + str(params['embedding_size']) + str(params['dropout']) + str(params['lr']) + str(params['batch_size'])
    if params['model_name'] == 'GCN3':
        model = GCN3(input_features=dataset.num_node_features, 
                    hidden_channels=params['hidden_size'], 
                    output_embeddings=params['embedding_size'], 
                    name=name, 
                    dist = params['distance']).to(device)
    elif params['model_name'] == 'GCN3_MLP':
        model = GCN3_MLP(input_features=dataset.num_node_features, 
                         hidden_channels=params['hidden_size'], 
                         output_embeddings=params['embedding_size'], 
                         name=name).to(device)
    elif params['model_name'] == 'GCN_k_m':
        model = GCN_k_m(input_features=dataset.num_node_features, 
                        hidden_channels=params['hidden_size'], 
                        output_embeddings=params['embedding_size'], 
                        n_conv_layers=params['n_conv_layers'], 
                        n_linear_layers=params['n_linear_layers'], 
                        p=params['dropout'], 
                        name=name, 
                        apply_relu_conv=params['apply_relu_conv'], 
                        dist=params['distance'], 
                        mlp_dist=params['mlp_distance']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.MSELoss().to(device)

    train_losses, validation_losses = training_loop(model, train_loader, optimizer, criterion, val_loader, epoch_number=params['epochs'], patience=params['patience'], return_losses=True)
    

    if for_testing:
        # Evaluate best model on the test set
        # Specify and save loss plots
        save_loss_directory = 'results/train_val_loss/' + name
        save_plot_losses(train_losses, validation_losses, save_loss_directory)
        # Load best model 
        if params['model_name'] == 'GCN3':
            model = GCN3(input_features=dataset.num_node_features, 
                        hidden_channels=params['hidden_size'], 
                        output_embeddings=params['embedding_size'], 
                        name=name, 
                        dist = params['distance']).to(device)
        elif params['model_name'] == 'GCN3_MLP':
            model = GCN3_MLP(input_features=dataset.num_node_features, 
                            hidden_channels=params['hidden_size'], 
                            output_embeddings=params['embedding_size'], 
                            name=name).to(device)
        elif params['model_name'] == 'GCN_k_m':
            model = GCN_k_m(input_features=dataset.num_node_features, 
                            hidden_channels=params['hidden_size'], 
                            output_embeddings=params['embedding_size'], 
                            n_conv_layers=params['n_conv_layers'], 
                            n_linear_layers=params['n_linear_layers'], 
                            p=params['dropout'], 
                            name=name, 
                            apply_relu_conv=params['apply_relu_conv'], 
                            dist=params['distance'], 
                            mlp_dist=params['mlp_distance']).to(device)
        saved_model_path = 'models/' + name + '.pt'
        model.load_state_dict(torch.load(saved_model_path))
        y, predictions = score(model, test_loader, device)
        predictions_path = 'results/actual_vs_predicted/' + name + '.png'
        plot_results(y, predictions, save_path = predictions_path)
    else:
        # Return the best (i.e. minimum) validation loss achieved by the model
        return min(validation_losses)


if __name__ == "__main__":
    search = pyhopper.Search(
        {   
            # Training hyper-parameters
            "lr": pyhopper.float(0.5, 0.05, precision=1, log=True),
            "batch_size": pyhopper.int(32, 128, power_of=2),

            # Model hyper-parameter
            "hidden_size": pyhopper.int(16, 128, power_of=2),
            "n_conv_layers": pyhopper.int(1, 3),
            "n_linear_layers": pyhopper.int(0, 2),
            "dropout": pyhopper.choice([0, 0.2, 0.4, 0.6]),
            "apply_relu_conv": pyhopper.choice([True, False]),
            "mlp_distance": pyhopper.choice([True, False]),

            # Not actual hyper-parameters: just fixed and needed for the function.
            "distance": pyhopper.choice(['L1']),
            "nhoms": pyhopper.choice([50]),
            "hom_types": pyhopper.choice(['counts']),
            "dataset": pyhopper.choice(['MUTAG']),
            "model_name": pyhopper.choice(['GCN_k_m']),
            "embedding_size": pyhopper.choice([1]),
            "epochs":pyhopper.choice([10]),
            "patience":pyhopper.choice([20])
        }
    )

    best_params = search.run(
        train_GCN_k_m,
        'minimize',
        runtime="1min",
        quiet=True,
        n_jobs=1,
    )

    train_GCN_k_m(best_params, for_testing=True)



