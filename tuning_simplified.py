import json
import os
import numpy as np
import pyhopper
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


from Utilities import score
from Utilities import plot_matrix_runs, plot_results, save_plot_losses
from Utilities import Add_ID_Count_Neighbours, PairData, prepare_dataloader_distance_scale

from training import train, evaluate, epoch_time

from models import GCN_k_m

def training_loop_tuning(model, train_iterator, optimizer, criterion, valid_iterator, epoch_number=1, patience=-1, return_losses=False):
    """
    Performs training for the specified number of epochs.
    Then returns training and validation losses if required
    Implemented parameter patience to control automatic early stopping. If -1 it is same as epoch number so not on.
    """
    # print(f"Entering training for {epoch_number} epochs")
    N_EPOCHS = epoch_number

    best_valid_loss = float("inf")
    train_losses = []
    validation_losses = []
    best_epoch = 0

    if patience == -1:
        patience = epoch_number

    no_improvement_count = 0
    initial_time = time.time() # To stop earlier if it's taking too long
    start_time = time.time()
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)

        train_losses.append(train_loss)
        validation_losses.append(valid_loss)

        # Save model with best validation loss
        if valid_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss
            model.save() # Save model in models folder
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if (epoch+1) % 10 == 0:
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            #print(f"Epoch: {epoch+1:02} | Time for 10 epochs: {epoch_mins}m {epoch_secs}s")
            #print(
            #    f"\tTrain Loss: {train_loss:.3f}"
            #)
            #print(
            #    f"\t Val. Loss: {valid_loss:.3f}"
            #)
            start_time = time.time()
        # Stop training with current hyper-parameters if it is taking longer than 40 minutes.
        if time.time() - initial_time > 60*40:
            print(f"Interrupting training at epoch {epoch}, taking longer than 40 minutes")
            break

        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            if return_losses == True:
                print(f"Best epoch was {best_epoch}")
                return train_losses, validation_losses
            break

    if return_losses == True:
        # print(f"Best epoch was {best_epoch}")
        return train_losses, validation_losses


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    # Specify the model, dataset and vector of homomorphisms.
    parser.add_argument('--model_name', type=str, required=True, choices=['GCN_k_m'],
                            help='Name of the model (choose from GCN3, GCN3_MLP, GCN_k_m)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MUTAG', 'ENZYMES'],
                            help='Name of the dataset (choose from MUTAG, ENZYMES)')
    parser.add_argument('--nhoms', type=int, required=True, help='Number of homomorphisms to compute the distance')

    
    # Specify parameters controlling the distance to be computed between the homcount vectors
    # (If models not ending with MLP are used, the same distance is also employed for the embeddings).  
    parser.add_argument('--distance', type=str, default='L1', choices=['L1', 'L2', 'cosine'],
                            help='Specify distance to use for embeddings (choose from L1, L2, cosine)')
    parser.add_argument('--hom_types', type=str, default='counts', choices=['counts', 'counts_density'],
                            help='Specify whether to use with homomorphism counts or counts densities')
    
    # Specify training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for automatic early stopping to occur. If -1 no early stopping.') 

    parser.add_argument('--seed', type=int, default=20224, help='Seed for random generation')
    return parser.parse_args()



def train_GCN_k_m(params, for_testing=False):
    start_time = time.time()
    #print(f"Trying out one")
    #print(f"Number of GCN_layers: {params['n_conv_layers']}")
    #print(f"Number of Linear layers: {params['n_linear_layers']}")
    #print(f"Relu After: {params['apply_relu_conv']}")
    #print(f"MLP Distance: {params['mlp_distance']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    hom_counts_path = 'data/homomorphism_counts/' + params['dataset'] + "_" + str(params['nhoms']) + ".homson"
    if not os.path.exists(hom_counts_path):
        raise FileNotFoundError(f"The file '{hom_counts_path}' was not found.")

    # Prepare dataloaders, where each element of the batch contains a pair of graphs and the specified distance obtained with homomorphism counts.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_loader, val_loader, test_loader = prepare_dataloader_distance_scale(hom_counts_path, dataset, batch_size=params['batch_size'], dist=params['distance'], device = device, scaling = params['hom_types'], scale_y=True)
   
    if params['model_name'] == 'GCN_k_m':
        name = args.dataset + "_" + str(params['nhoms']) + "_GCN_" + str(params['n_conv_layers']) + "_" + str(params['n_linear_layers']) + "_" + params['distance'] + "_" + params['hom_types'] + "_" + str(params['hidden_size']) + "_" + str(params['embedding_size']) + "_"  + str(params['lr']) + "_" + str(params['batch_size'])
        model = GCN_k_m(input_features=dataset.num_node_features, 
                        hidden_channels=params['hidden_size'], 
                        output_embeddings=params['embedding_size'], 
                        n_conv_layers=params['n_conv_layers'], 
                        n_linear_layers=params['n_linear_layers'], 
                        name=name, 
                        dist=params['distance']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.MSELoss().to(device)

    train_losses, validation_losses = training_loop_tuning(model, train_loader, optimizer, criterion, val_loader, epoch_number=params['epochs'], patience=params['patience'], return_losses=True)
    

    return min(validation_losses)


if __name__ == "__main__":
    args = parse_command_line_arguments()
    if args.dataset == 'MUTAG':
        dataset = TUDataset(root='MUTAG_transformed', name='MUTAG', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)
    if args.dataset == 'ENZYMES':
        dataset = TUDataset(root='ENZYMES_transformed', name='ENZYMES', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)
    print(f"Successfully loaded dataset")
    search = pyhopper.Search(
        {   
            # Model hyper-parameter
            "hidden_size": pyhopper.choice([16, 32, 64]),
            "n_conv_layers": pyhopper.int(1, 3),
            "n_linear_layers": pyhopper.int(0, 2),

            # Training hyper-parameters
            "lr": pyhopper.float(0.5, 0.05, precision=1, log=True),
            "batch_size": pyhopper.choice([32, 64]),

            # Not actual hyper-parameters: just fixed and needed for the function.
            "distance": pyhopper.choice([args.distance]),
            "nhoms": pyhopper.choice([args.nhoms]),
            "hom_types": pyhopper.choice([args.hom_types]),
            "dataset": pyhopper.choice([args.dataset]),
            "model_name": pyhopper.choice([args.model_name]),
            "embedding_size": pyhopper.choice([args.nhoms]),
            "epochs":pyhopper.choice([args.epochs]),
            "patience":pyhopper.choice([args.patience])
        }
    )

    best_params = search.run(
        train_GCN_k_m,
        'minimize',
        runtime="1h 20min",
        quiet=True,
        n_jobs=-1,
    )
    print(best_params)

    # train_GCN_k_m(best_params, for_testing=True)



