import json
import os
import numpy as np
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

from models import GCN_pairs_distance

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['GCN3'],
                            help='Name of the model (choose from GCN3)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MUTAG', 'ENZYMES'],
                            help='Name of the dataset (choose from MUTAG, ENZYMES)')
    parser.add_argument('--nhoms', type=int, required=True, help='Number of homomorphisms to compute the distance')
    parser.add_argument('--hidden_size', type=int, default=64, help='Dimension of the hidden model size')
    parser.add_argument('--embedding_size', type=int, default=300, help='Dimension of the embedding')
    parser.add_argument('--distance', type=str, default='L1', choices=['L1', 'L2', 'cosine'],
                            help='Specify distance to use for embeddings (choose from L1, L2, cosine)')
    parser.add_argument('--distance_scaling', type=str, default='counts', choices=['counts', 'counts_density', 'counts_density_rescaled'],
                            help='Specify scaling to use for the distance (choose from counts, counts_density, counts_density_rescaled])')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=1312, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Adam optimizer')
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    print(f'Model Name: {args.model_name}')
    print(f'Hidden Size: {args.hidden_size}')
    print(f'Output Embedding Size: {args.embedding_size}')
    print(f'Dataset: {args.dataset}')
    print(f'Distance: {args.distance}, Scaling: {args.distance_scaling}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'MUTAG':
        dataset = TUDataset(root='/tmp/MUTAG_transformed', name='MUTAG', pre_transform=Add_ID_Count_Neighbours())
    if args.dataset == 'ENZYMES':
        dataset = TUDataset(root='/tmp/ENZYMES_transformed', name='ENZYMES', pre_transform=Add_ID_Count_Neighbours())
    
    torch.manual_seed(args.seed)
    hom_counts_path = 'data/homomorphism_counts/' + args.dataset + "_" + str(args.nhoms) + ".homson"
    if not os.path.exists(hom_counts_path):
        raise FileNotFoundError(f"The file '{hom_counts_path}' was not found.")

    train_loader, val_loader, test_loader = prepare_dataloader_distance_scale(hom_counts_path, dataset, batch_size=32, dist=args.distance, device = device, scaling = args.distance_scaling)

    name = args.dataset + "_" + str(args.nhoms) + "_" + args.model_name + "_" + args.distance + "_" + args.distance_scaling + "_" + str(args.hidden_size) + "_" + str(args.embedding_size)
    if args.model_name == 'GCN3':
        model = GCN_pairs_distance(input_features=dataset.num_node_features, hidden_channels=args.hidden_size, output_embeddings=args.embedding_size, name=name, dist = args.distance).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss().to(device)
    
    # Perform training and obtain plot for train and validation loss
    #  save training and validation and also the plot
    train_losses, validation_losses = training_loop(model, train_loader, optimizer, criterion, val_loader, epoch_number=args.epochs, return_losses=True)

    # Specify the directory where you want to save the plot and text files
    save_loss_directory = 'results/train_val_loss/' + name
    save_plot_losses(train_losses, validation_losses, save_loss_directory)
    
    # Load best model and obtain predictions
    if args.model_name == 'GCN3':
        model = GCN_pairs_distance(input_features=dataset.num_node_features, hidden_channels=args.hidden_size, output_embeddings=args.embedding_size, name=name, dist = args.distance).to(device)
        
    saved_model_path = 'models/' + name + '.pt'
    model.load_state_dict(torch.load(saved_model_path))
    y, predictions = score(model, test_loader, device)
    predictions_path = 'results/actual_vs_predicted/' + name + '.png'
    plot_results(y, predictions, save_path = predictions_path)


if __name__ == "__main__":
    main()