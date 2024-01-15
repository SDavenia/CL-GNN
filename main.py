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

from models import GCN3, GCN3_MLP, GCN_k_m

"""
Example call of this script:
python main.py --model_name GCN_k_m --dataset MUTAG --nhoms 50 --hidden_size 32 --embedding_size 50 --dropout 0.2 --n_conv_layers 2 --n_lin_layers 2 --apply_relu_conv True --mlp_distance True --distance L1 --hom_types counts --epochs 10 --batch_size 32 --patience 10 --lr 0.01
"""


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    # Specify the model, dataset and vector of homomorphisms.
    parser.add_argument('--model_name', type=str, required=True, choices=['GCN3', 'GCN3_MLP', 'GCN_k_m'],
                            help='Name of the model (choose from GCN3, GCN3_MLP, GCN_k_m)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MUTAG', 'ENZYMES'],
                            help='Name of the dataset (choose from MUTAG, ENZYMES)')
    parser.add_argument('--nhoms', type=int, required=True, help='Number of homomorphisms to compute the distance')

    # Specify parameters of the architecture, i.e. hidden size, embedding size and dropout probability.
    parser.add_argument('--hidden_size', type=int, default=64, help='Dimension of the hidden model size')
    parser.add_argument('--embedding_size', type=int, default=300, help='Dimension of the embedding')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--n_conv_layers', type=int, default=2, help='Number of GCN layers in the model')
    parser.add_argument('--n_lin_layers', type=int, default=1, help='Number of linear layers in the model (after GCN layers)')
    parser.add_argument('--apply_relu_conv', type=bool, default=False, help='If True, apply relu after each convolutional layer')
    parser.add_argument('--mlp_distance', type=bool, default=False, help='If True, distance is computed using an MLP on the difference between embeddings')

    # Specify parameters controlling the distance to be computed between the homcount vectors
    # (If models not ending with MLP are used, the same distance is also employed for the embeddings).  
    parser.add_argument('--distance', type=str, default='L1', choices=['L1', 'L2', 'cosine'],
                            help='Specify distance to use for embeddings (choose from L1, L2, cosine)')
    parser.add_argument('--hom_types', type=str, default='counts', choices=['counts', 'counts_density'],
                            help='Specify whether to use with homomorphism counts or counts densities')
    
    # Specify training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--patience', type=int, default=-1, help='Patience for automatic early stopping to occur. If -1 no early stopping.')
    parser.add_argument('--seed', type=int, default=1312, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Adam optimizer')
    
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    # Print summary of the model choices.
    print(f'Model Name: {args.model_name}')
    print(f'Hidden Size: {args.hidden_size}')
    print(f'Output Embedding Size: {args.embedding_size}')
    print(f'Dataset: {args.dataset}')
    print(f'Distance: {args.distance}, On homomorphism: {args.hom_types}')

    # Set up device for cuda if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset and initialize features to be the number of neighbours.
    if args.dataset == 'MUTAG':
        dataset = TUDataset(root='/tmp/MUTAG_transformed', name='MUTAG', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)
    if args.dataset == 'ENZYMES':
        dataset = TUDataset(root='/tmp/ENZYMES_transformed', name='ENZYMES', pre_transform=Add_ID_Count_Neighbours(), use_node_attr=True)

    # Read homomorphism counts path (should be of the form <dataset>_<number of homomorphisms>.homson)
    hom_counts_path = 'data/homomorphism_counts/' + args.dataset + "_" + str(args.nhoms) + ".homson"
    if not os.path.exists(hom_counts_path):
        raise FileNotFoundError(f"The file '{hom_counts_path}' was not found.")

    # Prepare dataloaders, where each element of the batch contains a pair of graphs and the specified distance obtained with homomorphism counts.
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = prepare_dataloader_distance_scale(hom_counts_path, dataset, batch_size=args.batch_size, dist=args.distance, device = device, scaling = args.hom_types)

    # The name of the model has form <dataset>_<nhoms>_<model_name>_k_m_(RELU)_(MLP)_<distance>_<hom_types>_<hidden_size>_<embedding_size>_<dropout>_<lr>_<batch_size>
    # The name of the model has form <dataset>_<nhoms>_<model_name>_<distance>_<hom_types>_<hidden_size>_<embedding_size>
    model_details = ''
    if args.apply_relu_conv:
        model_details += '_RELU'
    if args.mlp_distance:
        model_details += '_mlp'

    name = args.dataset + "_" + str(args.nhoms) + "_" + args.model_name + model_details + "_" + args.distance + "_" + args.hom_types + "_" + str(args.hidden_size) + "_" + str(args.embedding_size) + str(args.dropout) + str(args.lr) + str(args.batch_size)
    if args.model_name == 'GCN3':
        model = GCN3(input_features=dataset.num_node_features, 
                    hidden_channels=args.hidden_size, 
                    output_embeddings=args.embedding_size, 
                    name=name, 
                    dist = args.distance).to(device)
    elif args.model_name == 'GCN3_MLP':
        model = GCN3_MLP(input_features=dataset.num_node_features, 
                         hidden_channels=args.hidden_size, 
                         output_embeddings=args.embedding_size, 
                         name=name).to(device)
    elif args.model_name == 'GCN_k_m':
        model = GCN_k_m(input_features=dataset.num_node_features, 
                        hidden_channels=args.hidden_size, 
                        output_embeddings=args.embedding_size, 
                        n_conv_layers=args.n_conv_layers, 
                        n_linear_layers=args.n_lin_layers, 
                        p=args.dropout, 
                        name=name, 
                        apply_relu_conv=args.apply_relu_conv, 
                        dist=args.distance, 
                        mlp_dist=args.mlp_distance).to(device)
    #print(f"Model name: {name}")
    #print(model)
    # Prepare optimizer and criterion to be used during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss().to(device)
    
    # Train the model on the training set, saving the best model on the validation set and save some results.
    # print(f"Training:")
    train_losses, validation_losses = training_loop(model, train_loader, optimizer, criterion, val_loader, epoch_number=args.epochs, patience=args.patience, return_losses=True)

    # Specify the directory where you want to save the plots and results from the training.
    save_loss_directory = 'results/train_val_loss/' + name
    save_plot_losses(train_losses, validation_losses, save_loss_directory)
    
    # Load best model.
    if args.model_name == 'GCN3':
        model = GCN3(input_features=dataset.num_node_features, 
                    hidden_channels=args.hidden_size, 
                    output_embeddings=args.embedding_size, 
                    name=name, 
                    dist = args.distance).to(device)
    elif args.model_name == 'GCN3_MLP':
        model = GCN3_MLP(input_features=dataset.num_node_features, 
                         hidden_channels=args.hidden_size, 
                         output_embeddings=args.embedding_size, 
                         name=name).to(device)
    elif args.model_name == 'GCN_k_m':
        model = GCN_k_m(input_features=dataset.num_node_features, 
                        hidden_channels=args.hidden_size, 
                        output_embeddings=args.embedding_size, 
                        n_conv_layers=args.n_conv_layers, 
                        n_linear_layers=args.n_lin_layers, 
                        p=args.dropout, 
                        name=name, 
                        apply_relu_conv=args.apply_relu_conv, 
                        dist=args.distance, 
                        mlp_dist=args.mlp_distance).to(device)
    saved_model_path = 'models/' + name + '.pt'
    model.load_state_dict(torch.load(saved_model_path))

    # Obtain predictions on the test set and save the results.
    y, predictions = score(model, test_loader, device)
    predictions_path = 'results/actual_vs_predicted/' + name + '.png'
    plot_results(y, predictions, save_path = predictions_path)

if __name__ == "__main__":
    main()