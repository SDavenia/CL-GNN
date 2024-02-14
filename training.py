"""
This file contains the functions used for training the network for both CL and TL approaches.
"""
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
import time

def epoch_time(start_time, end_time):
    """
    Helper function to compute epoch time during training.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    """
    Trains the model for one epoch.
    """
    epoch_loss = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        try:
            # This will execute if the batch has x_3, i.e. we are using triplet loss approach
            a, p, n = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                            batch.x_2.float(), batch.edge_index_2, batch.x_2_batch,
                            batch.x_3.float(), batch.edge_index_3, batch.x_3_batch)
            loss = criterion(a, p, n, batch.margin)
        except AttributeError:
            # This will execute if the batch does not containg x_3, i.e. we are using the contrastive approach.
            predictions = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
            loss = criterion(predictions, batch.distance)
              
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return (epoch_loss / len(iterator))


def evaluate(model, iterator, criterion):
    """
    Evaluates the model for one epoch.
    """
    epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            try:
                # This will execute if the batch has x_3, i.e. we are using triplet loss approach
                a, p, n = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch,
                                batch.x_3.float(), batch.edge_index_3, batch.x_3_batch)
                loss = criterion(a, p, n, batch.margin)
            except AttributeError:
                # This will execute if the batch does not containg x_3, i.e. we are using the contrastive approach.
                predictions = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                    batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
                loss = criterion(predictions, batch.distance)

            epoch_loss += loss.item()

    return (epoch_loss / len(iterator))



def training_loop(model, train_iterator, optimizer, criterion, valid_iterator, epoch_number=100, patience=-1, return_losses=False):
    """
    Input:
        - model: The model to be trained.
        - train_iterator: The iterator for the training data.
        - optimizer: The optimizer to be used for training.
        - criterion: The loss function to be used.
        - valid_iterator: The iterator for the validation data.
        - epoch_number: The number of epochs to train the model.
        - patience: The number of epochs to wait for improvement in validation loss before stopping training. If -1 no early stopping.
        - return_losses: If True, returns the training and validation losses.
    """
    N_EPOCHS = epoch_number

    best_valid_loss = float("inf")
    train_losses = []
    validation_losses = []
    best_epoch = 0

    if patience == -1:
        patience = epoch_number

    no_improvement_count = 0

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
            
            print(f"Epoch: {epoch+1:02} | Time for 10 epochs: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f}"
            )
            start_time = time.time()
        
        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            if return_losses == True:
                print(f"Best epoch was {best_epoch}")
                return train_losses, validation_losses
            break

    if return_losses == True:
        print(f"Best epoch was {best_epoch}")
        return train_losses, validation_losses
    

