import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        if batch.x_3 is None:
            predictions = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
            loss = criterion(predictions, batch.distance)
        else:
            a, p, n = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch,
                                batch.x_3.float(), batch.edge_index_3, batch.x_3_batch)
            loss = criterion(a, p, n, batch.margin)
              
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return (epoch_loss / len(iterator))


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            if batch.x_3 is None:
                predictions = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch)
                loss = criterion(predictions, batch.distance)
            else:
                a, p, n = model(batch.x_1.float(), batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2.float(), batch.edge_index_2, batch.x_2_batch,
                                batch.x_3.float(), batch.edge_index_3, batch.x_3_batch)
                loss = criterion(a, p, n, batch.margin)

            epoch_loss += loss.item()

    return (epoch_loss / len(iterator))



def training_loop(model, train_iterator, optimizer, criterion, valid_iterator, epoch_number=100, patience=-1, return_losses=False):
    """
    Performs training for the specified number of epochs.
    Then returns training and validation losses if required
    Implemented parameter patience to control automatic early stopping. If -1 it is same as epoch number so not on.
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
    

