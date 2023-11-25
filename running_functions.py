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
    # We will calculate loss and accuracy epoch-wise based on average batch accuracy
    epoch_loss = 0

    # You always need to set your model to training mode
    # If you don't set your model to training mode the error won't propagate back to the weights
    model.train()

    # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
    for batch in iterator:
        # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
        optimizer.zero_grad()

        # This runs the forward function on your model (you don't need to call it directly)
        predictions = model(batch.x_1, batch.edge_index_1, batch.x_1_batch, 
                            batch.x_2, batch.edge_index_2, batch.x_2_batch)

        # Calculate the loss on the output of the model
        loss = criterion(predictions, batch.distance)

        # Propagate the error back on the model (this means changing the initial weights in your model)
        # Calculate gradients on parameters that requries grad
        loss.backward()
        # Update the parameters
        optimizer.step()

        # We add batch-wise loss to the epoch-wise loss
        epoch_loss += loss.item()
    return (epoch_loss / len(iterator))


def evaluate(model, iterator, criterion):

    epoch_loss = 0

    # On the validation dataset we don't want training so we need to set the model on evaluation mode
    model.eval()

    # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
    # This is needed when you only want to make predictions and use your model in inference mode!
    with torch.no_grad():

        # The remaining part is the same with the difference of not using the optimizer to backpropagation
        for batch in iterator:
            predictions = model(batch.x_1, batch.edge_index_1, batch.x_1_batch, 
                                batch.x_2, batch.edge_index_2, batch.x_2_batch)
            loss = criterion(predictions, batch.distance)

            epoch_loss += loss.item()

    # Return averaged loss on the whole epoch!
    return (epoch_loss / len(iterator))



def training_loop(model, train_iterator, optimizer, criterion, valid_iterator, epoch_number=150):
    # Set an EPOCH number!
    N_EPOCHS = epoch_number

    best_valid_loss = float("inf")

    # We loop forward on the epoch number
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        # Train the model on the training set using the dataloader
        train_loss = train(model, train_iterator, optimizer, criterion)
        # And validate your model on the validation set
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # If we find a better model, we save the weights so later we may want to reload it
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f}"
            )