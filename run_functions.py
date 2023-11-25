import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader

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
        prediction = model(text_vecs, sen_lens)

        # Calculate the loss on the output of the model
        loss = criterion(prediction, labels)

        # Propagate the error back on the model (this means changing the initial weights in your model)
        # Calculate gradients on parameters that requries grad
        loss.backward()
        # Update the parameters
        optimizer.step()

        # We add batch-wise loss to the epoch-wise loss
        epoch_loss += loss.item()
    return (
        epoch_loss / len(iterator),
    )
