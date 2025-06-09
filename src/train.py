import torch
from tqdm import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Trains the model for one epoch.
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (nn.Module): Loss function used for training.
        optimizer (optim.Optimizer): Optimizer used for training.
        device (torch.device): Device to train the model on.
    Returns:
        epoch_loss (float): Average loss for the epoch.
        epoch_accuracy (float): Accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions.double() / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy
