import torch
from tqdm import tqdm

def evaluate(model, dataloader, loss_fn, device):
    """
    Evaluates the model on given data.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        loss_fn (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to evaluate the model on.
    Returns:
        epoch_loss (float): Average loss for the evaluation.
        epoch_accuracy (float): Accuracy for the evaluation.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions.double() / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy
