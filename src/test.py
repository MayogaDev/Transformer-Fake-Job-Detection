import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

def test_model(model, dataloader, device):
    """
    Tests the model on given data and returns accuracy and additional metrics.
    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): Device to test the model on.
    Returns:
        test_accuracy (float): Test accuracy.
        additional_metrics (tuple): Tuple containing precision, recall, and F1-score.
    """
    model.eval()
    correct_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct_predictions.double() / len(dataloader.dataset)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted')

    additional_metrics = (precision, recall, f1_score)
    return test_accuracy, additional_metrics
