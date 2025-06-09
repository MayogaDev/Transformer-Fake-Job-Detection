import torch.nn as nn
from transformers import BertModel

# Define the PyTorch model (using BERT)
class JobPostingClassifier(nn.Module):
    """
    A custom classifier model for job posting fraud detection using BERT.
    Args:
        n_classes (int): The number of classes to predict.
    Returns:
        A JobPostingClassifier object.
    """
    def __init__(self, n_classes):
        super(JobPostingClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.out(output)