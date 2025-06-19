import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Tokenization using Hugging Face's Transformers library
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Set the maximum sequence length
max_len = 128 

# Custom Dataset class for PyTorch
class JobPostingDataset(Dataset):
    """
    A custom Dataset class for loading and processing job posting data.
    Args:
        texts (pd.Series): A series of job posting texts.
        labels (pd.Series): A series of corresponding labels, where 0 represents non-fraudulent and 1 represents fraudulent postings.
        tokenizer (BertTokenizer): A tokenizer to process the texts.
        max_len (int): The maximum sequence length for tokenized texts.
    Returns:
        A JobPostingDataset object.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }