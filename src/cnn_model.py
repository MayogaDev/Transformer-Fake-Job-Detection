import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    A CNN-based classifier for job posting fraud detection.
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding vectors.
        n_filters (int): Number of filters per filter size.
        filter_sizes (list): List of filter sizes for convolutional layers.
        output_dim (int): Number of output classes.
        dropout (float): Dropout rate.
    Returns:
        A CNNClassifier object.
    """
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids).unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # [batch_size, n_filters, seq_len - fs + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch_size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, n_filters * len(filter_sizes)]
        return self.fc(cat)