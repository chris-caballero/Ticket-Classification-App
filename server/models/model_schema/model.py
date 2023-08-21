import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderTransformer(nn.Module):
    """
    Transformer-based encoder for text classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        seq_len (int): Maximum sequence length.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.

    Attributes:
        token_embedding (nn.Embedding): Token embedding layer.
        position_embedding (nn.Embedding): Positional embedding layer.
        multihead_attention (MultiHeadAttention): Multi-head self-attention layer.
        feedforward (FeedForwardLayer): Feedforward neural network layer.
        layernorm1 (nn.LayerNorm): Layer normalization for the first sub-layer.
        layernorm2 (nn.LayerNorm): Layer normalization for the second sub-layer.
        classifier (nn.Linear): Linear layer for classification.
    """

    def __init__(self, vocab_size, embedding_dim, seq_len, num_classes, num_heads=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        self.multihead_attention = MultiHeadAttention(num_heads, embedding_dim, dropout=0.1)
        self.feedforward = FeedForwardLayer(embedding_dim, dropout=0.1)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, input_ids):
        """
        Forward pass of the EncoderTransformer.

        Args:
            input_ids (torch.Tensor): Input token IDs.

        Returns:
            torch.Tensor: Logits for each class.
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len)               # (seq_len, )
        positions = positions.unsqueeze(0)              # (1, seq_len)
        positions = positions.expand(batch_size, -1)    # (batch_size, seq_len)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        x = self.layernorm1(x + self.multihead_attention(x))
        x = self.layernorm2(x + self.feedforward(x))
        x = torch.mean(x, dim=1)
        output = self.classifier(x)

        return output

class SelfAttentionHead(nn.Module):
    """
    Self-attention mechanism for a single head.

    Args:
        embedding_dim (int): Dimension of input embeddings.
        head_size (int): Size of the attention head.
        dropout (float): Dropout rate for regularization.

    Attributes:
        key (nn.Linear): Linear layer for key transformation.
        query (nn.Linear): Linear layer for query transformation.
        value (nn.Linear): Linear layer for value transformation.
        dropout (nn.Dropout): Dropout layer for regularization.
        head_dim (int): Dimension of each head.
    """

    def __init__(self, embedding_dim, head_size=32, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embedding_dim // head_size

    def forward(self, x):
        """
        Forward pass of the SelfAttentionHead.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of self-attention mechanism.
        """
        embedding_dim = x.size(2)

        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        A = Q @ torch.transpose(K, -2, -1)
        A = A / (embedding_dim**0.5)
        A = F.softmax(A, -1)
        A = self.dropout(A)
        output = A @ V

        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        num_heads (int): Number of attention heads.
        embedding_dim (int): Dimension of input embeddings.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, num_heads, embedding_dim, dropout=0.2):
        super().__init__()

        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, self.head_size, dropout=dropout) for _ in range(num_heads)
        ])
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of the MultiHeadAttention.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of multi-head self-attention mechanism.
        """
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dense(output)
        output = self.dropout(output)
        return output

class FeedForwardLayer(nn.Module):
    """
    Feedforward neural network layer.

    Args:
        embedding_dim (int): Dimension of input embeddings.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the FeedForwardLayer.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the feedforward layer.
        """
        x = self.mlp(x)
        return x
