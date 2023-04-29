import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).pooler_output
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits


class ConvNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, conv_activation_layer=nn.ELU(), dense_activation_layer=nn.Sigmoid(), dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim))
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.conv_activation_layer = conv_activation_layer
        self.dense_activation_layer = dense_activation_layer

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.unsqueeze(1)
        x = [self.conv_activation_layer(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dense_activation_layer(x)
        return x



class EncoderTransformer(nn.Module):
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
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len).to(device=device) # (seq_len, )
        positions = positions.unsqueeze(0)                  # (1, seq_len)
        positions = positions.expand(batch_size, -1)        # (batch_size, seq_len)

        # Maintains (batch_size, seq_len, embedding_dim)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        # Add skip connections
        x = self.layernorm1(x + self.multihead_attention(x))
        x = self.layernorm2(x + self.feedforward(x))

        # Reduce dimensionality (aggregate all timesteps) - (batch_size, embedding_dim)
        x = torch.mean(x, dim=1)

        # Output Logits - (batch_size, num_classes)
        output = self.classifier(x)

        return output

'''
Based on how each key performs with each query across timesteps, 
we want to scale the value of that embedding dimension appropriately, 
for each timestep.
The score from:
(k_i * q_1), (k_i * q_2), (k_i * q_3), ..., (k_i * q_T)
is used to reduce embedding dimension i for each timestep
'''
class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size=32, dropout=0.2):
        super().__init__()

        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embedding_dim // head_size

    def forward(self, x):
        embedding_dim = x.size(2)
        
        # Key, Query, and Value Vectors - K, Q, V - (batch_size, seq_len, head_size)
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
    
        # Attention Weights - A - (batch_size, seq_len, seq_len)
        A = Q @ torch.transpose(K, -2, -1)
        A = A / (embedding_dim**0.5)
        A = F.softmax(A, -1)
        A = self.dropout(A)
        # Model Output - (batch_size, seq_len, head_size)
        output = A @ V                          

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.2):
        super().__init__()

        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, self.head_size, dropout=dropout) for _ in range(num_heads)
        ])
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output = torch.cat([f(x) for f in self.heads], dim=-1)
        output = self.dense(output)
        output = self.dropout(output)
        return output

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x