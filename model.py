import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x): 
        # input & output size: [batch_size, seq_length, input_dim]
        assert len(x.shape) == 3 and x.shape[2] == self.input_dim, \
            "SelfAttention expect input of shape [batch_size, seq_length, "\
            f"input_dim={self.input_dim}], got {list(x.shape)}"
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, dim_feedforward, num_layers, num_classes, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes*4),
            nn.ReLU(),
            nn.Linear(num_classes*4, num_classes),
            nn.Softmax(dim=1),
        )


    def forward(self, x):
        # input size: [batch_size, seq_length(vocab_index)]
        x = self.embedding(x)  # [batch_size, seq_length] -> [batch_size, seq_length, embed_dim]
        x = self.encoder(x)
        # avg pool
        x = torch.mean(x, dim=1)  # [batch_size, seq_length, embed_dim] -> [batch_size, embed_dim]
        x = self.classifier(x)  # [batch_size, embed_dim] -> [batch_size, num_classes]
        
        return x


if __name__ == '__main__':
    """print model info"""

    model = MyModel(
        vocab_size=20000,
        embed_dim=300,
        nhead=2,
        dim_feedforward=1024,
        num_layers=4,
        num_classes=2,
    )

    from torchinfo import summary
    summary(model)

    from utils import estimate_model_size
    estimate_model_size(model)
