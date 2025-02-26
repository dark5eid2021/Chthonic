import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a single transformer block.
class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (sequence_length, batch_size, embed_dim)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

# GPT-like model for ArgoCD logs.
class GPTArgoModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding layers for tokens and positions.
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        
        # Stack transformer blocks.
        self.layers = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.size()
        token_embeddings = self.token_embed(x)
        token_embeddings = token_embeddings + self.pos_embed[:seq_len, :]
        
        # Transformer expects (seq_len, batch_size, embed_dim)
        x = token_embeddings.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        
        # Convert back to (batch_size, seq_len, embed_dim) then output logits.
        x = x.transpose(0, 1)
        logits = self.head(x)
        return logits

if __name__ == "__main__":
    # Hyperparameters (tune these based on your ArgoCD logs)
    vocab_size = 6000       # Example: Adjust based on your tokenizer
    embed_dim = 256
    num_layers = 6
    num_heads = 8
    max_seq_len = 512       # Adjust according to your log sequence length
    
    # Create an instance of the model.
    model = GPTArgoModel(vocab_size, embed_dim, num_layers, num_heads, max_seq_len)
    
    # Dummy input: simulate a batch of tokenized ArgoCD logs.
    dummy_input = torch.randint(0, vocab_size, (2, max_seq_len))
    
    # Forward pass.
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (2, max_seq_len, vocab_size)
