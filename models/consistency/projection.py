import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

from transformers import GPT2Config, GPT2Model

class CaptionProjector(nn.Module):
    def __init__(self, num_layers, 
                    caption_emb_dim, 
                    emb_dim,  # emb dim of the caption hidden state
                    num_heads, # num heads in the transformer
                    aggr, # how to pool the features
                    dim_ff, # ff dim of the transformer
                 ):
        super().__init__()

        self.hidden_state_projector = None

        self.aggr = aggr

        if self.aggr == 'cls':
            # create an embedding for a cls token using nn.Embedding
            self.cls_token = nn.Embedding(1, emb_dim)
        
        if caption_emb_dim != emb_dim:
            # different dims, need to project
            print(f'Caption model emb dim: {caption_emb_dim}, will be projected emb dim: {emb_dim}')
            self.hidden_state_projector = nn.Sequential(
                nn.Linear(caption_emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
            )

        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                        dim_feedforward=dim_ff, 
                        batch_first=True) # B, seqlen, embdim in our outputs from GPT model!
        
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, hidden_states):
        if self.hidden_state_projector:
            hidden_states = self.hidden_state_projector(hidden_states)

        if self.aggr == 'cls':
            # add the cls token to the input and then take the first output
            # one for each input in hidden states
            cls_token = self.cls_token(torch.zeros(hidden_states.size(0), 1).long().to(hidden_states.device))
            # along the seq dim
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)
        
        # just pass the hidden states through the transformer
        # nobj, seqlen, embdim
        hidden_emb = self.transformer(hidden_states)

        # avg the embeddings across the sequence dimp
        if self.aggr == 'mean':
            emb_out = hidden_emb.mean(dim=1)
        elif self.aggr == 'max':
            emb_out = hidden_emb.max(dim=1)[0]
        elif self.aggr == 'cls':
            # take the first output of the cls token
            emb_out = hidden_emb[:, 0, :]
        elif self.aggr == 'first':
            # take the first output from the hidden states, no cls token was added
            emb_out = hidden_emb[:, 0, :]
        elif self.aggr is None:
            emb_out = hidden_emb

        return emb_out