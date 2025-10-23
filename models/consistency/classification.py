import torch
from torch import nn

class CaptionClassifier(nn.Module):
    def __init__(self, num_layers, # num layers to use
                emb_dim,  # emb dim of the caption hidden state
                num_heads, # num heads in the transformer
                dim_ff, # ff dim of the transformer
                num_classes, # num classes to predict
                aggr, # how to pool the features
                caption_emb_dim): # input dim of the caption model
        super().__init__()

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

        # vanilla transformer decoder
        # project to num classes in the last layer
        # then use the avg of embeddings to predict the class
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                        dim_feedforward=dim_ff, 
                        batch_first=True) # B, seqlen, embdim in our outputs from GPT model!
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_dim, num_classes)
        print(f'>>>> Caption classification model with {num_classes} classes')

    def forward(self, hidden_states):
        # hidden_states: nobj, seqlen, embdim
        if self.hidden_state_projector:
            # project to emb dim of this model
            hidden_states = self.hidden_state_projector(hidden_states)

        if self.aggr == 'cls':
            # add the cls token to the input and then take the first output
            # one for each input in hidden states
            cls_token = self.cls_token(torch.zeros(hidden_states.size(0), 1).long().to(hidden_states.device))
            # along the seq dim
            hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        # pass the hidden states through the transformer
        # nobj, seqlen, embdim
        hidden_emb = self.transformer(hidden_states)

        # avg the embeddings across the sequence dimp
        if self.aggr == 'mean':
            aggr_emb = hidden_emb.mean(dim=1)
        elif self.aggr == 'max':
            aggr_emb = hidden_emb.max(dim=1)[0]
        elif self.aggr == 'cls':
            # take the first output of the cls token
            aggr_emb = hidden_emb[:, 0, :]
        elif self.aggr == 'first':
            # take the first output from the hidden states, no cls token was added
            aggr_emb = hidden_emb[:, 0, :]

        # predict the class
        # nobj, nclasses
        logits = self.classifier(aggr_emb)
        
        return logits

