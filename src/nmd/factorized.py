from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class FactorizedEmbeddingConfig:
    vocab_size:int=24000
    embedding_size:int=96
    hidden_size:int=256
    max_position_embeddings:int=512
    type_vocab_size:int=2
    pad_token_id:int=0
    layer_norm_eps:float=1e-12
    dropout:float=0.1

class FactorizedBertEmbeddings(nn.Module):
    """ALBERT-style factorized token embedding, but keeps a BERT-compatible hidden body."""
    def __init__(self,c:FactorizedEmbeddingConfig):
        super().__init__()
        self.word_embeddings=nn.Embedding(c.vocab_size,c.embedding_size,padding_idx=c.pad_token_id)
        self.embedding_projection=nn.Linear(c.embedding_size,c.hidden_size,bias=True)
        self.position_embeddings=nn.Embedding(c.max_position_embeddings,c.hidden_size)
        self.token_type_embeddings=nn.Embedding(c.type_vocab_size,c.hidden_size)
        self.LayerNorm=nn.LayerNorm(c.hidden_size,eps=c.layer_norm_eps)
        self.dropout=nn.Dropout(c.dropout)
        self.pad_token_id=c.pad_token_id
    def forward(self,input_ids,token_type_ids=None):
        B,L=input_ids.shape
        if token_type_ids is None: token_type_ids=torch.zeros_like(input_ids)
        pos=torch.arange(L,device=input_ids.device)[None,:].expand(B,L)
        w=self.embedding_projection(self.word_embeddings(input_ids))
        h=w+self.position_embeddings(pos)+self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(h))

def factorized_embedding_parameter_count(c:FactorizedEmbeddingConfig)->int:
    return (c.vocab_size*c.embedding_size + c.embedding_size*c.hidden_size + c.hidden_size
            + c.max_position_embeddings*c.hidden_size + c.type_vocab_size*c.hidden_size
            + 2*c.hidden_size)
