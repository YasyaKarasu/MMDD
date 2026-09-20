"""Shared-head, randomly initialized pair/QET Teacher structure reference.

No checkpoint loading, raw dataset reading or Qwen extraction is performed here.
Test inputs are synthetic. Production must implement batched masked packing and
stage-specific data/loss contracts described in EXPERIMENT_SPEC.zh-CN.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F

KINDS=('table','text','image')

@dataclass
class ObjectInput:
    kind: Literal['table','text','image']
    z: Tensor
    content: Tensor

class QueryPool(nn.Module):
    def __init__(self, width: int, heads: int, slots: int):
        super().__init__()
        self.queries=nn.Parameter(torch.empty(slots,width))
        self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm=nn.LayerNorm(width)
    def forward(self,x:Tensor)->Tensor:
        q=self.queries.unsqueeze(0)
        pooled,_=self.attn(q,x.unsqueeze(0),x.unsqueeze(0),need_weights=False)
        return self.norm(q+pooled).squeeze(0)

class FreshPathTeacher(nn.Module):
    def __init__(self,input_dim:int=4096,width:int=512,heads:int=8,layers:int=3,ffn:int=2048,text_slots:int=16,image_slots:int=24,dropout:float=.1):
        super().__init__()
        self.width=width
        self.adapters=nn.ModuleDict({k:nn.Linear(input_dim,width) for k in KINDS})
        self.poolers=nn.ModuleDict({'text':QueryPool(width,heads,text_slots),'image':QueryPool(width,heads,image_slots)})
        self.globals=nn.ModuleDict({k:nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width)) for k in KINDS})
        self.roles=nn.Embedding(3,width)
        self.modality=nn.Embedding(3,width)
        self.table_kind=nn.Embedding(2,width)
        self.pair_kind=nn.Embedding(9,width)
        self.rel=nn.Parameter(torch.empty(width))
        self.sep=nn.Parameter(torch.empty(width))
        layer=nn.TransformerEncoderLayer(width,heads,ffn,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.relation=nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.global_relation=nn.Sequential(nn.Linear(5*width,width),nn.GELU(),nn.Linear(width,width))
        self.scoring_head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,1))
        self.reset_fresh()

    def reset_fresh(self):
        # Reset cloned Transformer layers independently, not just the prototype.
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None: nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m,nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=.02)
        for x in (self.rel,self.sep,self.poolers['text'].queries,self.poolers['image'].queries):
            nn.init.normal_(x,std=.02)

    def compress(self,o:ObjectInput)->tuple[Tensor,Tensor]:
        if o.kind not in KINDS or o.z.ndim!=1 or o.content.ndim!=2 or len(o.content)==0:
            raise ValueError('invalid object payload')
        x=self.adapters[o.kind](o.content)
        if o.kind=='table':
            kinds=torch.ones(len(x),dtype=torch.long,device=x.device); kinds[0]=0
            x=x+self.table_kind(kinds)
        else: x=self.poolers[o.kind](x)
        return x,self.globals[o.kind](o.z)

    def forward(self,a:ObjectInput,b:ObjectInput,e:ObjectInput|None=None)->Tensor:
        if e is not None and (a.kind!='table' or b.kind!='table' or e.kind=='table'):
            raise ValueError('nonempty bridge only for table-evidence-table')
        ca,ga=self.compress(a); cb,gb=self.compress(b)
        ia,ib=KINDS.index(a.kind),KINDS.index(b.kind)
        pair=self.pair_kind.weight[3*ia+ib]
        parts=[(self.rel+pair).unsqueeze(0),ca+self.modality.weight[ia]+self.roles.weight[0],self.sep.unsqueeze(0)]
        if e is not None:
            ce,ge=self.compress(e)
            parts += [torch.cat([ge.unsqueeze(0),ce],0)+self.modality.weight[KINDS.index(e.kind)]+self.roles.weight[2], self.sep.unsqueeze(0)]
        parts += [cb+self.modality.weight[ib]+self.roles.weight[1]]
        rel=self.relation(torch.cat(parts,0).unsqueeze(0))[0,0]
        glob=self.global_relation(torch.cat([ga,gb,ga*gb,(ga-gb).abs(),pair]))
        return self.scoring_head(rel+glob).squeeze(-1)
