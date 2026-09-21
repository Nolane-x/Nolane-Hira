"""Parameter-free sparse lexical/exemplar sidecar for SD-HIRA.

The sidecar is dynamic-schema compatible: it can compile option descriptions and optional
examples at request/schema-registration time. It is not a fixed classifier weight matrix.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib, math, re
from typing import Iterable

_TOKEN = re.compile(r"[a-z0-9]+")


def _stable_hash(text: str, dims: int) -> int:
    h=hashlib.blake2b(text.encode('utf-8'),digest_size=8).digest()
    return int.from_bytes(h,'little') % dims


def features(text: str, dims: int=32768) -> Counter[int]:
    s=text.lower()
    toks=_TOKEN.findall(s)
    out=Counter()
    for i,t in enumerate(toks):
        out[_stable_hash('u:'+t,dims)] += 1
        if i+1<len(toks): out[_stable_hash('b:'+t+'_'+toks[i+1],dims)] += 1
    z=' '+re.sub(r'[^a-z0-9]+',' ',s).strip()+' '
    for n in (3,4,5):
        for i in range(max(0,len(z)-n+1)):
            out[_stable_hash(f'c{n}:'+z[i:i+n],dims)] += 0.35
    return out


def normalize(v: Counter[int]) -> dict[int,float]:
    w={k:(1+math.log(max(1e-9,float(c)))) for k,c in v.items() if c>0}
    n=math.sqrt(sum(x*x for x in w.values())) or 1.0
    return {k:x/n for k,x in w.items()}


def merge(vectors: Iterable[dict[int,float]]) -> dict[int,float]:
    acc=Counter(); n=0
    for v in vectors:
        n+=1
        for k,x in v.items():acc[k]+=x
    if not n:return {}
    out={k:x/n for k,x in acc.items()}
    norm=math.sqrt(sum(x*x for x in out.values())) or 1.0
    return {k:x/norm for k,x in out.items()}


def cosine(a: dict[int,float], b: dict[int,float]) -> float:
    if len(a)>len(b):a,b=b,a
    return sum(v*b.get(k,0.0) for k,v in a.items())


@dataclass(frozen=True)
class SparseOption:
    option_id: str
    vector: dict[int,float]


class SparseSchema:
    def __init__(self, options: list[SparseOption], dims: int=32768):
        self.options=options; self.dims=dims

    @classmethod
    def compile(cls, criteria: dict[str,str], exemplars: dict[str,list[str]]|None=None, dims: int=32768):
        opts=[]; exemplars=exemplars or {}
        for oid,desc in criteria.items():
            vecs=[normalize(features(desc,dims))]
            for x in exemplars.get(oid,[]): vecs.append(normalize(features(x,dims)))
            opts.append(SparseOption(oid,merge(vecs)))
        return cls(opts,dims)

    def score(self, text: str) -> dict[str,float]:
        q=normalize(features(text,self.dims))
        return {o.option_id:cosine(q,o.vector) for o in self.options}
