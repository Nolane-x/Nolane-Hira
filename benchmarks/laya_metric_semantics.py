from __future__ import annotations
import math
import numpy as np

def macro_f1(gold, pred) -> float:
    g=np.asarray(gold); p=np.asarray(pred); vals=[]
    for c in sorted(set(g.tolist())|set(p.tolist())):
        tp=int(((p==c)&(g==c)).sum()); fp=int(((p==c)&(g!=c)).sum()); fn=int(((p!=c)&(g==c)).sum())
        vals.append(2*tp/max(1,2*tp+fp+fn))
    return float(np.mean(vals)) if vals else float("nan")

def ece(conf, correct, bins:int=15) -> float:
    conf=np.asarray(conf,float); correct=np.asarray(correct,float)
    if len(conf)==0:return float("nan")
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        sel=(conf>lo)&(conf<=hi)
        if sel.any():out += float(sel.mean()*abs(conf[sel].mean()-correct[sel].mean()))
    return out

def aurc(conf, correct) -> float:
    conf=np.asarray(conf,float); correct=np.asarray(correct,float)
    if len(conf)==0:return float("nan")
    order=np.argsort(-conf)
    return float((np.cumsum(1-correct[order])/np.arange(1,len(order)+1)).mean())

def hard_metrics(gold, probs) -> dict:
    P=np.asarray(probs,float); g=np.asarray(gold,int)
    pred=P.argmax(1); conf=P.max(1); corr=(pred==g).astype(float)
    onehot=np.eye(P.shape[1])[g]
    out={
      "n":int(len(g)), "accuracy":float(corr.mean()), "macro_f1":macro_f1(g,pred),
      "ece":ece(conf,corr), "brier":float(((P-onehot)**2).sum(1).mean()),
      "nll":float(np.mean([-math.log(max(float(P[i,y]),1e-12)) for i,y in enumerate(g)])),
      "aurc":aurc(conf,corr), "mean_confidence":float(conf.mean())
    }
    for cov in (.5,.8):
        k=max(1,int(len(conf)*cov)); out[f"acc_at_{int(cov*100)}_coverage"]=float(corr[np.argsort(-conf)[:k]].mean())
    return out

def soft_distribution_metrics(pred_probs, teacher_probs) -> dict:
    P=np.asarray(pred_probs,float); G=np.asarray(teacher_probs,float)
    P=P/np.maximum(P.sum(1,keepdims=True),1e-12); G=G/np.maximum(G.sum(1,keepdims=True),1e-12)
    ratio=G/np.clip(P,1e-12,None)
    return {
      "soft_acc":float((P*G).sum(1).mean()),
      "brier_soft":float(((P-G)**2).sum(1).mean()),
      "tv":float((.5*np.abs(P-G).sum(1)).mean()),
      "kl":float((G*np.log(np.clip(ratio,1e-12,1e4))).sum(1).mean())
    }

def ordinal_metrics(probs, gold_score) -> dict:
    P=np.asarray(probs,float); gold=np.asarray(gold_score,float)
    exp=(P*np.arange(P.shape[1])[None,:]).sum(1); err=np.abs(exp-gold)
    return {"score_mae":float(err.mean()),"within_1":float((err<=1.0).mean())}
