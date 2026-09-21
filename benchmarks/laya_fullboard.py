from __future__ import annotations
import argparse, json
from pathlib import Path

HIGHER={"accuracy","macro_f1","acc_at_50_coverage","acc_at_80_coverage","soft_acc","within_1"}
LOWER={"ece","brier","nll","aurc","brier_soft","tv","kl","score_mae"}
METRICS=HIGHER|LOWER

def flatten_model(m:dict)->dict:
    out={}
    cal=m.get("calibrated") if isinstance(m,dict) else None
    if isinstance(cal,dict):
        out.update({k:v for k,v in cal.items() if k in METRICS and isinstance(v,(int,float))})
    if isinstance(m,dict):
        for k,v in m.items():
            if k in METRICS and isinstance(v,(int,float)) and k not in out: out[k]=v
    return out

def best_laya_t4(src:dict)->dict:
    out={}
    for suite,models in src.get("suites",{}).items():
        if not isinstance(models,dict): continue
        vals={}
        for model,m in models.items():
            for metric,v in flatten_model(m).items(): vals.setdefault(metric,[]).append((float(v),model))
        for metric,items in vals.items():
            direction="higher" if metric in HIGHER else "lower"
            best=max(items) if direction=="higher" else min(items)
            out[f"t4::{suite}::{metric}"]={"target":best[0],"model":best[1],"direction":direction}
    return out

def best_laya_cpu51(src:dict)->dict:
    out={}; models=src.get("part_a",{}).get("by_model",{}); languages=set()
    for m in models.values(): languages.update((m.get("per_language") or {}).keys())
    for lg in sorted(languages):
        vals={}
        for model,m in models.items():
            row=(m.get("per_language") or {}).get(lg,{})
            for metric,v in row.items():
                if metric in METRICS and isinstance(v,(int,float)): vals.setdefault(metric,[]).append((float(v),model))
        for metric,items in vals.items():
            direction="higher" if metric in HIGHER else "lower"
            best=max(items) if direction=="higher" else min(items)
            out[f"cpu51::{lg}::{metric}"]={"target":best[0],"model":best[1],"direction":direction}
    return out

def compare_board(board:dict,candidate:dict,tol=1e-9)->dict:
    vals=candidate.get("metrics",{}); counts={"WIN":0,"TIE":0,"LOSS":0,"MISSING":0}; rows=[]
    for key,t in sorted(board.items()):
        if key not in vals: status="MISSING"; value=None
        else:
            value=float(vals[key]); target=float(t["target"])
            if abs(value-target)<=tol: status="TIE"
            elif (t["direction"]=="higher" and value>target) or (t["direction"]=="lower" and value<target): status="WIN"
            else: status="LOSS"
        counts[status]+=1; rows.append({"id":key,**t,"value":value,"status":status})
    return {"schema_version":"r8-laya-fullboard-v1","candidate":candidate.get("candidate","unknown"),"counts":counts,"rows":rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--t4",type=Path); ap.add_argument("--cpu51",type=Path); ap.add_argument("--candidate",type=Path); ap.add_argument("--out",type=Path)
    a=ap.parse_args(); board={}
    if a.t4: board.update(best_laya_t4(json.loads(a.t4.read_text())))
    if a.cpu51: board.update(best_laya_cpu51(json.loads(a.cpu51.read_text())))
    result=compare_board(board,json.loads(a.candidate.read_text())) if a.candidate else {"schema_version":"r8-laya-board-v1","targets":board,"count":len(board)}
    txt=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.out:a.out.write_text(txt)
    else:print(txt,end="")

if __name__=="__main__": main()
