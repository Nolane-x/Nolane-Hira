from __future__ import annotations
import argparse, json
from pathlib import Path

def compare(value: float, target: float, direction: str, tol: float=1e-9) -> str:
    delta=value-target
    if abs(delta)<=tol: return "TIE"
    if direction=="higher": return "WIN" if value>target else "LOSS"
    if direction=="lower": return "WIN" if value<target else "LOSS"
    raise ValueError(f"unknown direction: {direction}")

def build_scorecard(targets: dict, results: dict) -> dict:
    tol=float(targets.get("rules",{}).get("win_tolerance",1e-9))
    vals=results.get("metrics",{}); rows=[]; counts={"WIN":0,"TIE":0,"LOSS":0,"MISSING":0}
    for t in targets["targets"]:
        if t["id"] not in vals:
            status="MISSING"; value=None
        else:
            value=float(vals[t["id"]]); status=compare(value,float(t["target"]),t["direction"],tol)
        counts[status]+=1
        rows.append({**t,"value":value,"status":status})
    return {"schema_version":"r8-scorecard-v1","candidate":results.get("candidate","unknown"),"counts":counts,"rows":rows}

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("targets",type=Path); ap.add_argument("results",type=Path); ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    score=build_scorecard(json.loads(args.targets.read_text()),json.loads(args.results.read_text()))
    text=json.dumps(score,indent=2,sort_keys=True)+"\n"
    if args.out: args.out.write_text(text)
    else: print(text,end="")

if __name__=="__main__": main()
