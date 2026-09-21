from __future__ import annotations
import argparse, json
from pathlib import Path

MUTABLE={"main","master","latest"}

def audit(reg:dict)->dict:
    bad=[]; blocked=[]
    for key,src in reg["sources"].items():
        rev=src.get("revision") or src.get("sha")
        if src.get("status")!="pinned" or not rev:
            blocked.append(key); continue
        if str(rev).lower() in MUTABLE:
            bad.append({"source":key,"reason":"mutable revision"})
    return {"pinned":len(reg["sources"])-len(blocked),"blocked":blocked,"invalid":bad,"total":len(reg["sources"])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("registry",type=Path); ap.add_argument("--strict",action="store_true")
    a=ap.parse_args(); out=audit(json.loads(a.registry.read_text()))
    print(json.dumps(out,indent=2,sort_keys=True))
    if out["invalid"]: raise SystemExit(1)
    if a.strict and out["blocked"]: raise SystemExit(2)

if __name__=="__main__": main()
