#!/usr/bin/env python3
"""Run the next exploratory comparison without decoding photographs again."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from soil_color_orgc.model_comparison import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--commit",required=True)
    args=p.parse_args()
    os.umask(0o077)
    if args.out.exists():
        p.error("Choose a new output directory")
    payload=args.features.read_bytes()
    records=json.loads(payload)
    summary, predictions=compare(records)
    summary.update({"features_sha256":hashlib.sha256(payload).hexdigest(),
                    "code_revision":args.commit,
                    "versions":{n:importlib.metadata.version(n) for n in ("numpy","scikit-learn")}})
    args.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    for name,obj in [("model-comparison-summary.json",summary),("model-predictions.local.json",predictions)]:
        with (args.out/name).open("w",encoding="utf-8") as h:
            json.dump(obj,h,indent=2,allow_nan=False)
            h.write("\n")
    for cohort in ("paired","all_raw"):
        block=summary[cohort]
        print(cohort,block["status"],block["n"],"samples")
        for name,m in block["models"].items():
            print(f"{name:36} MAE={m['mae_pct_points']:.4f} RMSE={m['rmse_pct_points']:.4f} R2={m['r2']:.4f}")
    print("Aggregate summary:",args.out/"model-comparison-summary.json")


if __name__=="__main__":
    main()
