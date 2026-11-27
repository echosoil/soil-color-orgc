import csv
import numpy as np
import pytest
from soil_color_orgc.validation import read_manifest, evaluate_pairs, parse_region


def test_manifest_requires_scientific_group(tmp_path):
    p = tmp_path/"manifest.csv"
    p.write_text("sample_id,group_id,image_path,orgC_lab\nA,,a.jpg,2\n")
    with pytest.raises(ValueError, match="row 2"):
        read_manifest(p)
    p.write_text("sample_id,group_id,image_path,orgC_lab\nA,g,a.jpg,2\n")
    assert read_manifest(p)[0]["image_path"] == tmp_path/"a.jpg"


def test_duplicate_sample_rejected(tmp_path):
    p=tmp_path/"manifest.csv"
    p.write_text("sample_id,group_id,image_path,orgC_lab\nA,g,a.jpg,2\nA,h,b.jpg,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(p)


def test_grouped_common_predictions():
    records=[]
    for i in range(30):
        y=float(i%10+1)
        lab=[70-y*2, 10.+i%3, 20.+i%4]
        records.append({"included":True,"reason":"paired_eligible","sample_id":str(i),
                        "group_id":str(i//5),"orgC_lab":y,
                        "raw":{"lab":[lab[0]+i%3, *lab[1:]]},"corrected":{"lab":lab}})
    summary, pred=evaluate_pairs(records)
    assert summary["status"]=="exploratory_grouped_cv_completed"
    assert len(pred)==30
    for g in set(p["group_id"] for p in pred):
        assert len({p["fold"] for p in pred if p["group_id"]==g})==1
    assert summary["metrics"]["gray_L"]["mae_pct_points"] < 1e-10
    assert summary["metrics"]["raw_L"]["mae_pct_points"] > 0
    assert summary["metrics"]["mean_baseline"]["mae_pct_points"] > 0


def test_small_dataset_not_validated():
    s,p=evaluate_pairs([])
    assert s["status"]=="insufficient_paired_data" and not p


def test_invalid_region():
    with pytest.raises(ValueError):
        parse_region("0,0,nan,1")
