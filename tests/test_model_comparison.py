import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from soil_color_orgc import model_comparison as mc


def records():
    out=[]
    for i in range(30):
        raw={"lab":[70-i,10.,20.],"method_version":"soil-colour-v2-preview","warnings":[]}
        corrected={**raw,"gray_calibration":"applied"}
        out.append({"sample_id":str(i),"group_id":str(i//5),"orgC_lab":float(i+1),
                    "image_sha256":str(i),"raw":raw,"corrected":corrected,
                    "included":i<25,"reason":"paired_eligible" if i<25 else "gray_not_applied"})
    return out


def test_comparison_groups_and_cohorts(monkeypatch):
    monkeypatch.setattr(mc,"candidates",lambda:{
        "linear_Lab":(LinearRegression(),{}),
        "linear_Lab_nonnegative":(LinearRegression(),{}),
    })
    summary,pred=mc.compare(records())
    assert summary["paired"]["n"]==25
    assert summary["all_raw"]["n"]==30
    assert summary["paired"]["status"]=="exploratory_nested_cv_completed"
    for cohort in pred.values():
        for g in {r["group_id"] for r in cohort}:
            assert len({r["fold"] for r in cohort if r["group_id"]==g})==1
        assert all(0<=r["raw_linear_Lab_nonnegative"]<=100 for r in cohort)
    assert summary["paired"]["models"]["raw_linear_Lab"]["mae_pct_points"]<1e-10


def test_nested_grid_and_bounds(monkeypatch):
    from sklearn.linear_model import Ridge
    monkeypatch.setattr(mc,"candidates",lambda:{"ridge":(Ridge(),{"alpha":[.1,1]})})
    summary,pred=mc.evaluate_cohort(records(),["raw"])
    assert len(summary["models"]["raw_ridge"]["inner_choices"])==5
    assert len(pred)==30


def test_duplicate_hash_rejected():
    rows=records()
    rows[1]["image_sha256"]=rows[0]["image_sha256"]
    with pytest.raises(ValueError,match="Duplicate cached image"):
        mc.compare(rows)


def test_range_coverage():
    y=np.array([0.,1.,2.,4.,8.,15.,100.])
    d=mc.diagnostics(y,y)
    assert sum(r["n"] for r in d["by_lab_range"])==len(y)
