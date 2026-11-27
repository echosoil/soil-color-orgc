"""Exploratory nested CV on cached service features; never exports a fitted model."""
from collections import Counter
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from .validation import metrics

BINS = [(0,1),(1,2),(2,4),(4,8),(8,15),(15,100)]
ACCEPTABLE = {"no_close_munsell_match"}


def usable_raw(record):
    raw = record.get("raw")
    return isinstance(raw, dict) and raw.get("lab") is not None and not (
        set(raw.get("warnings", [])) - ACCEPTABLE
    )


def clipped_mae(y, p):
    return mean_absolute_error(y, np.clip(p, 0, 100))


def candidates():
    return {
        "mean_baseline": (DummyRegressor(strategy="mean"), {}),
        "median_baseline": (DummyRegressor(strategy="median"), {}),
        "linear_L": (LinearRegression(), {}),
        "linear_Lab": (LinearRegression(), {}),
        "linear_Lab_nonnegative": (LinearRegression(), {}),
        "log_linear_Lab": (TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1), {}),
        "polynomial_ridge": (make_pipeline(PolynomialFeatures(include_bias=False),
                                          StandardScaler(), Ridge()),
                             {"polynomialfeatures__degree":[1,2], "ridge__alpha":[.1,1,10,100]}),
        "log_polynomial_ridge": (TransformedTargetRegressor(
            regressor=make_pipeline(PolynomialFeatures(include_bias=False), StandardScaler(), Ridge()),
            func=np.log1p, inverse_func=np.expm1),
            {"regressor__polynomialfeatures__degree":[1,2], "regressor__ridge__alpha":[.1,1,10,100]}),
        "gradient_boosting": (GradientBoostingRegressor(random_state=42, loss="huber"),
                              {"max_depth":[1,2], "n_estimators":[50,100], "min_samples_leaf":[5,15]}),
    }


def diagnostics(y,p):
    result = metrics(y,p)
    errors = np.abs(p-y)
    result.update({"negative_predictions":int((p<0).sum()),
                   "above_100_predictions":int((p>100).sum()),
                   "median_absolute_error":float(np.median(errors)),
                   "p90_absolute_error":float(np.quantile(errors,.9)),
                   "max_absolute_error":float(errors.max()),
                   "by_lab_range":[]})
    for low,high in BINS:
        mask = (y>=low if low==0 else y>low) & (y<=high)
        result["by_lab_range"].append({
            "low":low,"high":high,"n":int(mask.sum()),
            "mae_pct_points":float(errors[mask].mean()) if mask.any() else None,
            "bias_pct_points":float((p[mask]-y[mask]).mean()) if mask.any() else None,
        })
    return result


def evaluate_cohort(rows, modes):
    n=len(rows)
    groups=np.array([r["group_id"] for r in rows])
    summary={"n":n,"groups":len(set(groups)),"status":"insufficient_data","models":{}}
    if n<20 or len(set(groups))<5:
        return summary,[]
    y=np.array([r["orgC_lab"] for r in rows],dtype=float)
    data={mode:np.array([r[mode]["lab"] for r in rows],dtype=float) for mode in modes}
    if not np.isfinite(y).all() or np.any((y<0)|(y>100)) or any(
        x.shape!=(n,3) or not np.isfinite(x).all() for x in data.values()
    ):
        raise ValueError("Invalid cached features or laboratory target")
    if np.ptp(y)==0:
        summary["status"]="constant_target"
        return summary,[]
    folds=list(GroupKFold(n_splits=5).split(data[modes[0]],y,groups))
    if any(len(tr)<10 or len(set(groups[tr]))<3 or np.ptp(y[tr])==0 for tr,te in folds):
        return summary,[]
    fold_id=np.zeros(n,dtype=int)
    predictions={}
    chosen={}
    for mode,x in data.items():
        for name in candidates():
            if mode!=modes[0] and name.endswith("_baseline"):
                continue
            key=name if name.endswith("_baseline") else mode+"_"+name
            predictions[key]=np.full(n,np.nan)
            chosen[key]=[]
    scorer=make_scorer(clipped_mae,greater_is_better=False)
    for fold,(tr,te) in enumerate(folds,1):
        if set(groups[tr])&set(groups[te]):
            raise RuntimeError("Outer group leakage")
        fold_id[te]=fold
        for mode,x in data.items():
            for name,(estimator,grid) in candidates().items():
                if mode!=modes[0] and name.endswith("_baseline"):
                    continue
                key=name if name.endswith("_baseline") else mode+"_"+name
                features=x[:,:1] if name=="linear_L" else x
                if grid:
                    search=GridSearchCV(estimator,grid,scoring=scorer,
                                        cv=GroupKFold(n_splits=3),n_jobs=1,error_score="raise")
                    search.fit(features[tr],y[tr],groups=groups[tr])
                    estimator=search.best_estimator_
                    chosen[key].append({"fold":fold,"parameters":search.best_params_})
                else:
                    estimator.fit(features[tr],y[tr])
                pred=estimator.predict(features[te])
                # Preserve unconstrained linear predictions as historical comparators.
                if name not in {"linear_L","linear_Lab"}:
                    pred=np.clip(pred,0,100)
                if not np.isfinite(pred).all():
                    raise ValueError("Nonfinite prediction in "+key)
                predictions[key][te]=pred
    if np.any(fold_id==0) or any(not np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Incomplete outer predictions")
    summary.update({"status":"exploratory_nested_cv_completed",
                    "lab_range_pct":[float(y.min()),float(y.max())],
                    "fold_sizes":[int((fold_id==i).sum()) for i in range(1,6)],
                    "models":{key:{**diagnostics(y,p),"inner_choices":chosen[key]} for key,p in predictions.items()}})
    private=[{"sample_id":r["sample_id"],"group_id":r["group_id"],"fold":int(fold_id[i]),
              "orgC_lab":float(y[i]),**{key:float(p[i]) for key,p in predictions.items()}}
             for i,r in enumerate(rows)]
    return summary,private


def compare(records):
    ids=[r["sample_id"] for r in records]
    if len(set(ids))!=len(ids) or any(not r.get("group_id") for r in records):
        raise ValueError("Missing group or duplicate sample ID")
    hashes=[r["image_sha256"] for r in records if r.get("image_sha256")]
    if len(set(hashes))!=len(hashes):
        raise ValueError("Duplicate cached image hash")
    for r in records:
        for mode in ("raw","corrected"):
            data=r.get(mode)
            if data is not None and data.get("method_version")!="soil-colour-v2-preview":
                raise ValueError("Unexpected cached feature method")
    raw=[r for r in records if usable_raw(r)]
    paired=[r for r in records if r.get("included") is True]
    if any(not usable_raw(r) or r.get("corrected",{}).get("gray_calibration")!="applied"
           or r["corrected"].get("lab") is None
           or set(r["corrected"].get("warnings",[]))-ACCEPTABLE for r in paired):
        raise ValueError("Invalid paired eligibility")
    a,ap=evaluate_cohort(paired,["raw","corrected"])
    b,bp=evaluate_cohort(raw,["raw"])
    summary={"source_n":len(records),"source_dispositions":dict(Counter(r["reason"] for r in records)),
             "grouping":"sample_level" if all(r["sample_id"]==r["group_id"] for r in records)
                        else "operator_supplied_groups",
             "paired":a,"all_raw":b,
             "limitations":["Post-hoc exploratory model comparison on previously examined cohort.",
                            "Five outer group folds, three inner group folds for tuning by clipped MAE.",
                            "No independent holdout; model-family comparisons are not final generalization estimates.",
                            "Cohorts may differ: compare raw and corrected only within paired.",
                            "No uncertainty intervals or fitted model release."]}
    return summary,{"paired":ap,"all_raw":bp}
