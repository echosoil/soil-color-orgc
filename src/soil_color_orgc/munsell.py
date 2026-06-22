import pandas as pd
import numpy as np
from colour import XYZ_to_Lab
from colour.difference import delta_E


def load_munsell(csv_path: str):
    """
    Load RIT Munsell CSV.

    Expected columns:
    file order,h,V,C,x,y,Y,X_C,Y_C,Z_C,X_D65,Y_D65,Z_D65,R,G,B,dR,dG,dB

    Uses X_D65, Y_D65, Z_D65 directly and converts them to CIE Lab.
    """
    df = pd.read_csv(csv_path)

    required = {"h", "V", "C", "X_D65", "Y_D65", "Z_D65"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Munsell CSV is missing required columns: {sorted(missing)}")

    df["notation"] = (
        df["h"].astype(str)
        + " "
        + df["V"].astype(str)
        + "/"
        + df["C"].astype(str)
    )

    def xyz_to_lab(row):
        return tuple(XYZ_to_Lab([row["X_D65"], row["Y_D65"], row["Z_D65"]]))

    df["lab"] = df.apply(xyz_to_lab, axis=1)

    return {
        row["notation"]: row["lab"]
        for _, row in df.iterrows()
    }


def find_best_munsell(lab_color, munsell_dict, threshold: float = 8.0):
    """
    Find the closest Munsell chip using CIEDE2000.

    Returns:
        (best_notation, deltaE)

    If no match is closer than threshold, best_notation is:
        "No close Munsell match"
    """
    best_match = None
    best_delta = float("inf")

    sample = np.array(lab_color).reshape(1, 3)

    for notation, lab_ref in munsell_dict.items():
        ref = np.array(lab_ref).reshape(1, 3)
        dE = delta_E(sample, ref, method="CIE 2000")[0]

        if dE < best_delta:
            best_delta = float(dE)
            best_match = notation

    if best_delta > threshold:
        return "No close Munsell match", best_delta

    return best_match, best_delta