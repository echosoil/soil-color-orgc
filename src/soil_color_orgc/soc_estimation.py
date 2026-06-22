def soc_from_munsell(notation: str):
    """
    Estimate SOC (%) from Munsell Value and Chroma.

    Example notation:
        "10YR 3/4"

    Heuristic:
        SOC% ≈ 12 - 1.5*Value - 0.3*Chroma

    This should later be calibrated against lab measurements.
    """
    try:
        vc = notation.split()[1]
        value_str, chroma_str = vc.split("/")

        value = int(value_str)
        chroma = int(chroma_str)

        soc = 12 - 1.5 * value - 0.3 * chroma
        return max(0.0, float(soc))

    except Exception:
        return None


def soc_from_L(lab):
    """
    Fallback SOC (%) from CIE Lab lightness only.

    Heuristic:
        SOC% ≈ 10 - 0.15*L
    """
    L = float(lab[0])
    soc = 10 - 0.15 * L

    return max(0.0, float(soc))


def estimate_soc(lab, best_munsell: str):
    """
    Prefer Munsell-based SOC if match is valid.
    Otherwise use L-based fallback.

    Returns:
        (soc_estimate, method)
    """
    if best_munsell != "No close Munsell match":
        soc = soc_from_munsell(best_munsell)

        if soc is not None:
            return soc, "munsell"

    return soc_from_L(lab), "L_fallback"