#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)

"""
验证 BEDROC 不同公式实现：
  1. 当前 v6.1 代码实现（与 RDKit 一致）
  2. 用户要求的 Truchon & Bayly 2007 Eqs. 11-12 实现
  3. RDKit 官方 CalcBEDROC
"""
import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC


def bedroc_current(y_true, y_prob, alpha=20.0):
    """当前 v6.1 实现"""
    n = len(y_true)
    n_act = int(y_true.sum())
    if n_act == 0:
        return 0.0
    if n_act == n:
        return 1.0

    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    act_ranks = np.where(y_sorted == 1)[0] + 1

    denom = (1.0 / n) * ((1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0))
    sum_exp = np.sum(np.exp(-alpha * act_ranks / n))
    rie = sum_exp / (n_act * denom)

    ratio = n_act / n
    rie_max = (1.0 - np.exp(-alpha * ratio)) / (ratio * (1.0 - np.exp(-alpha)))
    rie_min = (1.0 - np.exp(alpha * ratio)) / (ratio * (1.0 - np.exp(alpha)))

    bedroc = (rie - rie_min) / (rie_max - rie_min)
    return float(np.clip(bedroc, 0.0, 1.0)), rie, rie_min, rie_max


def bedroc_user_request(y_true, y_prob, alpha=20.0):
    """
    用户要求的 Truchon & Bayly 2007 Eqs. 11-12 实现
    RIE_min = [exp(α/N) * (1 - exp(-α))] / [R_a * (exp(α/N) - 1)]
    RIE_max = [1 - exp(-α * R_a)] / [R_a * (1 - exp(-α/N))]
    这里尝试两种 R_a 解释：ratio 与 n_act
    """
    n = len(y_true)
    n_act = int(y_true.sum())
    if n_act == 0:
        return {}
    if n_act == n:
        return {"ratio": 1.0, "n_act": 1.0}

    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    act_ranks = np.where(y_sorted == 1)[0] + 1

    denom = (1.0 / n) * ((1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0))
    sum_exp = np.sum(np.exp(-alpha * act_ranks / n))
    rie = sum_exp / (n_act * denom)

    results = {}
    for label, R_a in [("ratio", n_act / n), ("n_act", n_act)]:
        rie_min = np.exp(alpha / n) * (1.0 - np.exp(-alpha)) / (R_a * (np.exp(alpha / n) - 1.0))
        rie_max = (1.0 - np.exp(-alpha * R_a)) / (R_a * (1.0 - np.exp(-alpha / n)))
        diff = rie_max - rie_min
        bedroc = np.nan if abs(diff) < 1e-15 else (rie - rie_min) / diff
        results[label] = {
            "RIE": rie,
            "RIE_min": rie_min,
            "RIE_max": rie_max,
            "RIE_max - RIE_min": diff,
            "BEDROC": bedroc,
        }
    return results


def bedroc_rdkit(y_true, y_prob, alpha=20.0):
    """RDKit 官方实现"""
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    scores = [(0.0, bool(y)) for y in y_sorted]
    return CalcBEDROC(scores, 1, alpha)


def test_case(name, y_true, y_prob):
    print(f"\n{'='*60}")
    print(f"测试: {name} | N={len(y_true)}, n_act={int(y_true.sum())}, alpha=20")
    print(f"{'='*60}")

    rdkit_val = bedroc_rdkit(y_true, y_prob, alpha=20.0)
    print(f"RDKit CalcBEDROC          = {rdkit_val:.6f}")

    cur_val, cur_rie, cur_min, cur_max = bedroc_current(y_true, y_prob, alpha=20.0)
    print(f"当前 v6.1 BEDROC          = {cur_val:.6f}  (RIE={cur_rie:.6e}, "
          f"RIE_min={cur_min:.6e}, RIE_max={cur_max:.6e})")

    user_res = bedroc_user_request(y_true, y_prob, alpha=20.0)
    for label, res in user_res.items():
        print(f"用户公式 (R_a={label:5s}) BEDROC = {res['BEDROC']:.6f}  "
              f"(RIE={res['RIE']:.6e}, RIE_min={res['RIE_min']:.6e}, "
              f"RIE_max={res['RIE_max']:.6e}, diff={res['RIE_max - RIE_min']:.6e})")


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # 基准测试: N=100, n_act=10, alpha=20
    y_true = np.zeros(100, dtype=int)
    y_true[:10] = 1

    # 最好情况：前 10 个全为 active
    y_prob_best = np.arange(100, 0, -1, dtype=float)
    test_case("最好排名 (actives 在前)", y_true, y_prob_best)

    # 最差情况：后 10 个为 active
    y_prob_worst = np.arange(1, 101, dtype=float)
    test_case("最差排名 (actives 在后)", y_true, y_prob_worst)

    # 随机情况
    rng = np.random.RandomState(42)
    y_prob_random = rng.rand(100)
    test_case("随机排名", y_true, y_prob_random)
