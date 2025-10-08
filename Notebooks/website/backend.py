# app.py
# FastAPI backend for POS/All Online Promotion Recommendation (Top-5)

import os
import json
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field

# Optional / lazy imports
try:
    import joblib
except Exception:
    joblib = None

# LightGBM: ใช้ได้ทั้งไฟล์ .pkl (sklearn wrapper) และ .txt (Booster native)
try:
    import lightgbm as lgb
except Exception:
    lgb = None


# ========= CONFIG =========

ROOT = Path(".").resolve()
# โฟลเดอร์ artifacts/ และ datasets — ปรับได้ตามโครงสร้างจริงของคุณ
ARTI = ROOT / "Notebooks" / "artifacts"
DATA = ROOT / "Datasets" / "mockup_ver2"

# ตั้งค่าคอร์ (ลด overhead บน Windows / loky)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

TOPK_FINAL = 5  # ปกติเราคืน 5 อันดับ

# ========= Pydantic Schemas =========

class BasketItem(BaseModel):
    product_id: str
    qty: int = Field(ge=1)
    price: float = Field(ge=0)

class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    store_id: Optional[str] = None
    basket: List[BasketItem]
    timestamp: Optional[str] = None  # ISO string; ถ้าไม่ส่งมาจะใช้ตอนเรียก

class RecommendItem(BaseModel):
    promotion_id: str
    promotion_type: str
    title: Optional[str] = None
    expected_uplift: Optional[float] = None
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class RecommendResponse(BaseModel):
    top_k: List[RecommendItem]
    latency_ms: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None


# ========= Utilities =========

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def ndcg_at_k(rels, k=5) -> float:
    """NumPy 2.0 safe NDCG@K"""
    rels = np.asarray(rels, dtype=float).ravel()
    if rels.size == 0:
        return 0.0
    k = int(min(k, rels.size))
    rels_k = rels[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = np.sum((np.power(2.0, rels_k) - 1.0) * discounts)
    ideal_k = np.sort(rels)[::-1][:k]
    idcg = np.sum((np.power(2.0, ideal_k) - 1.0) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_dt(ts: Optional[str]) -> datetime:
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now()


# ========= Artifact Loading (cached) =========

@lru_cache(maxsize=1)
def load_feature_config() -> Dict[str, Any]:
    cfg_path = ROOT / "feature_config.json"
    if not cfg_path.exists():
        cfg_path = ARTI / "configs" / "feature_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"feature_config.json not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_promotions() -> pd.DataFrame:
    # promotions_processed.csv
    for path in [
        ROOT / "promotions_processed.csv",
        ARTI / "data" / "promotions_processed.csv",
        DATA / "promotions_processed.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            # normalize columns commonly used
            for c in ["promotion_id", "promotion_type", "title", "product_scope", "start_ts", "end_ts", "channel"]:
                if c not in df.columns:
                    df[c] = None
            return df
    raise FileNotFoundError("promotions_processed.csv not found")

@lru_cache(maxsize=1)
def load_promo_products() -> pd.DataFrame:
    for path in [
        ROOT / "promotion_products.csv",
        ARTI / "data" / "promotion_products.csv",
        DATA / "promotion_products.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            # expect columns: promotion_id, product_id
            for c in ["promotion_id", "product_id"]:
                if c not in df.columns:
                    raise ValueError("promotion_products.csv must contain promotion_id, product_id")
            return df
    # optional file
    return pd.DataFrame(columns=["promotion_id", "product_id"])

@lru_cache(maxsize=1)
def load_need_profiles() -> pd.DataFrame:
    for path in [
        ROOT / "need_state_profiles.csv",
        ARTI / "data" / "need_state_profiles.csv",
        DATA / "need_state_profiles.csv",
    ]:
        if path.exists():
            return pd.read_csv(path)
    # optional
    return pd.DataFrame()

@lru_cache(maxsize=1)
def load_ptype_model():
    """โหลดโมเดลทำนายประเภทโปร (CalibratedClassifierCV/LightGBM sklearn)"""
    # ลองหา ptype_model.pkl ทั้งใน ROOT และ ARTI
    for path in [
        ROOT / "ptype_model.pkl",
        ARTI / "models" / "ptype_model.pkl",
    ]:
        if path.exists():
            if joblib is not None:
                try:
                    return joblib.load(path)
                except Exception:
                    pass
            # ใช้ pickle สำรอง
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("ptype_model.pkl not found")

@lru_cache(maxsize=1)
def load_ranker_model():
    """โหลด ranker: รองรับทั้ง .pkl (sklearn wrapper) และ .txt (LightGBM Booster)"""
    # ความสำคัญ: ต้องรู้ feature order สำหรับ ranker จาก feature_config["ranker_featcols"]
    # ลอง .pkl ก่อน
    for path in [
        ROOT / "ranker_model.pkl",
        ROOT / "ranker_model_tt.pkl",
        ARTI / "models" / "ranker_model.pkl",
        ARTI / "models" / "ranker_model_tt.pkl",
    ]:
        if path.exists():
            if joblib is not None:
                try:
                    return {"type": "sklearn", "model": joblib.load(path)}
                except Exception:
                    pass
            # pickle fallback
            import pickle
            with open(path, "rb") as f:
                return {"type": "sklearn", "model": pickle.load(f)}

    # ลองไฟล์ native LightGBM
    for path in [
        ROOT / "ranker_lgb.txt",
        ARTI / "models" / "ranker_lgb.txt",
    ]:
        if path.exists():
            if lgb is None:
                raise RuntimeError("LightGBM is not installed, cannot load ranker_lgb.txt")
            booster = lgb.Booster(model_file=str(path))
            return {"type": "booster", "model": booster}

    raise FileNotFoundError("Ranker model not found (.pkl or .txt)")

@lru_cache(maxsize=1)
def classes_for_ptype() -> List[str]:
    cfg = load_feature_config()
    classes = cfg.get("ptype_classes") or []
    if not classes:
        raise RuntimeError("ptype_classes is empty in feature_config.json")
    return list(classes)

@lru_cache(maxsize=1)
def ptype_featcols() -> List[str]:
    cfg = load_feature_config()
    cols = cfg.get("ptype_featcols") or []
    if not cols:
        raise RuntimeError("ptype_featcols is empty in feature_config.json")
    return list(cols)

@lru_cache(maxsize=1)
def ranker_featcols() -> List[str]:
    cfg = load_feature_config()
    cols = cfg.get("ranker_featcols") or []
    if not cols:
        # ในไฟล์โมเดล native อาจระบุ Column_0..Column_11 — ต้อง provide mapping ใน feature_config
        raise RuntimeError("ranker_featcols is empty in feature_config.json (need mapping for features in ranker)")
    return list(cols)

@lru_cache(maxsize=1)
def guardrails_cfg() -> Dict[str, Any]:
    # ถ้ามีไฟล์ guardrails จะโหลด, ถ้าไม่มีใช้ดีฟอลต์
    default = {
        "gap_rule_min_gap": 0.05,
        "min_real_promos": 2,
        "diversity_by": ["promotion_type"],
        "max_per_type": 2,
        "cap_nopromo": 1,
        "nopromo_label": "NoPromo",
        "relevance_thresh": 0.0,
        "topk_types": 3,
        "K_final": TOPK_FINAL,
    }
    for path in [
        ROOT / "guardrails_config.json",
        ARTI / "configs" / "guardrails_config.json",
    ]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                g = json.load(f)
            default.update(g)
            break
    return default


# ========= Feature builders =========

def encode_ptype_features(payload: Dict[str, Any], featcols: List[str]) -> pd.DataFrame:
    """
    payload -> DataFrame 1 แถว ตามคอลัมน์ featcols ที่ใช้ train ptype_model
    Payload นี้คุณสามารถเติม logic ให้ derive ฟีเจอร์จริง (qty sum, price sum, time, user segment, ฯลฯ)
    """
    # ตัวอย่าง baseline: สรุปตะกร้าแบบง่าย (คุณสามารถต่อยอดจาก tx_merge2 logic ได้)
    basket = payload["basket"]
    ts = to_dt(payload.get("timestamp"))
    qty_sum = sum(i["qty"] for i in basket)
    amount_sum = sum(i["qty"] * i["price"] for i in basket)

    base = {
        "qty": qty_sum,
        "price": amount_sum,
        "products.base_price": 0.0,  # ถ้ามี base_price จริงสามารถดึงจากแคตตาล็อก
        "stores.zone": 1,            # map จาก store_id -> zone
        "order_hour": ts.hour,
        "dayofweek": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
        "loyalty_score": 0.5,        # ดึงจาก users_features_with_segments ถ้ามี
        "expected_basket_items": max(qty_sum, 1),
        "price_elasticity": 0.2,
        "segment": 1,
    }
    # สร้าง df แล้ว align ตาม featcols
    df = pd.DataFrame([base])
    # เพิ่มคอลัมน์ที่หายไป
    for c in featcols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[featcols].astype(float)
    return df

def recall_candidates(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    ดึงชุดโปรโมชันผู้สมัครเบื้องต้น (recall) ก่อนส่งให้ ranker
    Logic ที่ใช้:
      - active ช่วงเวลาปัจจุบัน
      - แมตช์ product_id ในตะกร้ากับ promotion_products (ถ้ามี)
      - ถ้าแมตช์ไม่ได้ ใช้ product_scope / หรือดึงทั้งโปร type ที่โมเดล ptype บอกว่ามีโอกาสสูง
    """
    promos = load_promotions().copy()
    pmap = load_promo_products()

    ts = to_dt(payload.get("timestamp"))

    # Active window
    def _in_window(row):
        try:
            st = datetime.fromisoformat(str(row.get("start_ts")).replace("Z", "+00:00")) if pd.notna(row.get("start_ts")) else None
            ed = datetime.fromisoformat(str(row.get("end_ts")).replace("Z", "+00:00")) if pd.notna(row.get("end_ts")) else None
        except Exception:
            st = ed = None
        if st and ts < st:
            return False
        if ed and ts > ed:
            return False
        return True

    promos["__active"] = promos.apply(_in_window, axis=1)
    promos = promos[promos["__active"] == True]  # noqa: E712

    # Join ตามสินค้าถ้าเจอ
    basket_pids = {i["product_id"] for i in payload["basket"]}
    if not pmap.empty and basket_pids:
        cand_ids = pmap[pmap["product_id"].isin(basket_pids)]["promotion_id"].unique().tolist()
        sub = promos[promos["promotion_id"].isin(cand_ids)]
        # ถ้าได้บ้าง ใช้ก่อน
        if len(sub) >= 5:
            return sub.reset_index(drop=True)

    # ถ้าแมตช์สินค้าไม่พอ ลองใช้ product_scope แบบหยาบ
    if "product_scope" in promos.columns:
        scope_mask = np.zeros(len(promos), dtype=bool)
        for i, row in promos.iterrows():
            scope = str(row.get("product_scope") or "").lower()
            if not scope:
                continue
            # match แบบง่าย: product_id ใด ๆ ปรากฏใน scope string
            for pid in basket_pids:
                if pid.lower() in scope:
                    scope_mask[i] = True
                    break
        sub = promos[scope_mask]
        if len(sub) >= 5:
            return sub.reset_index(drop=True)

    # สุดท้าย: ถ้ายังน้อย ให้คืนโปร active ทั้งหมด (แล้วให้ ranker ช่วยคัด)
    return promos.reset_index(drop=True)


def build_ranker_features(payload: Dict[str, Any], candidates: pd.DataFrame, featcols: List[str]) -> pd.DataFrame:
    """
    แปลง (basket + promotion row) -> ranker feature vector ตามคอลัมน์ที่โมเดลเรียนรู้
    ตรงนี้ขึ้นกับ pipeline ที่คุณฝึก — ด้านล่างคือ baseline ให้ปรับแต่งเพิ่มได้
    """
    basket = payload["basket"]
    ts = to_dt(payload.get("timestamp"))
    qty_sum = sum(i["qty"] for i in basket)
    amount_sum = sum(i["qty"] * i["price"] for i in basket)

    rows = []
    for _, pr in candidates.iterrows():
        feat = {
            # ตัวอย่างฟีเจอร์ร่วมกัน (basket-level)
            "f_qty_sum": qty_sum,
            "f_amount_sum": amount_sum,
            "f_hour": ts.hour,
            "f_dow": ts.weekday(),
            "f_is_weekend": int(ts.weekday() >= 5),
            # ตัวอย่างฟีเจอร์โปรโมชัน (promotion-level)
            "f_type_is_brandday": 1.0 if str(pr.get("promotion_type")) == "Brandday" else 0.0,
            "f_type_is_b1g1": 1.0 if str(pr.get("promotion_type")) in ["Buy 1 get 1", "Buy1Get1"] else 0.0,
            "f_type_is_flash": 1.0 if str(pr.get("promotion_type")) == "Flash Sale" else 0.0,
            "f_type_is_mega": 1.0 if str(pr.get("promotion_type")) == "Mega Sale" else 0.0,
            "f_type_is_coupon": 1.0 if "coupon" in str(pr.get("promotion_type")).lower() else 0.0,
            # คุณสามารถเพิ่มฟีเจอร์จาก promotions_processed เช่น discount_value/ channel ฯลฯ
        }
        rows.append(feat)

    X = pd.DataFrame(rows)

    # align ตาม featcols (สำคัญ!)
    for c in featcols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[featcols].astype(float)
    return X


# ========= Inference =========

def predict_ptype_proba(payload: Dict[str, Any]) -> np.ndarray:
    model = load_ptype_model()
    featcols = ptype_featcols()
    X = encode_ptype_features(payload, featcols)
    # บางครั้ง CalibratedClassifierCV/LightGBM บน Windows จะช้าเพราะคอร์
    # คุณสามารถบังคับ n_jobs=1 ที่ตอนโหลดโมเดล (ดูโน้ตในงานก่อนหน้า) ถ้าจำเป็น
    prob = model.predict_proba(X)[0]  # shape: [n_classes]
    return prob

def rank_candidates(payload: Dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
    rk = load_ranker_model()
    featcols = ranker_featcols()

    X_rank = build_ranker_features(payload, candidates, featcols)

    if rk["type"] == "sklearn":
        # scikit-learn wrapper (เช่น LGBMRanker หรือ GBDT pointwise)
        scores = rk["model"].predict(X_rank)
    elif rk["type"] == "booster":
        # Native LightGBM Booster
        d = lgb.Dataset(X_rank) if lgb is not None else None
        # booster.predict รับ numpy / pandas ได้โดยตรง (ไม่ต้อง Dataset)
        scores = rk["model"].predict(X_rank)
    else:
        raise RuntimeError("Unknown ranker model type")

    out = candidates.copy()
    out["__score"] = scores
    out = out.sort_values("__score", ascending=False).reset_index(drop=True)
    return out

def apply_guardrails(df: pd.DataFrame, k_final: int) -> pd.DataFrame:
    g = guardrails_cfg()
    max_per_type = int(g.get("max_per_type", 2))
    min_real = int(g.get("min_real_promos", 2))
    nopromo_label = str(g.get("nopromo_label", "NoPromo"))
    cap_nopromo = int(g.get("cap_nopromo", 1))
    diversity_by = g.get("diversity_by", ["promotion_type"])

    chosen = []
    type_count = {}
    nopromo_count = 0

    for _, row in df.iterrows():
        t = str(row.get("promotion_type", "")) or "Unknown"
        if t == nopromo_label:
            if nopromo_count >= cap_nopromo:
                continue
            nopromo_count += 1
        else:
            type_count[t] = type_count.get(t, 0) + 1
            if type_count[t] > max_per_type:
                continue

        chosen.append(row)
        if len(chosen) >= k_final:
            break

    # ensure at least min_real promos (หากเกินจำกัด type ให้ผ่อนปรนในรอบสอง)
    if sum(1 for r in chosen if str(r.get("promotion_type", "")) != nopromo_label) < min_real:
        # ผ่อนปรน: เติมจาก df ที่เหลือ (ไม่จำกัด per type)
        for _, row in df.iterrows():
            if row in chosen:
                continue
            if str(row.get("promotion_type", "")) != nopromo_label:
                chosen.append(row)
                if len(chosen) >= k_final:
                    break

    if not chosen:
        return df.head(k_final)
    return pd.DataFrame(chosen).head(k_final).reset_index(drop=True)


# ========= FastAPI App =========

app = FastAPI(title="Promotion Recommender API", version="1.0.0")

@app.get("/health")
def health():
    try:
        _ = load_feature_config()
        _ = load_ptype_model()
        _ = load_ranker_model()
        _ = load_promotions()
        return {"status": "ok", "time": now_iso()}
    except Exception as e:
        return {"status": "error", "error": str(e), "time": now_iso()}

@app.get("/config")
def get_config():
    cfg = load_feature_config()
    g = guardrails_cfg()
    return {"feature_config": cfg, "guardrails": g}

@app.get("/promotions")
def list_promotions(limit: int = Query(50, ge=1, le=500)):
    promos = load_promotions().head(limit)
    return {"count": len(promos), "items": promos.to_dict(orient="records")}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, k: int = Query(TOPK_FINAL, ge=1, le=10)):
    import time
    t0 = time.time()
    payload = req.dict()
    if not payload.get("timestamp"):
        payload["timestamp"] = now_iso()

    # 1) p(type|X) → เอาไว้ช่วย recall โปร (ถ้าจำเป็น)
    try:
        pprob = predict_ptype_proba(payload)  # shape [n_classes]
        cls = classes_for_ptype()
        ptype_sorted = [cls[i] for i in np.argsort(-pprob)]
    except Exception as e:
        # ไม่จบชีวิต—หาก ptype ล้ม ให้ไปขั้นต่อไปเลย
        pprob = None
        ptype_sorted = []
        ptype_err = str(e)

    # 2) recall candidates
    cands = recall_candidates(payload)
    if cands.empty:
        raise HTTPException(404, "No active promotions found")

    # 3) rank
    ranked = rank_candidates(payload, cands)

    # 4) guardrails + cut to Top-K
    final = apply_guardrails(ranked, k_final=k)

    # 5) compose response
    items: List[RecommendItem] = []
    for _, r in final.iterrows():
        items.append(RecommendItem(
            promotion_id=str(r.get("promotion_id", "")),
            promotion_type=str(r.get("promotion_type", "")),
            title=str(r.get("title", "")) if pd.notna(r.get("title", "")) else None,
            expected_uplift=float(r.get("__score", np.nan)) if "__score" in r else None,
            reason=None,  # สามารถเติมเหตุผล เช่น จาก ptype หรือ scope-match
            meta={"product_scope": r.get("product_scope", None)}
        ))

    latency_ms = int((time.time() - t0) * 1000)
    return RecommendResponse(
        top_k=items,
        latency_ms=latency_ms,
        debug={
            "ptype_sorted": ptype_sorted[:5],
            "n_candidates": int(len(cands)),
        },
    )
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # โปรดจำกัด origin จริงตอนขึ้น prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)