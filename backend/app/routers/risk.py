"""Credit-default risk scoring endpoint (public-data demo model; see docs/MODEL_CARD.md)."""
import logging
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_current_user
from app.models.database import User
from risk_model.data import add_features
from risk_model.model import ARTIFACT, load_bundle, predict_proba, risk_band, top_factors

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/risk", tags=["Risk Model"])

PAY_STATUS = Field(..., ge=-2, le=9, description="-2/-1/0 = paid duly, 1..9 = months of payment delay")


class RiskScoreRequest(BaseModel):
    """Repayment profile of one credit-card client (fields follow the UCI dataset)."""
    limit_bal: float = Field(..., gt=0, le=10_000_000, description="Credit limit")
    age: int = Field(..., ge=18, le=100)
    education: int = Field(..., ge=0, le=6, description="1 graduate, 2 university, 3 high school, 4 other")
    pay_1: int = PAY_STATUS
    pay_2: int = PAY_STATUS
    pay_3: int = PAY_STATUS
    pay_4: int = PAY_STATUS
    pay_5: int = PAY_STATUS
    pay_6: int = PAY_STATUS
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float = Field(..., ge=0)
    pay_amt2: float = Field(..., ge=0)
    pay_amt3: float = Field(..., ge=0)
    pay_amt4: float = Field(..., ge=0)
    pay_amt5: float = Field(..., ge=0)
    pay_amt6: float = Field(..., ge=0)


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float
    effect: str


class RiskScoreResponse(BaseModel):
    probability_of_default: float
    score: int = Field(..., description="0-100, probability of default x 100")
    risk_band: str
    model: str
    top_factors: list[Factor]
    disclaimer: str = "Demo model trained on public credit-card data; not real underwriting data."


@lru_cache(maxsize=1)
def _bundle() -> dict:
    return load_bundle()


@router.post("/score", response_model=RiskScoreResponse)
async def score(req: RiskScoreRequest, current_user: User = Depends(get_current_user)):
    """Predict the probability of default and the features driving it."""
    if not ARTIFACT.exists():
        raise HTTPException(status_code=503, detail="Risk model not trained. Run: python -m risk_model.train")
    bundle = _bundle()
    payload = {k.upper(): v for k, v in req.model_dump().items()}
    p = float(predict_proba(bundle, add_features(pd.DataFrame([payload])))[0])
    return RiskScoreResponse(
        probability_of_default=round(p, 4), score=round(p * 100), risk_band=risk_band(p, bundle["threshold"]),
        model=bundle["model_name"], top_factors=top_factors(bundle, payload),
    )
