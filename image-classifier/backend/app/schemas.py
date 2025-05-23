from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Schema for creating a prediction (input)
class PredictionCreate(BaseModel):
    original_filename: str
    saved_filename: str
    file_path: str
    predicted_class: str
    confidence: float
    model_version: str = "SqueezeNet 1.1"
    prediction_time: datetime = datetime.utcnow()

# Schema for reading a prediction (output, includes ID and other DB-generated fields)
class PredictionResponse(PredictionCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True # Changed from from_attributes = True for Pydantic v1 compatibility if needed
                        # For Pydantic V2, from_attributes is the way.
                        # Assuming we might be in an env with Pydantic V1 via FastAPI default.
                        # If using Pydantic V2 explicitly, use: from_attributes = True
        # Compatibility for Pydantic v1/v2 with FastAPI
        # For Pydantic v2, it should be `model_config = {"from_attributes": True}`
        # For Pydantic v1, it's `orm_mode = True`
        # Let's stick to orm_mode for broader compatibility unless Pydantic V2 is enforced.
        pass 