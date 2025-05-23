from sqlalchemy import Column, Integer, String, Float, DateTime, func
from .database import Base
from datetime import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, index=True)
    saved_filename = Column(String, unique=True, index=True)
    file_path = Column(String) # Path on the server where the file is stored
    predicted_class = Column(String, index=True)
    confidence = Column(Float)
    model_version = Column(String, default="SqueezeNet 1.1")
    prediction_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now()) 