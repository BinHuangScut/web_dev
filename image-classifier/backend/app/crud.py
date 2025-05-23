from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def create_prediction_record(db: Session, prediction: schemas.PredictionCreate) -> models.Prediction:
    db_prediction = models.Prediction(
        original_filename=prediction.original_filename,
        saved_filename=prediction.saved_filename,
        file_path=prediction.file_path,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        model_version=prediction.model_version,
        prediction_time=prediction.prediction_time # This will be set by the schema default if not provided
        # created_at is handled by the database default
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction_by_id(db: Session, prediction_id: int) -> models.Prediction | None:
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()

def get_predictions(db: Session, skip: int = 0, limit: int = 100) -> list[models.Prediction]:
    return db.query(models.Prediction).order_by(models.Prediction.prediction_time.desc()).offset(skip).limit(limit).all() 