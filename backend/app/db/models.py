from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, index=True)
    filename = Column(String, nullable=False)
    r2_path = Column(String, nullable=False)
    row_count = Column(Integer)
    col_count = Column(Integer)
    status = Column(String, default="uploaded") # uploaded, profiled, training, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    profiles = relationship("Profile", back_populates="dataset", uselist=False)
    training_jobs = relationship("TrainingJob", back_populates="dataset")

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    dataset_id = Column(String, ForeignKey("datasets.id"), unique=True)
    profiles_json = Column(JSON) # List[ColumnProfile]
    suggested_target = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="profiles")

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    status = Column(String, default="training") # training, completed, failed
    model_uri = Column(String, nullable=True)
    roc_auc = Column(Float, nullable=True)
    optimal_threshold = Column(Float, nullable=True)
    target_column = Column(String)
    prior_model_uri = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    
    dataset = relationship("Dataset", back_populates="training_jobs")
