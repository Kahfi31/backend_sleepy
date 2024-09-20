from pydantic import BaseModel, EmailStr
from datetime import date, datetime, time

class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    
class NameRequest(BaseModel):
    name: str
    email: str
        
class UserUpdate(BaseModel):
    name: str = None
    email: str = None
    gender: str = None
    work: str = None
    date_of_birth: str = None
    weight: int = None
    height: int = None
    upper_pressure: int = None
    lower_pressure: int = None
    heart_rate: int = None
    daily_steps: int = None
    sleep_time: int = None
    wake_time: int = None
    
class UserProfile(BaseModel):
    name: str = None
    email: str = None
    gender: int = None
    date_of_birth: str = None
    
class SleepData(BaseModel):
    email: str
    sleep_time: datetime
    wake_time: datetime
    
class PredictionInput(BaseModel):
    age: int
    work_id: int
    gender: int  # 0 for male, 1 for female
    height: float
    weight: float
    upper_pressure: float
    lower_pressure: float
    heart_rate: float
    physical_activity_level: float
    quality_of_sleep: float
    stress_level: float
    
class PredictRequest(BaseModel):
    email: EmailStr

class OtpRequest(BaseModel):
    email: EmailStr
    
class UserData(BaseModel):
    email: str
    name: str
    gender: int
    work: str
    date_of_birth: date
    height: float
    weight: float 
    
class SleepDataResponse(BaseModel):
    sleep_time: str
    wake_time: str
    
    class Config:
        orm_mode = True