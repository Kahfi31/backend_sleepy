import logging
from fastapi import FastAPI, Depends, HTTPException,Request, APIRouter
from sqlalchemy.orm import Session
from datetime import datetime
from . import models, schemas, database
from pydantic import BaseModel, EmailStr
from fastapi import Request
import joblib
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from .database import get_db
from .models import User


model_path = os.path.join(os.getcwd(), 'ml_model', 'xgb_model_Test.pkl')
model = joblib.load(model_path)
gender_path = os.path.join(os.getcwd(), 'ml_model', 'Gender_label_encoder.pkl')
gender_encoder = joblib.load(gender_path)
occupation_encoder_path = os.path.join(os.getcwd(), 'ml_model', 'Occupation_label_encoder.pkl')
occupation_encoder = joblib.load(occupation_encoder_path)
bmi_encoder_path = os.path.join(os.getcwd(), 'ml_model', 'BMI Category_label_encoder.pkl')
bmi_encoder = joblib.load(bmi_encoder_path)
scaler_path = os.path.join(os.getcwd(), 'ml_model', 'minmax_scaler_split.pkl')
scaler = joblib.load(scaler_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
load_dotenv()
router = APIRouter()

user_router = APIRouter(prefix="/user", tags=["Users"])

def normalize_work_title(work_title: str) -> str:
    """Normalisasi pekerjaan dengan menghapus spasi tambahan, mengubah ke huruf kecil, dan menghapus karakter non-alfabet."""
    return ''.join(e for e in work_title.lower().strip() if e.isalnum() or e.isspace()).replace(" ", "")

@user_router.post("/save-data/")
def save_data(data: schemas.UserData, db: Session = Depends(get_db)):
    # Query the user by email
    user = db.query(User).filter(User.email == data.email).first()

    if user:
        # Update existing user data
        user.name = data.name
        user.gender = data.gender
        user.work = data.work
        user.date_of_birth = data.date_of_birth
        user.height = data.height
        user.weight = data.weight
    else:
        # Create a new user entry if it doesn't exist
        new_user = User(
            email=data.email,
            name=data.name,
            gender=data.gender,
            work=data.work,
            date_of_birth=data.date_of_birth,
            height=data.height,
            weight=data.weight,
        )
        db.add(new_user)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save data")

    return {"message": "Data saved successfully"}

@user_router.put("/save-name/")
async def save_name(name_request: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Received request to save name: {name_request.name} for email: {name_request.email}")
    try:
        # Cari user berdasarkan email, bukan nama
        user = db.query(models.User).filter(models.User.email == name_request.email).first()
        
        if user:
            logger.info(f"Found user with email: {user.email}")
            # Update nama user
            user.name = name_request.name
            db.commit()
            db.refresh(user)
            logger.info("Name saved successfully")
            return {"message": "Name saved successfully", "user": user}
        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error saving name: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save name: {e}")

@user_router.put("/save-gender/")
async def save_gender(user_data: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Received request to update user data: {user_data}")
    try:
        # Cari user berdasarkan email, bukan name
        user = db.query(models.User).filter(models.User.email == user_data.email).first()
        
        if user:
            logger.info(f"Found user: {user.email}")
            if user_data.gender is not None:
                try:
                    # Konversi gender ke integer dan tambahkan logging
                    gender_value = int(user_data.gender)
                    logger.info(f"Updating gender to: {gender_value}")
                    user.gender = gender_value
                except ValueError:
                    logger.error("Invalid gender value received")
                    raise HTTPException(status_code=400, detail="Invalid gender value")
            db.commit()
            db.refresh(user)
            logger.info("Gender saved successfully")
            return {"message": "Gender saved successfully", "user": user}
        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {e}")

@user_router.put("/save-work/")
async def save_work(user_data: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Menerima permintaan untuk memperbarui data user: {user_data}")
    
    try:
        # Normalisasi pekerjaan
        normalized_work = normalize_work_title(user_data.work)
        
        user = db.query(models.User).filter(models.User.email == user_data.email).first()
        
        if user:
            logger.info(f"User ditemukan: {user.email}")
            
            # Mapping pekerjaan yang sudah ada (gunakan key yang sudah dinormalisasi)
            work_id_map = {
                normalize_work_title('Accountant'): 0,
                normalize_work_title('Doctor'): 1,
                normalize_work_title('Engineer'): 2,
                normalize_work_title('Lawyer'): 3,
                normalize_work_title('Manager'): 4,
                normalize_work_title('Nurse'): 5,
                normalize_work_title('Sales Representative'): 6,
                normalize_work_title('Salesperson'): 7,
                normalize_work_title('Scientist'): 8,
                normalize_work_title('Software Engineer'): 9,
                normalize_work_title('Teacher'): 10,
                normalize_work_title('Ibu Rumah Tangga'): 11,
                normalize_work_title('Administrator'): 12,
                normalize_work_title('Artist'): 13,
                normalize_work_title('Librarian'): 14,
                normalize_work_title('Marketing'): 15,
                normalize_work_title('Programmer'): 16,
                normalize_work_title('Writer'): 17,
                normalize_work_title('Editing'): 18,
                normalize_work_title('Penyiar Radio'): 19,
                normalize_work_title('Technician'): 20
            }

            if normalized_work not in work_id_map:
                # Jika pekerjaan tidak ada di work_id_map, tambahkan pekerjaan baru
                new_work_id = len(work_id_map)  # Menggunakan panjang work_id_map sebagai ID baru
                work_id_map[normalized_work] = new_work_id
                logger.info(f"Menambahkan pekerjaan baru: {user_data.work} dengan work_id: {new_work_id}")

                # Tambahkan pekerjaan baru ke database
                new_work = models.Work(title=user_data.work)  # Simpan pekerjaan baru ke database
                db.add(new_work)
                db.commit()
                db.refresh(new_work)
                user.work_id = new_work_id

            else:
                # Jika pekerjaan sudah ada di work_id_map, ambil ID-nya
                logger.info(f"Pekerjaan sudah ada: {user_data.work} dengan work_id: {work_id_map[normalized_work]}")
                user.work_id = work_id_map[normalized_work]

            user.work = user_data.work

            # Data untuk pekerjaan baru atau default (gunakan key yang sudah dinormalisasi)
            work_data = {
                normalize_work_title('Accountant'): (7.891892, 58.108108, 4.594595),
                normalize_work_title('Doctor'): (6.647887, 55.352113, 6.732394),
                normalize_work_title('Engineer'): (8.412698, 51.857143, 3.888889),
                normalize_work_title('Lawyer'): (7.893617, 70.425532, 5.063830),
                normalize_work_title('Manager'): (7.0, 55.0, 5.0),
                normalize_work_title('Nurse'): (7.369863, 78.589041, 5.547945),
                normalize_work_title('Sales Representative'): (4.0, 30.0, 8.0),
                normalize_work_title('Salesperson'): (6.0, 45.0, 7.0),
                normalize_work_title('Scientist'): (5.0, 41.0, 7.0),
                normalize_work_title('Software Engineer'): (6.5, 48.0, 6.0),
                normalize_work_title('Teacher'): (6.975, 45.625, 4.525),
                normalize_work_title('Ibu Rumah Tangga'): (4.32, 50.2, 7.56372),
                normalize_work_title('Administrator'): (3.21, 42,5, 5.3029),
                normalize_work_title('Artist'): (8.82617, 80.44, 7.86746),
                normalize_work_title('Librarian'): (4.583, 65.221, 6.56372),
                normalize_work_title('Marketing'): (7.6727, 75.102, 8.4135),
                normalize_work_title('Programmer'): (8.8675, 38.73, 8.3919),
                normalize_work_title('Writer'): (6.70, 43.65, 7.125),
                normalize_work_title('Editing'): (8.847, 40.58, 8.147),
                normalize_work_title('Penyiar Radio'): (4.6282, 56.74, 5.435),
                normalize_work_title('Technician'): (7.3182, 68.43, 8.321)
            }

            # Mengambil data pekerjaan
            quality_of_sleep, physical_activity_level, stress_level = work_data.get(normalized_work, (5.0, 50.0, 5.0))

            # Pemrosesan fitur dan prediksi
            X_input = [[user.work_id, quality_of_sleep, physical_activity_level, stress_level] + [0] * 8]
            X_scaled = scaler.transform(X_input)  # Scaling input data
            prediction = model.predict(X_scaled)

            # Simpan data ke tabel daily
            daily_data = db.query(models.Work).filter(models.Work.email == user.email).first()
            if daily_data:
                daily_data.quality_of_sleep = quality_of_sleep
                daily_data.physical_activity_level = physical_activity_level
                daily_data.stress_level = stress_level
                daily_data.work_id = user.work_id
            else:
                new_daily = models.Work(
                    email=user.email,
                    quality_of_sleep=quality_of_sleep,
                    physical_activity_level=physical_activity_level,
                    stress_level=stress_level,
                    work_id=user.work_id
                )
                db.add(new_daily)
                
            db.commit()
            db.refresh(user)
            
            logger.info("Data pekerjaan dan harian berhasil disimpan dengan prediksi")
            return {"message": "Data pekerjaan dan harian berhasil disimpan", "user": user, "prediction": prediction}
        else:
            logger.warning("User tidak ditemukan")
            raise HTTPException(status_code=404, detail="User tidak ditemukan")
            
    except Exception as e:
        logger.error(f"Kesalahan saat memperbarui data user: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui data user: {e}")
    

@user_router.put("/save-dob/")
async def save_dob(user_data: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Received request to update user data: {user_data}")
    try:
        user = db.query(models.User).filter(models.User.email == user_data.email).first()
        if user:
            logger.info(f"Found user: {user.email}")
            if user_data.date_of_birth:
                # Set date_of_birth
                user.date_of_birth = user_data.date_of_birth
                logger.info(f"Setting date_of_birth to: {user_data.date_of_birth}")
                
                # Calculate age based on date_of_birth
                birth_date = datetime.strptime(user_data.date_of_birth, '%Y-%m-%d')
                today = datetime.today()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                
                # Set age to user
                user.age = age
                logger.info(f"Setting age to: {age}")
            
            db.commit()
            db.refresh(user)
            
            # Cetak informasi pengguna yang relevan
            logger.info(f"User after refresh: {user.email}, {user.name}, {user.date_of_birth}, {user.age}")
            logger.info("Date of birth and age saved successfully")
            return {"message": "Date of birth and age saved successfully", "user": user}

        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {e}")

@user_router.put("/save-weight/")
async def save_weight(user_data: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Received request to update user data: {user_data}")
    try:
        user = db.query(models.User).filter(models.User.email == user_data.email).first()
        if user:
            logger.info(f"Found user: {user.email}")
            if user_data.weight:
                user.weight = user_data.weight
            db.commit()
            db.refresh(user)
            logger.info("Weight saved successfully")
            return {"message": "Weight saved successfully", "user": user}
        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {e}")

@user_router.put("/save-height/")
async def save_height(user_data: schemas.UserUpdate, db: Session = Depends(database.get_db)):
    logger.info(f"Received request to update user data: {user_data}")
    try:
        user = db.query(models.User).filter(models.User.email == user_data.email).first()
        if user:
            logger.info(f"Found user: {user.email}")
            if user_data.height:
                user.height = user_data.height
            db.commit()
            db.refresh(user)
            logger.info("Height saved successfully")
            return {"message": "Height saved successfully", "user": user}
        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {e}")

@user_router.put("/save-blood-pressure/")
async def save_blood_pressure(request: Request, db: Session = Depends(database.get_db)):
    data = await request.json()
    email = data.get('email')
    upper_pressure = data.get('upperPressure')
    lower_pressure = data.get('lowerPressure')

    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        user.upper_pressure = upper_pressure
        user.lower_pressure = lower_pressure
        db.commit()
        db.refresh(user)
        return {"message": "Blood pressure saved successfully", "user": user}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save blood pressure: {e}")
    
@user_router.put("/save-daily-steps/")
async def save_daily_steps(request: Request, db: Session = Depends(database.get_db)):
    data = await request.json()
    email = data.get('email')
    daily_steps = data.get('dailySteps')

    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        user.daily_steps = daily_steps
        db.commit()
        db.refresh(user)
        return {"message": "Daily steps saved successfully", "user": user}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save daily steps: {e}")
    
@user_router.put("/save-heart-rate/")
async def save_heart_rate(request: Request, db: Session = Depends(database.get_db)):
    data = await request.json()
    email = data.get('email')
    heart_rate = data.get('heartRate')

    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        user.heart_rate = heart_rate  # Ganti daily_steps dengan heart_rate
        db.commit()
        db.refresh(user)
        return {"message": "Heart rate saved successfully", "user": user}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save heart rate: {e}")


async def update_user_data(request: schemas.UserUpdate, db: Session):
    logger.info(f"Received request to update user data: {request}")
    try:
        user = db.query(models.User).filter(models.User.email == request.email).first()
        if user:
            if request.name is not None:
                user.name = request.name
            if request.gender is not None:
                user.gender = request.gender
            if request.work is not None:
                user.work = request.work
            if request.date_of_birth is not None:
                user.date_of_birth = request.date_of_birth
            if request.weight is not None:
                user.weight = request.weight
            if request.height is not None:
                user.height = request.height
            db.commit()
            db.refresh(user)
            logger.info("User data updated successfully")
            return {"message": "User data updated successfully", "user": user}
        else:
            logger.warning("User not found")
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error updating user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user data: {e}")

@user_router.get("/user-profile/")
async def get_user_profile(email: str, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


@user_router.put("/user-profile/update")
async def update_user_profile(user_data: schemas.UserProfile, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == user_data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update user fields if provided in user_data
    update_fields = ["name", "gender", "date_of_birth"]
    for field in update_fields:
        if getattr(user_data, field) is not None:
            setattr(user, field, getattr(user_data, field))

    db.commit()
    db.refresh(user)

    return {"message": "User profile updated successfully", "user": user}

class Feedback(BaseModel):
    email: EmailStr  # Validates the format of the email
    feedback: str

@user_router.post("/submit-feedback/")
async def submit_feedback(feedback: Feedback, db: Session = Depends(database.get_db)):
    new_feedback = models.Feedback(
        email=feedback.email,
        feedback=feedback.feedback,
        created_at=datetime.now()
    )
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    return {"message": "Feedback submitted successfully"}
