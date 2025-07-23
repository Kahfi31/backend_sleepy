import logging
from fastapi import FastAPI, Depends, HTTPException, Form, Query, APIRouter
from sqlalchemy import extract
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date
from . import models, schemas, database
from pydantic import BaseModel, EmailStr
from sqlalchemy import func
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ML models
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

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    raise RuntimeError("Model loading failed")

# Database initialization
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
router = APIRouter()

predict_router = APIRouter(prefix="/predict", tags=["Prediction"])

class PredictRequest(BaseModel):
    email: EmailStr 

@predict_router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(database.get_db)):
    try:
        # Extract email from the request body
        email = request.email

        user_data = db.query(models.User).filter(models.User.email == email).first()
        if not user_data:
            logging.error("User not found")
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch data from the relevant tables using email
        sleep_record = db.query(models.SleepRecord).filter(models.SleepRecord.email == email).first()
        work_data = db.query(models.Work).filter(models.Work.email == email).first()

        if not sleep_record:
            logging.error("Sleep record not found for email: " + email)
            raise HTTPException(status_code=404, detail="Incomplete data for prediction (missing sleep record)")
        if not work_data:
            logging.error("Work data not found for email: " + email)
            raise HTTPException(status_code=404, detail="Incomplete data for prediction (missing work data)")

        # Extract user data from the tables
        age = user_data.age
        gender = user_data.gender  # 0 for female, 1 for male
        occupation = work_data.work_id  # Encoded occupation ID
        bmi_category = 0  # Set a default value or obtain it from another source
        quality_of_sleep = work_data.quality_of_sleep
        physical_activity_level = work_data.physical_activity_level
        stress_level = work_data.stress_level
        heart_rate = user_data.heart_rate
        daily_step = user_data.daily_steps
        systolic = user_data.upper_pressure
        diastolic = user_data.lower_pressure
        sleep_duration = sleep_record.duration
        
        # Example additional feature for a total of 12 features
        additional_feature = 0  # Replace with the appropriate feature if necessary

        # Prepare numerical features for scaling
        numerical_features = [
            age, sleep_duration, quality_of_sleep, physical_activity_level,
            stress_level, heart_rate, daily_step, systolic, diastolic, additional_feature
        ]

        # Initialize complete_features with zeros (ensure it has 12 features)
        complete_features = np.zeros((1, 12))  # Ensure total of 12 features

        # Insert the first 10 numerical features into complete_features
        complete_features[0, :10] = numerical_features  # Update to accommodate 10 features

        # Scale numerical features
        scaled_features = scaler.transform(complete_features).flatten()

        # Construct the final features list for the model
        features = np.array([
            gender, 
            scaled_features[0], 
            occupation,  
            scaled_features[1],  
            scaled_features[2],  
            scaled_features[3],  
            scaled_features[4],  
            bmi_category,  
            scaled_features[5],  
            scaled_features[6],  
            scaled_features[7], 
            scaled_features[8]  
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]  # Get the prediction result (0, 1, or 2)

        # Map prediction to disorder type
        if prediction == 0:
            result = 'Insomnia'
        elif prediction == 1:
            result = 'Normal'
        elif prediction == 2:
            result = 'Sleep Apnea'

        # Check if there's already an entry for today's date
        today = date.today()
        daily_record = db.query(models.Daily).filter(
            models.Daily.email == email,
            models.Daily.date == today
        ).first()

        if daily_record:
            # If an entry exists, update it
            logging.info(f"Updating prediction result for {email} on {today} to {int(prediction)}")
            daily_record.prediction_result = int(prediction)
        else:
            # If no entry exists, create a new one
            logging.info(f"Inserting new daily record for {email} on {today} with prediction {int(prediction)}")
            new_daily_record = models.Daily(
                email=email,
                date=today,
                upper_pressure=systolic,
                lower_pressure=diastolic,
                daily_steps=daily_step,
                heart_rate=heart_rate,
                duration=sleep_duration,
                prediction_result=int(prediction)
            )
            db.add(new_daily_record)
            db.flush()

        # Log before committing
        logging.info("Committing the prediction result to the database")
        db.commit()

        # Verify commit success
        logging.info("Commit successful")

        return {"prediction": result}

    except ValueError as e:
        logging.error(f"ValueError during prediction: {e}")
        raise HTTPException(status_code=400, detail="Data format issue.")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
class SavePredictionRequest(BaseModel):
    email: str
    prediction_result: int
    
@predict_router.post("/save_prediction")
def save_prediction(request: SavePredictionRequest, db: Session = Depends(database.get_db)):
    try:
        today = date.today()
        daily_record = db.query(models.Daily).filter(
            models.Daily.email == request.email,
            models.Daily.date == today
        ).first()

        if daily_record:
            daily_record.prediction_result = request.prediction_result
            db.commit()
            return {"message": "Prediction updated successfully"}
        else:
            new_daily_record = models.Daily(
                email=request.email,
                date=today,
                prediction_result=request.prediction_result
            )
            db.add(new_daily_record)
            db.commit()
            return {"message": "Prediction saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
class WeeklyPredictRequest(BaseModel):
    email: str
    
@predict_router.post("/weekly_predict")
def weekly_predict(request: WeeklyPredictRequest, db: Session = Depends(database.get_db)):
    try:
        email = request.email
        today = date.today()
        seven_days_ago = today - timedelta(days=7)

        # Ambil data harian selama seminggu terakhir untuk email tertentu
        weekly_data = db.query(models.Daily).filter(
            models.Daily.email == email,
            models.Daily.date >= seven_days_ago,
            models.Daily.date <= today
        ).all()

        if not weekly_data:
            raise HTTPException(status_code=404, detail="Tidak ada data harian untuk seminggu terakhir.")

        # Hitung jumlah kemunculan setiap jenis gangguan tidur
        normal_count = sum(1 for record in weekly_data if record.prediction_result == 1)
        insomnia_count = sum(1 for record in weekly_data if record.prediction_result == 0)
        sleep_apnea_count = sum(1 for record in weekly_data if record.prediction_result == 2)

        # Total hari data yang tersedia
        total_days = len(weekly_data)

        # Tentukan hasil prediksi mingguan
        if normal_count > (insomnia_count + sleep_apnea_count):
            result = 'Normal'
        else:
            if insomnia_count > sleep_apnea_count:
                result = 'Insomnia'
            elif sleep_apnea_count > insomnia_count:
                result = 'Sleep Apnea'
            else:  # insomnia_count == sleep_apnea_count
                # Pilih yang lebih parah jika jumlah sama
                result = 'Sleep Apnea'  # Asumsi Sleep Apnea lebih parah daripada Insomnia

        return {"weekly_prediction": result}

    except Exception as e:
        logging.error(f"Weekly prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat melakukan prediksi mingguan: {str(e)}")

prediction_mapping = {
    0: "Insomnia",
    1: "Normal",
    2: "Sleep Apnea"
}

class SavePredictionRequestWeek(BaseModel):
    email: str
    prediction_result: int  # Mengharapkan input berupa integer

@predict_router.post("/save_prediction_week")
def save_prediction(request: SavePredictionRequestWeek, db: Session = Depends(database.get_db)):
    try:
        # Ambil email dan hasil prediksi dari request
        email = request.email
        prediction_result = request.prediction_result

        # Konversi integer ke string berdasarkan mapping
        if prediction_result in prediction_mapping:
            prediction_enum_value = prediction_mapping[prediction_result]
        else:
            raise HTTPException(status_code=400, detail="Invalid prediction result")

        # Simpan ke database
        prediction = models.WeeklyPrediction(
            email=email,
            prediction_result=prediction_enum_value  # Simpan sebagai string enum
        )
        db.add(prediction)
        db.commit()

        return {"message": "Prediction saved successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving prediction: {str(e)}")
    
class MonthlyPredictRequest(BaseModel):
    email: str

@predict_router.post("/monthly_predict")
def monthly_predict(request: MonthlyPredictRequest, db: Session = Depends(database.get_db)):
    try:
        email = request.email
        today = date.today()
        thirty_days_ago = today - timedelta(days=30)

        # Ambil data harian selama 30 hari terakhir untuk email tertentu
        monthly_data = db.query(models.Daily).filter(
            models.Daily.email == email,
            func.date(models.Daily.date) >= thirty_days_ago,
            func.date(models.Daily.date) <= today
        ).all()

        if not monthly_data:
            raise HTTPException(status_code=404, detail="Tidak ada data harian untuk 30 hari terakhir.")

        # Hitung jumlah kemunculan setiap jenis gangguan tidur
        normal_count = sum(1 for record in monthly_data if record.prediction_result == 1)
        insomnia_count = sum(1 for record in monthly_data if record.prediction_result == 0)
        sleep_apnea_count = sum(1 for record in monthly_data if record.prediction_result == 2)

        # Debugging: cek berapa banyak data yang terambil dan hitungan masing-masing
        logging.info(f"Normal Count: {normal_count}")
        logging.info(f"Insomnia Count: {insomnia_count}")
        logging.info(f"Sleep Apnea Count: {sleep_apnea_count}")

        # Tentukan hasil prediksi bulanan
        if normal_count > (insomnia_count + sleep_apnea_count):
            result = 'Normal'
        else:
            if insomnia_count > sleep_apnea_count:
                result = 'Insomnia'
            elif sleep_apnea_count > insomnia_count:
                result = 'Sleep Apnea'
            else:  # insomnia_count == sleep_apnea_count
                # Pilih yang lebih parah jika jumlah sama
                result = 'Sleep Apnea'  # Asumsi Sleep Apnea lebih parah daripada Insomnia

        return {"monthly_prediction": result}

    except Exception as e:
        logging.error(f"Monthly prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat melakukan prediksi bulanan: {str(e)}")


prediction_mapping = {
    0: "Insomnia",
    1: "Normal",
    2: "Sleep Apnea"
}

class SavePredictionRequestMonth(BaseModel):
    email: str
    prediction_result: int  # Mengharapkan input berupa integer

@predict_router.post("/save_prediction_month")
def save_prediction_month(request: SavePredictionRequestMonth, db: Session = Depends(database.get_db)):
    try:
        # Ambil email dan hasil prediksi dari request
        email = request.email
        prediction_result = request.prediction_result

        # Konversi integer ke string berdasarkan mapping
        if prediction_result in prediction_mapping:
            prediction_enum_value = prediction_mapping[prediction_result]
        else:
            raise HTTPException(status_code=400, detail="Invalid prediction result")

        # Simpan ke database
        prediction = models.MonthlyPrediction(
            email=email,
            prediction_result=prediction_enum_value  # Simpan sebagai string enum
        )
        db.add(prediction)
        db.commit()

        return {"message": "Monthly prediction saved successfully"}

    except Exception as e:
        db.rollback()
        print(f"Exception occurred: {str(e)}")  # Debug log untuk melihat kesalahan
        raise HTTPException(status_code=500, detail=f"Error saving monthly prediction: {str(e)}")
    
@predict_router.get("/get-weekly-sleep-data/{email}")
async def get_weekly_sleep_data(email: str, start_date: str, end_date: str, db: Session = Depends(database.get_db)):
    # Convert string dates to datetime objects
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=2)

    # Retrieve sleep records for the user between the start and end date
    sleep_records = db.query(models.SleepRecord).filter(
        models.SleepRecord.email == email,
        models.SleepRecord.sleep_time >= start_date_obj,
        models.SleepRecord.wake_time <= end_date_obj
    ).order_by(models.SleepRecord.sleep_time.desc()).all()


    if not sleep_records:
        raise HTTPException(status_code=404, detail="No sleep records found for the week")

    # Filter to keep only the latest record per day
    latest_records_per_day = {}
    for record in sleep_records:
        record_date = record.sleep_time.date()
        # Only keep the latest record for each date
        if record_date not in latest_records_per_day:
            latest_records_per_day[record_date] = record

    # Initialize dictionaries to store sleep durations, start times, and wake times for each day
    daily_sleep_durations = {i: timedelta() for i in range(7)}  # {0: Monday, ..., 6: Sunday}
    daily_sleep_start_times = {i: [] for i in range(7)}  # Store sleep start times for each day
    daily_wake_times = {i: [] for i in range(7)}  # Store wake times for each day

    for record in latest_records_per_day.values():
        # Handle cross-midnight sleep records
        if record.wake_time < record.sleep_time:
            record.wake_time += timedelta(days=1)

        duration = record.wake_time - record.sleep_time

        # Calculate the day of the week for the sleep time (0=Monday, 6=Sunday)
        day_of_week = record.sleep_time.weekday()
        daily_sleep_durations[day_of_week] += duration

        # Store the sleep start time for the day
        daily_sleep_start_times[day_of_week].append(record.sleep_time.strftime("%H:%M"))
        
        # Store the wake time for the day
        daily_wake_times[day_of_week].append(record.wake_time.strftime("%H:%M"))

    # Convert timedelta to hours for each day
    daily_sleep_durations_hours = [round(daily_sleep_durations[i].total_seconds() / 3600, 2) for i in range(7)]

    # Calculate total duration as the sum of daily sleep durations
    total_duration = sum(daily_sleep_durations_hours)

    # Calculate averages
    avg_duration = total_duration / len(latest_records_per_day)
    avg_sleep_time = (sum((timedelta(hours=int(time[:2]), minutes=int(time[3:])) 
                          for times in daily_sleep_start_times.values() for time in times), timedelta()) 
                      / len(latest_records_per_day))
    avg_wake_time = (sum((timedelta(hours=int(time[:2]), minutes=int(time[3:])) 
                         for times in daily_wake_times.values() for time in times), timedelta()) 
                     / len(latest_records_per_day))

    return {
        "daily_sleep_durations": daily_sleep_durations_hours,
        "daily_sleep_start_times": daily_sleep_start_times,  # Field with sleep start times
        "daily_wake_times": daily_wake_times,  # New field with wake times
        "avg_duration": f"{int(avg_duration)} jam {int((avg_duration * 60) % 60)} menit",
        "avg_sleep_time": (datetime.min + avg_sleep_time).strftime("%H:%M"),
        "avg_wake_time": (datetime.min + avg_wake_time).strftime("%H:%M"),
        "total_duration": f"{int(total_duration)} jam {int((total_duration * 60) % 60)} menit"
    }

@predict_router.get("/get-monthly-sleep-data/{email}")
async def get_monthly_sleep_data(email: str, month: str, year: int, db: Session = Depends(database.get_db)):
    # Calculate the start and end dates for the month
    start_date_obj = datetime(year, int(month), 1)
    next_month = start_date_obj.replace(day=28) + timedelta(days=4)  # This will always jump to the next month
    end_date_obj = next_month - timedelta(days=next_month.day)

    # Retrieve sleep records for the user between the start and end dates
    sleep_records = db.query(models.SleepRecord).filter(
        models.SleepRecord.email == email,
        models.SleepRecord.sleep_time >= start_date_obj,
        models.SleepRecord.wake_time < end_date_obj + timedelta(days=1)  # Include the entire end day
    ).order_by(models.SleepRecord.sleep_time.desc()).all()  # Sort by sleep_time descending

    if not sleep_records:
        raise HTTPException(status_code=404, detail="No sleep records found for the month")

    # Filter to keep only the latest record per day
    latest_records_per_day = {}
    for record in sleep_records:
        record_date = record.sleep_time.date()
        if record_date not in latest_records_per_day:
            latest_records_per_day[record_date] = record

    # Initialize dictionaries to store weekly and daily sleep durations, start times, and wake times
    weekly_sleep_durations = {i: timedelta() for i in range(4)}
    weekly_sleep_start_times = {i: [] for i in range(4)}
    weekly_wake_times = {i: [] for i in range(4)}

    # Initialize daily sleep duration list
    days_in_month = (end_date_obj - start_date_obj).days + 1
    daily_sleep_durations = [0.0] * days_in_month  # Initialize daily sleep durations with 0

    for record in latest_records_per_day.values():
        # Handle cross-midnight sleep records
        if record.wake_time < record.sleep_time:
            record.wake_time += timedelta(days=1)

        duration = record.wake_time - record.sleep_time

        # Calculate the day of the month for the sleep time
        day_of_month = (record.sleep_time - start_date_obj).days
        daily_sleep_durations[day_of_month] = round(duration.total_seconds() / 3600, 2)  # Convert to hours

        # Calculate the week of the month for the sleep time
        week_of_month = day_of_month // 7
        if week_of_month > 3:
            week_of_month = 3

        weekly_sleep_durations[week_of_month] += duration
        weekly_sleep_start_times[week_of_month].append(record.sleep_time.strftime("%H:%M"))
        weekly_wake_times[week_of_month].append(record.wake_time.strftime("%H:%M"))

    weekly_sleep_durations_hours = [round(weekly_sleep_durations[i].total_seconds() / 3600, 2) for i in range(4)]
    total_duration = sum(weekly_sleep_durations_hours)

    avg_duration = total_duration / len(latest_records_per_day)

    # Adjust sleep times for proper average calculation
    sleep_times_in_minutes = []
    for times in weekly_sleep_start_times.values():
        for time in times:
            hours, minutes = map(int, time.split(':'))
            if hours < 12:
                hours += 24  # Adjust early morning times past midnight to make calculations accurate
            sleep_times_in_minutes.append(hours * 60 + minutes)

    avg_sleep_minutes = sum(sleep_times_in_minutes) / len(sleep_times_in_minutes)
    avg_sleep_hours = int(avg_sleep_minutes // 60)
    avg_sleep_minutes = int(avg_sleep_minutes % 60)

    wake_times_in_minutes = []
    for times in weekly_wake_times.values():
        for time in times:
            hours, minutes = map(int, time.split(':'))
            wake_times_in_minutes.append(hours * 60 + minutes)

    avg_wake_minutes = sum(wake_times_in_minutes) / len(wake_times_in_minutes)
    avg_wake_hours = int(avg_wake_minutes // 60)
    avg_wake_minutes = int(avg_wake_minutes % 60)

    return {
        "weekly_sleep_durations": weekly_sleep_durations_hours,
        "weekly_sleep_start_times": weekly_sleep_start_times,
        "weekly_wake_times": weekly_wake_times,
        "daily_sleep_durations": daily_sleep_durations,  # Send daily data to the frontend
        "avg_duration": f"{int(avg_duration)} jam {int((avg_duration * 60) % 60)} menit",
        "avg_sleep_time": f"{avg_sleep_hours:02d}:{avg_sleep_minutes:02d}",
        "avg_wake_time": f"{avg_wake_hours:02d}:{avg_wake_minutes:02d}",
        "total_duration": f"{int(total_duration)} jam {int((total_duration * 60) % 60)} menit"
    }

@predict_router.post("/save-sleep-record/")
async def save_sleep_record(sleep_data: schemas.SleepData, db: Session = Depends(database.get_db)):
    # Find the user by email
    user = db.query(models.User).filter(models.User.email == sleep_data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate the sleep duration in hours (float)
    sleep_time = sleep_data.sleep_time
    wake_time = sleep_data.wake_time

    if sleep_time >= wake_time:
        wake_time += timedelta(days=1)  # Handle crossing midnight

    duration = (wake_time - sleep_time).total_seconds() / 3600  # Duration in hours

    # Check for existing sleep record for the same date
    existing_record = db.query(models.SleepRecord).filter(
        models.SleepRecord.email == user.email,
        extract('year', models.SleepRecord.sleep_time) == sleep_time.year,
        extract('month', models.SleepRecord.sleep_time) == sleep_time.month,
        extract('day', models.SleepRecord.sleep_time) == sleep_time.day
    ).first()

    if existing_record:
        # Update existing record with the new data only if it's more recent
        if sleep_time > existing_record.sleep_time:
            existing_record.sleep_time = sleep_time
            existing_record.wake_time = wake_time
            existing_record.duration = duration
            db.commit()
            db.refresh(existing_record)
            return {"message": "Sleep record updated successfully", "sleep_record": existing_record}
        else:
            return {"message": "Older sleep record ignored; existing record is more recent", "sleep_record": existing_record}
    else:
        # Create a new SleepRecord associated with the user's email
        new_sleep_record = models.SleepRecord(
            email=user.email,  
            sleep_time=sleep_time,
            wake_time=wake_time,
            duration=duration  
        )

        db.add(new_sleep_record)
        db.commit()
        db.refresh(new_sleep_record)

        return {"message": "Sleep record saved successfully", "sleep_record": new_sleep_record}

@predict_router.get("/get-sleep-records/{email}")
async def get_sleep_records(email: str, db: Session = Depends(database.get_db)):
    # Retrieve the user based on the email
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch all sleep records associated with the user, ordered by sleep_time descending
    sleep_records = db.query(models.SleepRecord).filter(
        models.SleepRecord.email == email
    ).order_by(models.SleepRecord.sleep_time.desc()).all()

    if not sleep_records:
        raise HTTPException(status_code=404, detail="No sleep records found")

    # Use a dictionary to store the latest record for each date
    latest_records = {}
    for record in sleep_records:
        record_date = record.sleep_time.date()
        # Only keep the latest record for each date
        if record_date not in latest_records:
            latest_records[record_date] = record

    # Prepare the response data
    response_data = []
    for record in latest_records.values():
        # Calculate duration using DateTime difference
        duration = record.wake_time - record.sleep_time
        # Format the duration to hours and minutes
        formatted_duration = f"{duration.seconds // 3600} jam {duration.seconds % 3600 // 60} menit"
        # Format sleep and wake times
        formatted_time = f"{record.sleep_time.strftime('%H:%M')} - {record.wake_time.strftime('%H:%M')}"
        # Format the date
        formatted_date = record.sleep_time.strftime('%d %B %Y')  # Use sleep_time's date
        # Add the data to the response
        response_data.append({
            "date": formatted_date,
            "duration": formatted_duration,
            "time": formatted_time
        })

    return response_data
