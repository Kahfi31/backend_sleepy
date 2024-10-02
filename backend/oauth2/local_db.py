import sqlite3
from enum import Enum


class WorkEnum(str, Enum):
    accountant = 'Accountant'
    doctor = 'Doctor'
    engineer = 'Engineer'
    lawyer = 'Other'
    manager = 'Manager'
    nurse = 'Nurse'
    sales_representative = 'Sales Representative'
    sales_person = 'Sales Person'
    scientist = 'Scientist'
    software_engineer = 'Software Engineer'
    teacher = 'Teacher'

class PredictionEnum(str, Enum) :
    insomnia = 'Insomnia'
    normal = 'Normal'
    sleep_apnea = 'Sleep Apnea'


# Fungsi untuk membuat tabel jika belum ada
def create_table():
    conn = sqlite3.connect('local_data.db')
    conn.execute("PRAGMA foreign_keys = ON")  # Mengaktifkan foreign key di SQLite
    cursor = conn.cursor()
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                name TEXT,
                gender INTEGER,
                work TEXT,
                date_of_birth DATE,
                age INTEGER,
                weight REAL,
                height REAL,
                upper_pressure INTEGER,
                lower_pressure INTEGER,
                daily_steps INTEGER,
                heart_rate INTEGER,
                reset_token TEXT
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                email TEXT NOT NULL,
                date DATE NOT NULL,
                upper_pressure INTEGER,
                lower_pressure INTEGER,
                daily_steps INTEGER,
                heart_rate INTEGER,
                duration REAL NOT NULL,
                prediction_result TEXT NOT NULL,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                feedback TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                sleep_time TIMESTAMP NOT NULL,
                wake_time TIMESTAMP NOT NULL,
                duration REAL,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        # Contoh pembuatan trigger di SQLite
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_or_insert_daily_from_sleep_records
            AFTER INSERT ON sleep_records
            FOR EACH ROW
            BEGIN
                INSERT INTO daily (email, date, duration)
                VALUES (NEW.email, DATE(NEW.sleep_time), NEW.duration)
                ON CONFLICT(email, date) DO UPDATE
                SET duration = NEW.duration;
            END;
        ''')

        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saat membuat tabel: {e}")
    finally:
        conn.close()

# Fungsi untuk menyimpan data ke database lokal
def save_user_to_local(email, hashed_password, work, name=None, gender=None, date_of_birth=None, age=None, weight=None, height=None):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        # Validasi nilai work sebelum disimpan
        if work not in WorkEnum._value2member_map_:
            raise ValueError(f"Invalid work value: {work}")

        cursor.execute('''
            INSERT INTO users (email, hashed_password, work, name, gender, date_of_birth, age, weight, height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (email, hashed_password, work, name, gender, date_of_birth, age, weight, height))
        
        conn.commit()
        print("Data berhasil disimpan ke SQLite.")
    except ValueError as ve:
        print(f"Error validasi work: {ve}")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data: {e}")
    finally:
        conn.close()

def save_daily_to_local(email, upper_pressure, lower_pressure, daily_steps, heart_rate, duration, prediction_result):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO daily (email, upper_pressure, lower_pressure, daily_steps, heart_rate, duration, prediction_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (email, upper_pressure, lower_pressure, daily_steps, heart_rate, duration, prediction_result))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel daily.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel daily: {e}")
    finally:
        conn.close()

def save_feedback_to_local(email, feedback):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO feedback (email, feedback)
            VALUES (?, ?)
        ''', (email, feedback))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel feedback.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel feedback: {e}")
    finally:
        conn.close()

def save_sleep_record_to_local(email, sleep_time, wake_time, duration):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO sleep_records (email, sleep_time, wake_time, duration)
            VALUES (?, ?, ?, ?)
        ''', (email, sleep_time, wake_time, duration))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel sleep_records.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel sleep_records: {e}")
    finally:
        conn.close()

def save_weekly_prediction_to_local(email, prediction_result):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO weekly_predictions (email, prediction_result)
            VALUES (?, ?)
        ''', (email, prediction_result))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel weekly_predictions.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel weekly_predictions: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    create_table()  # Membuat tabel jika belum ada
    
    # Contoh penyimpanan user ke database lokal
    save_user_to_local(
        email="example@example.com", 
        hashed_password="hashedpassword123", 
        work="Software Engineer",  # Menggunakan salah satu nilai yang ada di WorkEnum
        name="John Doe",
        gender=1,
        date_of_birth="1990-01-01",
        age=34,
        weight=70.5,
        height=175.0
    )
