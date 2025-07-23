import sqlite3
from enum import Enum
from werkzeug.security import check_password_hash
import requests


class WorkEnum(str, Enum):
    accountant = 'Accountant'
    doctor = 'Doctor'
    engineer = 'Engineer'
    lawyer = 'Lawyer'
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
                reset_token TEXT,
                synced INTEGER DEFAULT 0
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
                synced INTEGER DEFAULT 0,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                feedback TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced INTEGER DEFAULT 0
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sleep_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                sleep_time TIMESTAMP NOT NULL,
                wake_time TIMESTAMP NOT NULL,
                duration REAL,
                synced INTEGER DEFAULT 0,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
                synced INTEGER DEFAULT 0,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                prediction_result TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced INTEGER DEFAULT 0,
                FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
            );
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS work_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                quality_of_sleep FLOAT DEFAULT NULL,
                physical_activity_level FLOAT DEFAULT NULL,
                stress_level FLOAT DEFAULT NULL,
                work_id INTEGER DEFAULT NULL,
                synced INTEGER DEFAULT 0,       
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
# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
def save_users_to_local(email, hashed_password, work, name=None, gender=None, date_of_birth=None, age=None, weight=None, height=None):
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

# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
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

# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
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

# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
def save_sleep_records_to_local(email, sleep_time, wake_time, duration):
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

# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
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

# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
def save_monthly_prediction_to_local(email, prediction_result):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO monthly_predictions (email, prediction_result)
            VALUES (?, ?)
        ''', (email, prediction_result))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel monthly_predictions.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel monthly_predictions: {e}")
    finally:
        conn.close()
    
# Perbaikan: Tidak menyebutkan kolom id pada insert (karena AUTOINCREMENT)
def save_work_data_to_local(email, quality_of_sleep=None, physical_activity_level=None, stress_level=None, work_id=None):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO work_data (email, quality_of_sleep, physical_activity_level, stress_level, work_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (email, quality_of_sleep, physical_activity_level, stress_level, work_id))
        
        conn.commit()
        print("Data berhasil disimpan ke tabel work_data.")
    except sqlite3.Error as e:
        print(f"Error saat menyimpan data ke tabel work_data: {e}")
    finally:
        conn.close()

# Perbaikan urutan parameter check_password_hash (hash, plain_password)
def authenticate_local(email, password):
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT hashed_password FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        if result:
            hashed_password = result[0]
            if check_password_hash(hashed_password, password):
                return True  # Login berhasil
            else:
                return False  # Password salah
        else:
            return False  # Email tidak ditemukan
    except sqlite3.Error as e:
        print(f"Error saat autentikasi lokal: {e}")
        return False
    finally:
        conn.close()

# Perbaikan: Tambahkan handling error requests agar tidak crash jika server mati
import time

def safe_post(url, json):
    try:
        response = requests.post(url, json=json, timeout=5)
        return response
    except requests.RequestException as e:
        print(f"Gagal mengirim ke server: {e}")
        return None

# Sinkronisasi tabel daily
def sync_daily_to_server(cursor):
    cursor.execute("SELECT * FROM daily WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_daily', json={
            'id': record[0],
            'email': record[2],
            'upper_pressure': record[3],
            'lower_pressure': record[4],
            'daily_steps': record[5],
            'heart_rate': record[6],
            'duration': record[7],
            'prediction_result': record[8]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE daily SET synced = 1 WHERE id = ?", (record[0],))

# Sinkronisasi tabel feedback
def sync_feedback_to_server(cursor):
    cursor.execute("SELECT * FROM feedback WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_feedback', json={
            'id': record[0],
            'email': record[1],
            'feedback': record[2],
            'created_at': record[3]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE feedback SET synced = 1 WHERE id = ?", (record[0],))

# Sinkronisasi tabel sleep_records
def sync_sleep_records_to_server(cursor):
    cursor.execute("SELECT * FROM sleep_records WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_sleep_records', json={
            'id': record[0],
            'email': record[1],
            'sleep_time': record[2],
            'wake_time': record[3],
            'duration': record[4]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE sleep_records SET synced = 1 WHERE id = ?", (record[0],))

# Sinkronisasi tabel weekly_predictions
def sync_weekly_predictions_to_server(cursor):
    cursor.execute("SELECT * FROM weekly_predictions WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_weekly_predictions', json={
            'id': record[0],
            'email': record[1],
            'prediction_result': record[2],
            'created_at': record[3]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE weekly_predictions SET synced = 1 WHERE id = ?", (record[0],))

def sync_monthly_predictions_to_server(cursor):
    cursor.execute("SELECT * FROM monthly_predictions WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_monthly_predictions', json={
            'id': record[0],
            'email': record[1],
            'prediction_result': record[2],
            'created_at': record[3]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE monthly_predictions SET synced = 1 WHERE id = ?", (record[0],))

def sync_users_to_server(cursor):
    cursor.execute("SELECT * FROM users WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_users', json={
            'id': record[0],
            'email': record[1],
            'hashed_password': record[2],
            'created_at': record[3],
            'name': record[4],
            'gender': record[5],
            'work': record[6],
            'date_of_birth': record[7],
            'age': record[8],
            'weight': record[9],
            'height': record[10],
            'upper_pressure': record[11],
            'lower_pressure': record[12],
            'daily_steps': record[13],
            'heart_rate': record[14],
            'reset_token': record[15]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE users SET synced = 1 WHERE id = ?", (record[0],))

def sync_work_data_to_server(cursor):
    cursor.execute("SELECT * FROM work_data WHERE synced = 0")
    unsynced_data = cursor.fetchall()
    for record in unsynced_data:
        response = safe_post('http://103.129.148.233/api/sync_work_data', json={
            'id': record[0],
            'email': record[1],
            'quality_of_sleep': record[2],
            'physical_activity_level': record[3],
            'stress_level': record[4],
            'work_id': record[5]
        })
        if response and response.status_code == 200:
            cursor.execute("UPDATE work_data SET synced = 1 WHERE id = ?", (record[0],))


# Fungsi utama sinkronisasi semua tabel
def sync_data_to_server():
    conn = sqlite3.connect('local_data.db')
    cursor = conn.cursor()
    try:
        sync_daily_to_server(cursor)
        sync_feedback_to_server(cursor)
        sync_sleep_records_to_server(cursor)
        sync_weekly_predictions_to_server(cursor)
        sync_monthly_predictions_to_server(cursor)
        sync_users_to_server(cursor)
        sync_work_data_to_server(cursor)
        
        conn.commit()
        print("Data dari semua tabel berhasil disinkronkan ke server.")
    except sqlite3.Error as e:
        print(f"Error saat sinkronisasi data: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    create_table()  # Membuat tabel jika belum ada
    
    # Contoh penyimpanan user ke database lokal
    save_users_to_local(
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
