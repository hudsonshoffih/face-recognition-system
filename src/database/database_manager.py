import mysql.connector
from mysql.connector import Error
import numpy as np
from .models import Face, Entry
from config.database import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            self.setup_tables()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            
    def setup_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                face_id INT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (face_id) REFERENCES faces(id)
            )
        ''')
        self.conn.commit()

    def add_face(self, name, face_encoding):
        query = "INSERT INTO faces (name, face_encoding) VALUES (%s, %s)"
        encoding_blob = face_encoding.tobytes()
        self.cursor.execute(query, (name, encoding_blob))
        face_id = self.cursor.lastrowid
        self.conn.commit()
        return face_id

    def record_entry(self, face_id):
        query = "INSERT INTO entries (face_id) VALUES (%s)"
        self.cursor.execute(query, (face_id,))
        self.conn.commit()

    def get_all_faces(self):
        self.cursor.execute("SELECT id, name, face_encoding FROM faces")
        faces = []
        for row in self.cursor.fetchall():
            face_id, name, encoding_blob = row
            face_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            faces.append(Face(face_id, name, face_encoding, None))
        return faces