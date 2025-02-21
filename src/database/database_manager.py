# src/database/database_manager.py
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

# src/face_recognition/recognition.py
import cv2
import face_recognition
import numpy as np
from ..database.database_manager import DatabaseManager
from .face_detector import FaceDetector

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = FaceDetector()
        self.db_manager = DatabaseManager()
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
    def initialize(self):
        self.db_manager.connect()
        self.load_known_faces()
        
    def load_known_faces(self):
        faces = self.db_manager.get_all_faces()
        for face in faces:
            self.known_face_encodings.append(face.face_encoding)
            self.known_face_names.append(face.name)
            self.known_face_ids.append(face.id)
    
    def register_new_face(self, name, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return False, "No face detected"
            
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_encoding = face_encodings[0]
        
        # Save to database
        face_id = self.db_manager.add_face(name, face_encoding)
        
        # Update local cache
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        self.known_face_ids.append(face_id)
        
        return True, face_id
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            
            if True in matches:
                match_index = matches.index(True)
                name = self.known_face_names[match_index]
                face_id = self.known_face_ids[match_index]
                
                # Record entry
                self.db_manager.record_entry(face_id)
                
                # Draw rectangle and name
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame and draw recognitions
            processed_frame = self.process_frame(frame)
            
            # Display the result
            cv2.imshow('Face Recognition', processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# src/main.py
from face_recognition.recognition import FaceRecognitionSystem
import cv2

def register_face(system):
    name = input("Enter person's name: ")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Registration - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):  # Space key
            success, result = system.register_new_face(name, frame)
            if success:
                print(f"Successfully registered {name}")
            else:
                print(f"Failed to register: {result}")
            break
        elif key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    system = FaceRecognitionSystem()
    system.initialize()
    
    while True:
        print("\nFace Recognition System")
        print("1. Start Recognition")
        print("2. Register New Face")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            system.run_recognition()
        elif choice == '2':
            register_face(system)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()