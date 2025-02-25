import cv2
import numpy as np
import sqlite3
from datetime import datetime
import face_recognition as fr
import os
from typing import Tuple, List, Optional

class FaceRecognitionSystem:
    def __init__(self, db_path: str = 'face_attendance.db'):
        """Initialize the face recognition system."""
        self.conn = self._setup_database(db_path)
        self.known_face_encodings = []
        self.known_face_ids = []
        self.load_known_faces()
        
    def _setup_database(self, db_path: str) -> sqlite3.Connection:
        """Set up SQLite database with required tables."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create people table
        c.execute('''CREATE TABLE IF NOT EXISTS people
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     team_name TEXT,
                     face_encoding BLOB,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Create entries table
        c.execute('''CREATE TABLE IF NOT EXISTS entries
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     person_id INTEGER,
                     entry_type TEXT,
                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (person_id) REFERENCES people(id))''')
        
        conn.commit()
        return conn

    def load_known_faces(self):
        """Load known face encodings from database."""
        c = self.conn.cursor()
        c.execute("SELECT id, face_encoding FROM people")
        rows = c.fetchall()
        
        self.known_face_encodings = []
        self.known_face_ids = []
        
        for row in rows:
            person_id, face_encoding_blob = row
            if face_encoding_blob is not None:
                face_encoding = np.frombuffer(face_encoding_blob, dtype=np.float64)
                self.known_face_encodings.append(face_encoding)
                self.known_face_ids.append(person_id)

    def _get_camera(self) -> Tuple[cv2.VideoCapture, bool]:
        """Initialize camera with proper settings."""
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera, camera.isOpened()

    def register_new_face(self, name: str, team_name: str):
        """Register a new face in the system."""
        camera, success = self._get_camera()
        if not success:
            print("Error: Could not open camera")
            return False

        print("Position face in frame and press SPACE to capture. Press ESC to cancel.")
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            cv2.imshow('Registration - Press SPACE when ready', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                try:
                    # Detect faces in frame
                    face_locations = fr.face_locations(rgb_frame, model="hog")
                    
                    if len(face_locations) == 1:
                        # Get face encoding
                        face_encodings = fr.face_encodings(rgb_frame, face_locations)
                        
                        if face_encodings:
                            # Store in database
                            c = self.conn.cursor()
                            c.execute("""
                                INSERT INTO people (name, team_name, face_encoding)
                                VALUES (?, ?, ?)
                            """, (name, team_name, face_encodings[0].tobytes()))
                            self.conn.commit()
                            
                            # Update local cache
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_ids.append(c.lastrowid)
                            
                            print(f"Successfully registered {name}")
                            break
                        else:
                            print("Could not encode face. Please try again.")
                    else:
                        print(f"Found {len(face_locations)} faces. Please ensure exactly one face is visible.")
                
                except Exception as e:
                    print(f"Error during registration: {str(e)}")
            
            elif key == 27:  # ESC key
                print("Registration cancelled")
                break

        camera.release()
        cv2.destroyAllWindows()

    def recognize_and_track(self):
        """Start face recognition and tracking."""
        camera, success = self._get_camera()
        if not success:
            print("Error: Could not open camera")
            return False

        print("Starting face recognition... Press ESC to stop.")
        while True:
            ret, frame = camera.read()
            if not ret:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Detect faces
                face_locations = fr.face_locations(rgb_frame, model="hog")
                face_encodings = fr.face_encodings(rgb_frame, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with known faces
                    matches = fr.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    
                    if True in matches:
                        match_index = matches.index(True)
                        person_id = self.known_face_ids[match_index]
                        
                        # Record entry
                        c = self.conn.cursor()
                        c.execute("""
                            INSERT INTO entries (person_id, entry_type)
                            VALUES (?, ?)
                        """, (person_id, 'check_in'))
                        
                        # Get person details
                        c.execute("SELECT name, team_name FROM people WHERE id=?", (person_id,))
                        name, team_name = c.fetchone()
                        self.conn.commit()
                        
                        # Draw box and label
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                  f"{name} - {team_name}", 
                                  (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0, 255, 0),
                                  2)
            
            except Exception as e:
                print(f"Error during recognition: {str(e)}")
                continue
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        camera.release()
        cv2.destroyAllWindows()

    def view_entry_counts(self):
        """Display entry counts for all registered people."""
        c = self.conn.cursor()
        c.execute("""
            SELECT 
                p.name,
                p.team_name,
                COUNT(e.id) as entry_count,
                MAX(e.timestamp) as last_entry
            FROM people p
            LEFT JOIN entries e ON p.id = e.person_id
            GROUP BY p.id, p.name, p.team_name
            ORDER BY p.name
        """)
        
        rows = c.fetchall()
        if not rows:
            print("No registered people found.")
            return
        
        print("\nEntry Counts:")
        print("-" * 70)
        print(f"{'Name':<20} {'Team':<15} {'Entries':<10} {'Last Entry':<20}")
        print("-" * 70)
        
        for row in rows:
            name, team, count, last_entry = row
            last_entry = last_entry if last_entry else "Never"
            print(f"{name:<20} {team:<15} {count:<10} {last_entry:<20}")

def main():
    print("Initializing Face Recognition System...")
    try:
        system = FaceRecognitionSystem()
        
        while True:
            print("\nFace Recognition System Menu:")
            print("1. Register new face")
            print("2. Start recognition system")
            print("3. View entry counts")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ")
            
            if choice == '1':
                name = input("Enter name: ")
                team_name = input("Enter team name: ")
                system.register_new_face(name, team_name)
            
            elif choice == '2':
                system.recognize_and_track()
            
            elif choice == '3':
                system.view_entry_counts()
            
            elif choice == '4':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nPlease ensure all required libraries are properly installed:")
        print("1. pip install opencv-python")
        print("2. pip install face-recognition")
        print("3. pip install numpy")

if __name__ == "__main__":
    main()

    # conda activate face_env