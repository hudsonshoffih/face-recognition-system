import cv2
import face_recognition
import numpy as np
#from ..database.database_manager import DatabaseManager
from src.database.database_manager import DatabaseManager
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