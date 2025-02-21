from .face_detector import FaceDetector
from ..database.database_manager import DatabaseManager
import face_recognition
import cv2

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = FaceDetector()
        self.db_manager = DatabaseManager()
        self.known_face_encodings = []
        self.known_face_names = []
        
    def initialize(self):
        self.db_manager.connect()
        self.db_manager.setup_tables()
        self.load_known_faces()