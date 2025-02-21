from face_recognition.recognition import FaceRecognitionSystem

def main():
    system = FaceRecognitionSystem()
    system.initialize()
    system.run_recognition()

if __name__ == "__main__":
    main()