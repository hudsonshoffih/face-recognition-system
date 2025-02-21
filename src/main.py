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