import cv2
import numpy as np
import os
import pickle
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImageDU
from datetime import datetime

# Email configuration - REPLACE THESE VALUES
EMAIL_ADDRESS = "munirathnamkpm21@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "mynq rget amvs emsp"  # Replace with your app password
RECIPIENT_EMAIL = "munirathnamkpm21@gmail.com"  # Your email for receiving alerts


class FaceRecognitionSystem:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Fixed threshold of 50 (good balance between strict and lenient)
        self.confidence_threshold = 50

        self.known_faces_dir = "known_faces"
        self.model_path = "face_model.yml"
        self.label_map_path = "face_labels.pickle"
        self.alert_images_dir = "alert_images"
        self.label_map = {}

        # Create directories if they don't exist
        for directory in [self.known_faces_dir, self.alert_images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def collect_and_train(self):
        """Collect face data and automatically train the model"""
        person_name = input("Enter person's name: ")
        person_dir = os.path.join(self.known_faces_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        print(f"Collecting face data for {person_name}")
        print("We'll take 20 photos of your face from different angles")
        print("Move your head slightly between photos for better results")

        cap = cv2.VideoCapture(0)
        count = 0
        max_samples = 20

        print("Press SPACE to start capturing...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            # Display instructions
            cv2.putText(frame, "Press SPACE to start capturing", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Detect face and show rectangle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                break
            elif key == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                return

        # Now capture images
        print("Capturing faces... Move your head slightly between captures")
        while count < max_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                cv2.putText(frame, "No face detected - move into frame", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract and save face
                face_img = gray[y:y + h, x:x + w]

                # Ensure minimum size for faces (important for LBPH)
                if face_img.shape[0] >= 100 and face_img.shape[1] >= 100:
                    # Save face image
                    face_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
                    cv2.imwrite(face_path, cv2.resize(face_img, (200, 200)))

                    count += 1

                    # Display progress
                    cv2.putText(frame, f"Progress: {count}/{max_samples}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Move your head slightly", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Add a small delay to avoid duplicate images
                    time.sleep(0.5)
                else:
                    cv2.putText(frame, "Move closer to camera", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Data Collection", frame)

            # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if count >= max_samples:
                break

        cap.release()
        cv2.destroyAllWindows()

        if count > 0:
            print(f"\nCollected {count} samples for {person_name}")
            print("Training the model with all collected faces...")
            self.train_model()
        else:
            print("No faces were collected. Please try again.")

    def train_model(self):
        """Train face recognition model with collected data"""
        faces = []
        labels = []
        current_label = 0

        # Loop through each person's directory
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            # Map label ID to person name
            self.label_map[current_label] = person_name

            # Process each image
            image_count = 0
            for img_name in os.listdir(person_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        # Apply histogram equalization to improve training
                        img = cv2.equalizeHist(img)
                        faces.append(img)
                        labels.append(current_label)
                        image_count += 1

            print(f"Loaded {image_count} images for {person_name}")
            current_label += 1

        # Train the model if we have faces
        if faces and labels:
            print(f"Training model with {len(faces)} total images...")

            # Create a fresh recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()

            # Train with the face data
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(self.model_path)

            # Save the label mapping
            with open(self.label_map_path, "wb") as f:
                pickle.dump(self.label_map, f)

            print(f"Training complete. Model saved to {self.model_path}")
            print(f"Registered users: {list(self.label_map.values())}")
        else:
            print("No face data found. Please collect face data first.")

    def load_model(self):
        """Load trained model and label map"""
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)

            if os.path.exists(self.label_map_path):
                with open(self.label_map_path, "rb") as f:
                    self.label_map = pickle.load(f)
                return True
            else:
                print("Label map not found.")
                return False
        else:
            print("Model file not found. Please train the model first.")
            return False

    def send_email_alert_with_image(self, subject, message, image_path):
        """Send email alert with attached image"""
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            # Attach the image
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(image)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            print("Email alert with image sent successfully")
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            print("Make sure you've updated the email configuration at the top of the file.")
            return False

    def recognize_faces(self):
        """Recognize faces and send alerts with images"""
        if not self.load_model():
            print("Please collect face data and train the model first (Option 1)")
            return

        print("Starting face recognition...")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(0)

        # To prevent multiple alerts for the same person
        last_detection_time = {}
        cooldown_period = 60  # seconds between alerts for same person

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make a copy of the frame for saving
            original_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization for better matching
            gray = cv2.equalizeHist(gray)

            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Extract and preprocess the face
                face = gray[y:y + h, x:x + w]

                # Ensure minimum size for recognition
                if face.shape[0] < 100 or face.shape[1] < 100:
                    cv2.putText(frame, "Face too small", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    continue

                # Resize for consistency with training data
                face_for_recognition = cv2.resize(face, (200, 200))

                # Try to recognize the face
                label_id, confidence = self.recognizer.predict(face_for_recognition)

                # Calculate match confidence (for display purposes)
                match_confidence = 100 - confidence

                current_time = datetime.now()
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_filename = current_time.strftime("%Y%m%d_%H%M%S")

                # Save the captured face with frame
                face_filename = f"{timestamp_filename}.jpg"
                face_path = os.path.join(self.alert_images_dir, face_filename)

                # Draw rectangle around the face for the saved image
                cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(face_path, original_frame)

                # FIXED: Lower confidence = better match in OpenCV LBPH
                if confidence < self.confidence_threshold:
                    # Authorized person
                    name = self.label_map.get(label_id, "Unknown")
                    color = (0, 255, 0)  # Green

                    # Check cooldown before sending alert
                    if name not in last_detection_time or \
                            (current_time - last_detection_time[name]).total_seconds() > cooldown_period:
                        subject = f"Security Alert: Authorized Access by {name}"
                        message = f"""
                        Authorized access detected:
                        User: {name}
                        Time: {timestamp}
                        Confidence: {match_confidence:.2f}%

                        Please see the attached image for verification.
                        """
                        self.send_email_alert_with_image(subject, message, face_path)
                        last_detection_time[name] = current_time
                else:
                    # Intruder (unrecognized face)
                    name = "Intruder"
                    color = (0, 0, 255)  # Red

                    # Check cooldown before sending alert
                    if name not in last_detection_time or \
                            (current_time - last_detection_time[name]).total_seconds() > cooldown_period:
                        subject = "SECURITY ALERT: Unauthorized Access Detected!"
                        message = f"""
                        WARNING: Unauthorized access detected!
                        Time: {timestamp}
                        Location: Primary camera

                        Please see the attached image and take appropriate action immediately.
                        """
                        self.send_email_alert_with_image(subject, message, face_path)
                        last_detection_time[name] = current_time

                # Display recognition info
                label = f"{name} ({match_confidence:.2f}%)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    system = FaceRecognitionSystem()

    while True:
        print("\nFace Recognition Security System")
        print("1. Collect face data and train model")
        print("2. Start face recognition")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            system.collect_and_train()

        elif choice == '2':
            system.recognize_faces()

        elif choice == '3':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()