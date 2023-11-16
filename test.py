import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import socket

model = load_model('emotion_detection_model.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

cap = cv2.VideoCapture(0)

# Create a folder to store images if it doesn't exist
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Get local IP address
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        input_face = np.expand_dims(normalized_face, axis=0)

        # Make a prediction for the face
        predictions = model.predict(input_face)
        emotion_index = np.argmax(predictions)
        predicted_emotion = emotions[emotion_index]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the predicted emotion
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Save the image if the predicted emotion is "surprised"
        if predicted_emotion == 'surprised':
            date_format = time.strftime("%Y-%m-%d")
            time_format = time.strftime("%H-%M-%S")
            timestamp = f'{date_format}_{time_format}'
            image_name = f'{data_folder}/pic_{timestamp}_{ip_address}.jpg'
            cv2.imwrite(image_name, frame)
            print("Image Captured Successfully.")

    # Display the frame with rectangles and predicted emotions
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
