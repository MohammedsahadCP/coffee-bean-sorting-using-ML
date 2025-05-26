import cv2
import time
import serial
from model import CoffeeBeanModel  # Importing YOLO model class

# Initialize YOLO model
model = CoffeeBeanModel('C:\\Users\\moham\\Desktop\\group16proj\\best11.pt')

# Open Serial connection to Arduino (update COM port)
arduino = serial.Serial('COM3', 9600, timeout=1)  
time.sleep(2)  # Allow time for serial connection

# Open mobile camera (IP Webcam)
cap = cv2.VideoCapture('http://100.72.87.100:8080/video')

if not cap.isOpened():
    raise ConnectionError("❌ Error opening video stream")

def preprocess_frame(frame):
    """ Crop the frame to 1/3 of its original size while keeping the center intact """
    h, w, _ = frame.shape
    crop_size = min(h, w) // 3  # Crop size is 1/3rd of the smaller dimension

    # Calculate cropping coordinates (centered)
    center_x, center_y = w // 2, h // 2
    x1 = max(center_x - crop_size // 2, 0)
    y1 = max(center_y - crop_size // 2, 0)
    x2 = min(center_x + crop_size // 2, w)
    y2 = min(center_y + crop_size // 2, h)

    cropped = frame[y1:y2, x1:x2]  # Crop the region
    resized = cv2.resize(cropped, (224, 224))  # Resize to standard input size (224x224)

    return resized  # Return the preprocessed image

while True:
    # **1. Wait for stepper motor to stop moving (idle)**
    print("🔄 Waiting for stepper motor to finish 90° rotation...")
    arduino.write(b"step_done")  # Request status from Arduino
    while True:
        status = arduino.readline().decode().strip()  # Read response from Arduino
        if status == "idle":  # Check if the motor is idle
            print("✅ Stepper motor is idle. Capturing frame...")
            break
        time.sleep(0.1)  # Small delay to avoid flooding Arduino with requests

    # **2. Capture a new frame only when motor is idle**
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        continue  # Skip processing if frame capture failed

    # **3. Preprocess the frame**
    processed_frame = preprocess_frame(frame)

    # **4. Save image temporarily for model inference**
    image_path = "detected_image.jpg"
    cv2.imwrite(image_path, processed_frame)

    # **5. Run YOLO model prediction**
    predicted_class = model.predict_class(image_path)+1
    print(f"✅ Predicted Class: {predicted_class}")

    # **6. Send predicted class to Arduino for sorting**
    arduino.write(f"{predicted_class}\n".encode())

    # **7. Wait for sorting to complete**
    #time.sleep(1)  # Small delay for stability
