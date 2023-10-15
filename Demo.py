import subprocess
import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("numpy")
# install("tensorflow")
# install("Pillow==9.5.0")
# install("opencv-python")


import numpy
import tensorflow

from PIL import Image
import os
import cv2
import numpy as np
from contextlib import redirect_stdout

# pip Install:
# Tensorflow
# Pillow==9.5.0
# opencv-python



model = tensorflow.keras.models.load_model("model_keras.h5")

classes = ["R", "U", "I", "N", "G", "Z", "T", "S", "A", "F", "O", "H", " ", "M", "J", "C", "D", "V", "Q", "X", "E", "B", "K", "L", "Y", "P", "W"]

word = ""


# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_counter = 0  # Initialize frame counter

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize the frame for display
    display_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))

    # Display the resized frame
    cv2.imshow('Webcam View', display_frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF

    frame_counter += 1  # Increment frame counter

    if frame_counter % 5 == 0:  # Process every 5th frame
        # Preprocess the image for the model
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.ANTIALIAS)
        inp_numpy = numpy.array(img)[None]

        # Get the predictions
        with redirect_stdout(open(os.devnull, 'w')):
            class_scores = model.predict(inp_numpy)[0]
        
        # Find the class with the highest score
        predicted_class_index = np.argmax(class_scores)
        predicted_class = classes[predicted_class_index]
        
        # Print the predicted class
        os.system('cls' if os.name == 'nt' else 'clear')
        print("TEXT: ", word)
        print(predicted_class)

    if key == 32:  # 32 is the ASCII value for space
        word += predicted_class
    
    # If 'q' is pressed, exit the loop
    elif key == ord('q'):
        break
    elif key == ord('w'):
        word = ""

# Release the web cam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
