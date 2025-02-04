import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk

# Load the Haarcascade file using an absolute path
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load the trained model
def load_model():
    with open("model_a.json", "r") as file:
        loaded_model_json = file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_weights.weights.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# GUI Initialization
top = tk.Tk()
top.geometry('900x700')  # Increased window size
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

# UI Elements
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    """ Detects emotion from an image file. """
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Improve face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected! Try a clearer image.")
        return
    
    for (x, y, w, h) in faces:
        fc = gray_image[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))  # Resize to 48x48 as required by model
        roi = roi.astype("float32") / 255.0  # Normalize
        roi = np.reshape(roi, (1, 48, 48, 1))  # Ensure correct shape

        pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
        label1.configure(foreground="#011638", text="Predicted Emotion: " + pred)

        # Draw bounding box and show detected face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Increased thickness
        cv2.imshow("Detected Face", cv2.resize(image, (600, 600)))  # Display in larger window
        cv2.waitKey(3000)  # Display for 3 seconds
        cv2.destroyAllWindows()

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    """ Opens file dialog and displays the selected image. """
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return  # If no file is selected, do nothing
        
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((400, 400), Image.Resampling.LANCZOS)  # Increased displayed image size
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')

        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error loading image: {e}")

# Buttons and UI Placement
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()


