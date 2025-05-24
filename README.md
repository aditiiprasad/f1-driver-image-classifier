# 🏎️ F1 Driver Classifier

A web app that identifies F1 drivers from images using a deep learning model. Upload an image of a driver (Charles Leclerc, Carlos Sainz, Max Verstappen, Lewis Hamilton, or Fernando Alonso), and the app predicts who it is!

---
## Demo

![App Screenshot](./f1-face-classifier//sample_images/demo.gif)  

---

## Tech Stack

- **Frontend & UI:** [Streamlit](https://streamlit.io/)
- **Model Framework:** TensorFlow , Keras
- **Python Libraries:**  
  - `tensorflow` for loading and running the model  
  - `numpy` for image preprocessing  
  - `Pillow` for image handling  
  - `base64` and `io` for image encoding in the app
- **Model:** Custom CNN trained on F1 driver images
- **Training Environment** Google Colab (GPU-enabled notebook for faster model training)
- **Deployment:** Render

---

## Folder Structure

```
F1-Driver-Classifier/
│
|── f1_dataset
├── f1-face-classifier
|  ├── app.py # Main Streamlit app script
|  ├── model.h5 # Trained TensorFlow model
|  ├── requirements.txt # Python dependencies
|  ├── sample_images/ # Sample images for users to try
│           ├── A.jpg
│           ├── B.jpg
│           └── ...
|
├── README.md # This README file

```
---

## Model Training

Google Colab was used for training because it provides free access to GPUs, which significantly speeds up the training process and offers a convenient cloud-based environment without the need for local setup.

- **Dataset:** Collected images of 5 F1 drivers: Carlos Sainz, Charles Leclerc, Fernando Alonso, Lewis Hamilton, Max Verstappen.

```
from google.colab import files
uploaded = files.upload()
```

```
import zipfile
import os

with zipfile.ZipFile("f1_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("f1_dataset")


os.listdir("f1_dataset")

```

- **Preprocessing:**  
  - Images resized to 100x100 pixels  
  - Normalized pixel values (scaled between 0 and 1)

  ```
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

  train_gen = datagen.flow_from_directory(
    "f1_dataset/f1_dataset",  # Note the extra folder
    target_size=(100, 100),
    batch_size=16,
    class_mode="categorical",
    subset="training"
  )

  val_gen = datagen.flow_from_directory(
    "f1_dataset/f1_dataset",
    target_size=(100, 100),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
  )
  ```


- **Model Architecture:** Custom Convolutional Neural Network (CNN) built with Keras.
- **Training Details:**  
  - Split dataset into training and validation sets  
  - Used categorical cross-entropy loss and Adam optimizer  
  - Trained for multiple epochs until validation accuracy stabilized

  ```
  history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
   )
  ``` 
  
- **Output:** Saved the trained model as `model.h5` for inference in the Streamlit app.

```
model.save("model.h5")
```
```
from google.colab import files
files.download("model.h5")
```



---

## How to Run Locally

1. Clone the repo:  
   ```bash
   git clone https://github.com/aditiiprasad/f1-driver-image-classifier.git
   cd f1-driver-classifier
   cd f1-face-classifier

2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:

```
streamlit run app.py
```
Upload or select a sample image and see predictions instantly!

### Challenges Faced
1. ***Dataset collection:*** Gathering enough labeled images of each driver was time-consuming.

2. ***Image quality and consistency:*** Varying lighting, angles, and backgrounds made model training harder.

3. ***Model accuracy:*** Overfitting was a problem initially; solved by data augmentation and dropout layers.

4. ***UI layout:*** Customizing Streamlit’s UI and CSS to achieve the glassmorphism effect and responsive design took multiple iterations.

5. ***State management:*** Handling image upload and prediction flow smoothly with Streamlit’s session state was tricky.

### Future Improvements
1. Expand the model to recognize more drivers and different racing categories.

2. Improve UI responsiveness on mobile devices.

3. Add more user feedback and confidence scores.

4. Deploy the app on Streamlit Cloud or similar platforms for easy sharing.

### Author
Developed by Aditi


