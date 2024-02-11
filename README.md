# DRIVER ACTIVITY TRACKER

This project is an application designed to run on any device to track driver activity. It aims to enhance road safety by monitoring whether a driver is distracted or drowsy, and alerting the driver accordingly based on an alert meter. This project is made using the model trained using YoloV8, Dlib, OpenCV's libraries, etc.

<div>
<img src="https://github.com/granthgg/Distracted-Drowsy-Driver-Detection/assets/69439823/68ec3240-654e-4631-acfa-1cf2a7488a5f" width="260" height="230"/>&nbsp; 
<img src="https://github.com/granthgg/Distracted-Drowsy-Driver-Detection/assets/69439823/f72f1b44-5f68-4b91-b2c5-4ec495267a0e" width="260" height="230"/>&nbsp; 
<img src="https://github.com/granthgg/Distracted-Drowsy-Driver-Detection/assets/69439823/e49fac6a-a71b-4af1-be0e-327370720261" width="260" height="230"/>&nbsp; 
<div>

## Capabilities
The application is capable of tracking and detecting various driver activities such as:
- Whether the Driver is wearing a Seatbelt
- Whether the Driver is using a Phone
- Whether the Driver is using a Yawing
- Where the Driver is Looking
- Whether the Driver's Eyes are Opened or Closed
- And many more...



## Features

- **Real-time Detection**: Utilizes live video feed to detect drowsiness and distraction behaviors.
- **Multiple Behaviors Recognition**: Capable of recognizing various behaviors indicating drowsiness (e.g., frequent yawning, eye closure) and distraction (e.g., mobile phone usage).
- **Alert System**: Generates audible alerts to wake or alert the driver upon detection of any drowsy or distracted behavior.
- **Extensive Training Data**: Includes trained models and datasets for eyes closed, mobile usage, yawning detection, and seatbelt usage.

## Project Structure

- `Application.py`: The main application script for running the detection system.
- `Project Images/`: Contains various project-related images, including abstracts, infographics, and sample detection images.
- `Train/`: Includes training scripts and results for different behaviors (Eyes Closed, Mobile Usage, Yawning, Seatbelt).
  - `Eyes_Closed_Train/`, `Mobile_Train/`, `Yawn_Train/`, `Seatbelt_Train/`: Each directory contains training results, curves (F1, PR, P, R), confusion matrices, and trained model weights.
- `alert_sound.wav`: The alert sound file played when a drowsy or distracted behavior is detected.
- `best_*.pt`: Pre-trained model weights for different detection modules.
- `shape_predictor_68_face_landmarks.dat`: Facial landmark predictor used for detecting facial features related to drowsiness and distraction.

## Getting Started

To run the Distracted Drowsy Driver Detection system, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python installed along with necessary libraries such as OpenCV, PyTorch, and dlib.
3. Run `Application.py` to start the detection system.
4. Adjust the camera to ensure it captures the driver's face clearly.

## Contributing

Contributions to the Distracted Drowsy Driver Detection project are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.




