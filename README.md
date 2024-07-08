
# Palestinian Accent Recognition System

🔍 Description:This project focuses on recognizing Palestinian dialects from audio files using machine learning techniques.
## 🎯 Project Overview
The project involves the following key components:

Loading Audio Files: Load audio files from specified directories.

Feature Extraction: Extract relevant features such as MFCCs, chroma, and spectral contrast from the audio data.

Model Training: Train a RandomForestClassifier on the extracted features.

Prediction: Predict the dialect of a given audio file.

GUI Interface: A Tkinter-based GUI for user interaction, including file selection, dialect prediction, and report generation.
## 🛠️ Dependencies
Make sure to install the following dependencies before running the project:

pip install numpy scipy scikit-learn librosa pyaudio pydub tkinter

## Functions:

load_audio_files(directory_path): Loads audio files from a given directory.

extract_features(audio_data): Extracts MFCCs, chroma, and spectral contrast features from the audio data.

prepare_data(data_dirs, labels): Prepares the data for training/testing by loading and extracting features.

predict_dialect(clf, le, audio_file): Predicts the dialect of a given audio file.

train_model_and_generate_initial_report(): Trains the model and generates an initial classification report.

convert_to_wav(file_path): Converts an audio file to WAV format.

play_sound(filename): Plays an audio file.

select_file(entry_widget): Handles file selection through a GUI.

clear_entry(entry_widget): Clears the entry widget in the GUI.

predict_dialect_ui(entry_widget): Handles dialect prediction through the GUI.

print_report(): Prints the classification report in the GUI.
## Screenshots

![1](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/6850dec0-9629-4d13-a2f9-9effc318ef0e)
![2](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/d759232e-6850-442a-97df-495f84dbaab5)

![3](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/524567ee-44ac-4a7e-a00a-d61e1bce34dd)
![4](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/6267db5e-ed52-4177-8398-86c392fd1359)
![5](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/67e381dc-913b-45d5-b391-ae9a1ab6a41e)
![6](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/d95a6ff7-7164-4d22-9448-119c47a7bf6c)
![7](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/10085c75-fa95-4a3c-b726-eac3d4f35694)
![8](https://github.com/Rivanjaradat/Palestinian_Accent_Recognition_System/assets/103911286/90878e3a-8a68-4f8b-8835-1523007da9ee)



