import os
import wave
import pyaudio
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
import soundfile as sf

warnings.filterwarnings('ignore')

# Function to load audio files
def load_audio_files(directory_path):
    audio_data = []
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.wav')]
    for path in file_paths:
        try:
            data, _ = librosa.load(path, sr=None)
            audio_data.append(data)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return audio_data

# Function to extract features from audio data
def extract_features(audio_data):
    features = []
    for idx, data in enumerate(audio_data):
        try:
            mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0)
            chroma = librosa.feature.chroma_stft(y=data, sr=22050)
            chroma = np.mean(chroma.T, axis=0)
            spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=22050)
            spectral_contrast = np.mean(spectral_contrast.T, axis=0)
            combined_features = np.concatenate((mfccs, chroma, spectral_contrast))
            features.append(combined_features)
        except Exception as e:
            print(f"Error extracting features from file {idx}: {e}")
    return features

# Function to prepare data for training/testing
def prepare_data(data_dirs, labels):
    all_features = []
    all_labels = []
    for idx, directory in enumerate(data_dirs):
        audio_data = load_audio_files(directory)
        features = extract_features(audio_data)
        all_features.extend(features)
        all_labels.extend([labels[idx]] * len(features))
    return all_features, all_labels

# Function to predict dialect of a given audio file
def predict_dialect(clf, le, audio_file):
    try:
        data, _ = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=data, sr=22050)
        chroma = np.mean(chroma.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=22050)
        spectral_contrast = np.mean(spectral_contrast.T, axis=0)
        combined_features = np.concatenate((mfccs, chroma, spectral_contrast)).reshape(1, -1)
        prediction = clf.predict(combined_features)
        predicted_label = le.inverse_transform(prediction)
        return predicted_label[0]
    except Exception as e:
        print(f"Error predicting dialect for {audio_file}: {e}")
        return None

# Train the model and generate initial report
def train_model_and_generate_initial_report():
    print('Training data...')
    trainData = [
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\Ramallah_Reef',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\Nablus',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\Jerusalem',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\Hebron'
    ]

    labels = ['Ramallah', 'Nablus', 'Jerusalem', 'Hebron']

    train_features, train_labels = prepare_data(trainData, labels)
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_features, train_labels_encoded)

    # Generate initial report
    testData = [
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\testing data\Ramallah-Reef',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\testing data\Nablus',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\testing data\Jerusalem',
        r'D:\reev\BZU\fourth year\2 sem\SPOKEN\spoken\project\testing data\Hebron'
    ]

    test_labels = ['Ramallah', 'Nablus', 'Jerusalem', 'Hebron']
    test_features, test_labels = prepare_data(testData, test_labels)
    test_labels_encoded = le.transform(test_labels)
    predictions = clf.predict(test_features)

    acc = accuracy_score(test_labels_encoded, predictions)
    report = classification_report(test_labels_encoded, predictions, target_names=le.classes_)

    print(f"Accuracy: {acc}")
    print("Classification Report:")
    print(report)

    return clf, le, acc, report

# Function to convert audio file to WAV format
def convert_to_wav(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        wav_path = "temp.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting audio file to WAV: {e}")
        return None

# Function to play sound
def play_sound(filename):
    chunk = 1024
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()

# Function to handle file selection and prediction
def select_file(entry_widget):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac")])
    if file_path:
        entry_widget.configure(state='normal')
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        entry_widget.configure(state='readonly')

# Function to clear entry
def clear_entry(entry_widget):
    entry_widget.configure(state='normal')
    entry_widget.delete(0, tk.END)
    entry_widget.configure(state='readonly')

# Function to handle dialect prediction
def predict_dialect_ui(entry_widget):
    file_path = entry_widget.get()
    if file_path:
        wav_path = convert_to_wav(file_path)
        predicted_dialect = predict_dialect(clf, le, wav_path)
        if predicted_dialect:
            messagebox.showinfo("Prediction", f"The predicted dialect is: {predicted_dialect}")
        else:
            messagebox.showerror("Error", "Failed to predict the dialect.")
    else:
        messagebox.showwarning("Input Error", "Please select an audio file first.")

# Function to print the classification report
def print_report():
    report_text.delete(1.0, tk.END)
    report_text.insert(tk.END, f"Accuracy: {initial_acc}\n")
    report_text.insert(tk.END, "Classification Report:\n")
    report_text.insert(tk.END, initial_report)

# Train the model and generate initial report
clf, le, initial_acc, initial_report = train_model_and_generate_initial_report()

# Create the main window
root = tk.Tk()
root.geometry("800x600")
root.title("Dialect Recognition System")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = tk.Label(frame, text="Dialect Recognition System", font=("Arial", 20), bg="#f0f0f0")
label.pack(pady=10, padx=10, anchor="w")

label_entry = tk.Label(frame, text="Sound to be analyzed:", font=("Arial", 16), bg="#f0f0f0")
label_entry.pack(pady=5, padx=10, anchor="w")

entry = tk.Entry(frame, width=50, font=("Arial", 14), state='readonly')
entry.pack(pady=12, padx=30)

browse_button = tk.Button(frame, text="Browse", command=lambda: select_file(entry), font=("Arial", 14), bg="#4CAF50", fg="white")
browse_button.pack(pady=5, padx=10)

dialect_button = tk.Button(frame, text="Predict Dialect", command=lambda: predict_dialect_ui(entry), font=("Arial", 14), bg="#4CAF50", fg="white")
dialect_button.pack(pady=5, padx=10)

sound_button = tk.Button(frame, text="Play Sound", command=lambda: play_sound(entry.get()), font=("Arial", 14), bg="#4CAF50", fg="white")
sound_button.pack(pady=5, padx=10)

clear_button = tk.Button(frame, text="Clear", command=lambda: clear_entry(entry), font=("Arial", 14), bg="#4CAF50", fg="white")
clear_button.pack(pady=12, padx=5)

report_button = tk.Button(frame, text="Generate Report", command=print_report, font=("Arial", 14), bg="#4CAF50", fg="white")
report_button.pack(pady=5, padx=10)

report_text = tk.Text(frame, wrap='word', font=("Arial", 12), bg="#f0f0f0", height=10, width=70)
report_text.pack(pady=5, padx=10, fill="both", expand=True)

root.mainloop()
