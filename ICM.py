import librosa
import numpy as np
import matplotlib.pyplot as plt

def classify_indian_classical(tempo, spectral_centroid):
    """
    Classifies the given audio into Hindustani or Carnatic classical music 
    based on tempo (BPM) and spectral centroid (Hz).
    """
    tempo_threshold = 120  # Carnatic generally >120 BPM
    spectral_threshold = 2000  # Carnatic has a brighter tone >2000 Hz

    classification = "Carnatic Classical Music" if (tempo > tempo_threshold and spectral_centroid > spectral_threshold) else "Hindustani Classical Music"
    
    # Print debug message to check if function runs
    print(f"DEBUG: Classification function called with Tempo={tempo} BPM, Spectral Centroid={spectral_centroid} Hz")
    
    return classification

def analyze_music(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Extract features
    tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Ensure tempo is a single scalar value
    tempo = tempo_array.item() if tempo_array.ndim > 0 and tempo_array.size > 0 else 0.0  

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # Compute average spectral centroid
    avg_spectral_centroid = np.mean(spectral_centroid)

    # Print extracted values for debugging
    print(f"Tempo: {tempo:.2f} BPM")
    print(f"Spectral Centroid: {avg_spectral_centroid:.2f} Hz")
    
    # Plotting the features
    plt.figure(figsize=(12, 8))

    # Plot Spectral Centroid
    plt.subplot(2, 1, 1)
    plt.plot(spectral_centroid, label='Spectral Centroid', color='b')
    plt.ylabel('Hz')
    plt.xlim([0, len(spectral_centroid)])
    plt.title('Spectral Centroid')
    plt.legend(loc='upper right')

    # Plot Chroma Features
    plt.subplot(2, 1, 2)
    plt.imshow(chroma_stft, aspect='auto', origin='lower', cmap='coolwarm')
    plt.ylabel('Pitch Class')
    plt.xlabel('Time Frames')
    plt.title('Chroma Features')

    plt.tight_layout()
    plt.show()

    # Return the extracted features
    return tempo, avg_spectral_centroid

if __name__ == "__main__":
    file_path = r"C:\Users\Akshay\Downloads\download.wav"
    
    # Get tempo and spectral centroid from the analysis
    tempo, spectral_centroid = analyze_music(file_path)
    
    # Ensure values are valid before classification
    if tempo > 0 and spectral_centroid > 0:
        classification = classify_indian_classical(tempo, spectral_centroid)
        print(f"The given music is classified as: {classification}")
    else:
        print("Error: Could not extract valid features from the audio.")
