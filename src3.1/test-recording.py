import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = 4  # Duration

print("🎤 Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
print("✅ Done recording!")

write("test.wav", fs, recording)
print("📁 Saved as test.wav")
