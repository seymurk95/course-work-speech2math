import torch
import librosa
import gc
import os
from transformers import pipeline
import sys

# === Оптимизация памяти ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

# Получаем путь к аудио
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    audio_file = "nepr.wav"

# Определяем устройство
device = 0 if torch.cuda.is_available() else -1
print(f"Whisper запущен на: {'GPU' if device == 0 else 'CPU'}")

# Загружаем модель
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    dtype=torch.float16 if device == 0 else torch.float32,
    device=device,
    model_kwargs={"attn_implementation": "sdpa"}
)

print(f"Загрузка аудио: {audio_file}...")
audio_array, sampling_rate = librosa.load(audio_file, sr=16000)

print("Распознавание речи...")
result = pipe(
    audio_array,
    chunk_length_s=30,
    stride_length_s=(5, 5),
    batch_size=8,
    return_timestamps=False,
    generate_kwargs={"language": "russian", "task": "transcribe"}
)

text = result["text"]

with open("raw_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Транскрипция сохранена в raw_text.txt")
