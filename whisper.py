import torch
import librosa
from transformers import pipeline

# Определение устройства
device = 0 if torch.cuda.is_available() else -1

# Инициализируем пайплайн
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    torch_dtype=torch.float16 if device == 0 else torch.float32,
    device=device
)

audio_file = "Rastvori.wav" 

print("Загрузка аудио...")
# Загружаем
audio_array, sampling_rate = librosa.load(audio_file, sr=16000)

print("Распознавание речи с перекрытием (stride)...")


result = pipe(
    audio_array,
    chunk_length_s=30,      # Длина куска
    stride_length_s=(5, 5), # ВАЖНО: Делаем перекрытие по 5 секунд с краев
    batch_size=8,           # Обработка пачками
    return_timestamps=True, # Помогает модели не терять нить повествования
    generate_kwargs={"language": "russian", "task": "transcribe"}
)

text = result["text"]
print("\nмодель Wisper завершила свою работу")
#print(text)

with open("raw_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
