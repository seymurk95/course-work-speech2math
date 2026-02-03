import torch
import librosa
import sys
from transformers import pipeline

# Определение устройства
device = 0 if torch.cuda.is_available() else -1

# Инициализируем пайплайн
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    torch_dtype=torch.float16 if device == 0 else torch.float32,
    device=device
)import torch
import librosa
from transformers import pipeline
import sys # Добавляем для чтения аргументов

# Проверка наличия аргумента
if len(sys.argv) < 2:
    print("Ошибка: Не указан путь к аудиофайлу.")
    print("Использование: python whisper.py <путь_к_файлу>")
    sys.exit(1)



device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    torch_dtype=torch.float16 if device == 0 else torch.float32,
    device=device
)

print(f"Загрузка аудио: {audio_file}...")
audio_array, sampling_rate = librosa.load(audio_file, sr=16000)

print("Распознавание речи...")
result = pipe(
    audio_array,
    chunk_length_s=30,
    stride_length_s=(5, 5),
    batch_size=8,
    return_timestamps=True,
    generate_kwargs={"language": "russian", "task": "transcribe"}
)

text = result["text"]

with open("raw_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Транскрипция сохранена в raw_text.txt")

if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    # Оставляем значение по умолчанию, чтобы файл работал сам по себе
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


