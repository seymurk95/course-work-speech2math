from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
)

# Укажи путь к твоему файлу
audio_file = r"Zadachi.wav"

# Распознаём речь
result = pipe(audio_file, language="ru", task="transcribe")

# Текст из модели
text = result["text"]

# Выводим в терминал
print("TEXT:", text)

# Записываем в файл
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Готово! Текст сохранён в result.txt")

