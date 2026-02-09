import subprocess
import sys
import os

def run_pipeline(audio_path):
    # Проверяем, существует ли вообще аудиофайл
    if not os.path.exists(audio_path):
        print(f"Ошибка: Файл {audio_path} не найден!")
        return

    print(f"\nНачинаем обработку файла: {audio_path}")

    # Этап 1: Whisper
    # Мы передаем audio_path как дополнительный аргумент в командную строку
    print("--- Запуск Whisper ---")
    try:
        subprocess.run([sys.executable, "whisper.py", audio_path], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка на этапе Whisper.")
        return

    # Этап 2: Qwen
    print("\n--- Запуск Qwen (LaTeX formatting) ---")
    try:
        subprocess.run([sys.executable, "qwen.py"], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка на этапе Qwen.")
        return

    # Этап 3: Стилист (Важно: переносим 'Готово' в самый конец)
    print("\n--- ШАГ 3: Стилистическое оформление (stilist.py) ---")
    try:
        subprocess.run([sys.executable, "stilist.py"], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка на этапе стилиста.")
        return

    print("\n========================================")
    print("ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print("Финальный результат: final_lecture.tex")
    print("========================================")

if __name__ == "__main__":
    # Здесь вы можете вручную менять название файла
    file_to_process = "nulmer.wav" 
    
    # Или даже спрашивать пользователя в консоли:
    
    run_pipeline(file_to_process)
