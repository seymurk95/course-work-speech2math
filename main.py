import subprocess
import sys
import os
import time
import requests

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

def unload_lm_model():
    print("⏳ Выгружаем модель из LM Studio...")
    try:
        # 1. Пытаемся по-хорошему через API (как и раньше)
        resp = requests.get(f"{LM_STUDIO_URL}/v1/models")
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                model_id = models[0].get("id")
                requests.post(f"{LM_STUDIO_URL}/v1/models/unload", json={"model": model_id})
        
        # 2. ГРУБАЯ СИЛА (добавь это!)
        # Убиваем все процессы, в имени которых есть 'lmstudio' или которые запускаются из папки .lmstudio
        print("🧹 Принудительная очистка VRAM...")
        os.system("pkill -9 -f lmstudio")
        os.system("pkill -9 -f .lmstudio")
        
        # Даем системе 2 секунды, чтобы видеокарта осознала свободу
        time.sleep(2)
        print("✅ Видеопамять должна быть свободна")

    except Exception as e:
        print(f"⚠️ Ошибка при очистке: {e}")


def run_pipeline(audio_path):
    if not os.path.exists(audio_path):
        print(f"Ошибка: Файл {audio_path} не найден!")
        return

    print(f"\nНачинаем обработку файла: {audio_path}")

    # === Whisper ===
    print("--- Запуск Whisper (на GPU) ---")
    unload_lm_model()

    try:
        subprocess.run([sys.executable, "whisper.py", audio_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка на этапе Whisper: {e}")
        return
    finally:
        # Загружаем модель обратно в любом случае
        print("⏳ Загружаем модель LM Studio обратно...")
        time.sleep(3)

    # === Qwen ===
    print("\n--- Запуск Qwen ---")
    try:
        subprocess.run([sys.executable, "qwen.py"], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка на этапе Qwen.")
        return

    # === Стилист ===
    print("\n--- Запуск стилиста ---")
    try:
        subprocess.run([sys.executable, "stilist.py"], check=True)
    except subprocess.CalledProcessError:
        print("Ошибка на этапе стилиста.")
        return

    print("\n🎉 ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")


if __name__ == "__main__":
    file_to_process = "nulmer.wav"
    run_pipeline(file_to_process)
