import subprocess
import sys
import os
import time
import requests

LM_STUDIO_URL = "http://127.0.0.1:1234"

def unload_lm_model():
    print("⏳ Выгружаем модель из LM Studio...")
    try:
        # Получаем список моделей
        resp = requests.get(f"{LM_STUDIO_URL}/v1/models")
        if resp.status_code != 200:
            print("⚠️ Не удалось получить список моделей")
            return

        models = resp.json().get("data", [])
        if not models:
            print("ℹ️ Нет загруженных моделей")
            return

        # Берём первую загруженную модель
        model_id = models[0].get("id")
        print(f"Найдена модель: {model_id}")

        # Пытаемся выгрузить
        unload_resp = requests.post(
            f"{LM_STUDIO_URL}/api/v1/models/unload",
            json={"instance_id": model_id}
        )

        if unload_resp.status_code in (200, 204):
            print("✅ Модель успешно выгружена")
            time.sleep(4)
        else:
            print(f"⚠️ Не удалось выгрузить: {unload_resp.text}")

    except Exception as e:
        print(f"⚠️ Ошибка при выгрузке: {e}")


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
    file_to_process = "nepr.wav"
    run_pipeline(file_to_process)
