import subprocess
import os
import sys

def run_script(script_name):
    """Запускает Python-скрипт и ждет его завершения."""
    print(f"\n--- Запуск этапа: {script_name} ---")
    try:
        # Запускаем скрипт тем же интерпретатором Python, который запустил этот файл
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении {script_name}: {e}")
        return False

def main():
    # 1. Пути к твоим файлам
    whisper_script = "whisper.py"
    qwen_script = "qwen.py"
    intermediate_file = "raw_text.txt"

    # Шаг 1: Транскрибация (Whisper)
    if not run_script(whisper_script):
        print("Остановка конвейера на этапе распознавания речи.")
        return

    # Проверка: появился ли промежуточный файл?
    if not os.path.exists(intermediate_file):
        print(f"Ошибка: {intermediate_file} не был создан. Проверьте whisper.py.")
        return

    # Шаг 2: Форматирование (Qwen)
    if not run_script(qwen_script):
        print("Остановка конвейера на этапе обработки текста.")
        return

    print("\n==========================================")
    print("ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
    print("Итоговый файл: result.tex")
    print("==========================================")

if __name__ == "__main__":
    main()