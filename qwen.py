import os
from openai import OpenAI

# Настройка клиента
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1", 
    api_key="lm-studio"
)

input_file = "raw_text.txt"
output_file = "result.tex" # Сохраняем сразу как .tex файл

# Шаблон LaTeX документа с поддержкой русского языка
latex_template_start = r"""
\documentclass[12pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[left=2cm, right=2cm, top=2cm, bottom=2cm]{geometry}

\begin{document}

"""

latex_template_end = r"""
\end{document}
"""

if not os.path.exists(input_file):
    print(f"Ошибка: Файл {input_file} не найден.")
    exit()

print(f"Чтение файла {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Промпт (оставляем тот же, он сработал хорошо)
system_instruction = """
Ты — редактор LaTeX. Твоя задача: найти в тексте математические выражения, записанные словами, и заменить их на формулы LaTeX.
Правила:
1. Окружай формулы знаком доллара: $x^2$.
2. Русский текст оставляй без изменений.
3. "Гамма" -> $\\gamma$, "интеграл от а до б" -> $\\int_{a}^{b}$.
4. "д гамма по д т" -> $\\frac{d\\gamma}{dt}$.
5. Верни ТОЛЬКО исправленный текст, без вступлений.
"""

print("Обработка текста нейросетью...")

try:
    completion = client.chat.completions.create(
        model="qwen2.5-7b-instruct-1m",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.2,
    )

    content_body = completion.choices[0].message.content

    # Склеиваем шаблон + текст от нейросети + конец шаблона
    full_latex_document = latex_template_start + content_body + latex_template_end

    print("Сохранение результата...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_latex_document)

    print(f"\nГОТОВО! Файл сохранен как: {output_file}")
    print("Теперь вы можете открыть этот файл в TeXStudio, Overleaf или любом другом редакторе и скомпилировать.")

except Exception as e:
    print(f"Ошибка: {e}")