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
Ты — математический корректор. Твоя единственная задача: заменить в предоставленной расшифровке речи слова на математические символы LaTeX, сохраняя при этом КАЖДОЕ слово оригинального текста.

ПРАВИЛА:
1. НИЧЕГО НЕ УДАЛЯЙ И НЕ СОКРАЩАЙ. Если в исходном тексте есть вводные слова, повторы или пояснения — они ДОЛЖНЫ остаться. Не переходи на списки, если текст идет сплошным абзацем ,но если перечисляются новые определения, утверждения , леммы , теоремы или следствия пиши их с новой строки для читабельности .
2. ЕДИНООБРАЗИЕ СИМВОЛОВ: Если по смыслу текста речь идет об одном и том же объекте (например, кривой), следи, чтобы буква везде была одинаковой. Если слышится "джи", "г" или "гамма" в контексте производной по времени — всегда пиши $\gamma$.
3. ИСПРАВЛЕНИЕ ОШИБОК: Whisper может ошибаться ("в один" -> $v_1$, "икс два" -> $x^2$, "де гамма по де те" -> $\frac{d\gamma}{dt}$). Исправляй это, исходя из логики формулы.
4. ОФОРМЛЕНИЕ: Все переменные (x, y, t, v) — только в долларах $...$.
5. СТРОГИЙ ВЫВОД: Выдай только текст. Никаких "Вот ваш исправленный файл" или "Я добавил формулы".

Текст должен выглядеть так, будто человек читал лекцию, а ты просто заменил слова-формулы на знаки.
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
