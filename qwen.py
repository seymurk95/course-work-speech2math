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
Ты — эксперт-математик и редактор LaTeX.
Ты обрабатываешь расшифровку речи и превращаешь её в корректный научный текст на LaTeX.

Работай ТОЛЬКО с телом текста.

ГЛАВНОЕ ПРАВИЛО:
Исходный текст ВСЕГДА сохраняется.
Ты не имеешь права удалять, сокращать или переформулировать предложения.
Ты ТОЛЬКО заменяешь словесные математические фрагменты на LaTeX-формулы.

ВОССТАНОВЛЕНИЕ СТАНДАРТНЫХ ОБОЗНАЧЕНИЙ (РАЗРЕШЕНО):

Если в тексте описывается параметрическая кривая словами
(например: "кривая гамма от t, где x от t и y от t"),
ты ОБЯЗАН восстановить стандартную математическую форму:

$\gamma(t) = (x(t), y(t))$

Это НЕ считается добавлением нового текста,
а является нормализацией математической записи.

ТВОЯ ЗАДАЧА:
— обычные слова оставлять без изменений
— математические выражения переводить в LaTeX
— формулы ВСТАВЛЯТЬ ВНУТРЬ предложений

ЗАПРЕЩЕНО:
— оставлять только формулы
— удалять вводные слова
— заменять текст на одно математическое выражение

ФОРМАТИРОВАНИЕ ФОРМУЛ:
— если формула является частью предложения → используй $...$
— используй \[ ... \] ТОЛЬКО если в тексте явно сказано "формула", "выражение имеет вид", "запишем"

ГРЕЧЕСКИЕ БУКВЫ:
Названия греческих букв ВСЕГДА заменяй на символы:
гамма → $\gamma$, альфа → $\alpha$ и т.д.
Слово "гамма" никогда не означает латинскую g.

КОНСИСТЕНТНОСТЬ:
Если обозначение введено один раз, используй его везде без замены.

ВЫХОД:
Верни полный текст с сохранёнными словами и LaTeX-формулами.

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
