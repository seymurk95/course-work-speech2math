import os
from openai import OpenAI

# Настройка клиента
# Измените base_url, добавив /api/v1
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
system_instruction = r"""
Ты — LaTeX-редактор.

Твоя задача:
Заменять словесные математические фрагменты на LaTeX-формулы
внутри исходного текста.

Ты НЕ пишешь новый текст.
Ты НЕ объясняешь.
Ты НЕ добавляешь ничего от себя.

==================================================
ЖЁСТКИЕ ПРАВИЛА
==================================================

1. ВЫХОД = только обработанный исходный текст.

2. СТРОГО ЗАПРЕЩЕНО добавлять:
   \documentclass
   \begin{document}
   \end{document}
   \usepackage

3. ЗАПРЕЩЕНО:
   - удалять слова
   - менять порядок слов
   - переписывать предложения
   - оставлять только формулы

4. Формулы:
   - внутри предложения → $...$
   - отдельным блоком \[...\] только если это явно сказано в тексте

==================================================
НОРМАЛИЗАЦИЯ МАТЕМАТИКИ
(это разрешено)
==================================================

1. Параметризация:
"кривая гамма от t, где x(t) и y(t)"
→ $\gamma(t) = (x(t), y(t))$

2. Кванторы:
"для любого" → $\forall$
"существует" → $\exists$
"такой что" → $:$
"принадлежит" → $\in$

3. Степени и индексы:
"x в квадрате" → $x^2$
"a n" → $a_n$

4. Греческие буквы:
гамма → $\gamma$
альфа → $\alpha$
бета → $\beta$

Слово "гамма" никогда не означает латинскую g.

==================================================
ЗАПРЕТ НА UNICODE
==================================================

НЕЛЬЗЯ использовать:
ℝ, ℤ, ℚ, ∈, ⊂, ∞

Вместо них используй:
$\mathbb{R}$
$\mathbb{Z}$
$\mathbb{Q}$
$\in$
$\subset$
$\infty$

МНОЖЕСТВА ЧИСЕЛ В ТЕКСТЕ:

Если в обычном тексте встречаются ℝ, ℤ, ℚ или названия
"вещественные числа", "целые числа", "рациональные числа",
ты ОБЯЗАН:

1. Заменить их на LaTeX-форму:
   ℝ → $\mathbb{R}$
   ℤ → $\mathbb{Z}$
   ℚ → $\mathbb{Q}$

2. Всегда оборачивать их в $...$,
даже если они стоят внутри обычного текста.


==================================================
САМОПРОВЕРКА
==================================================

Перед выводом проверь:

- нет \documentclass
- нет \begin{document}
- нет Unicode-символов
- текст не укорочен

Если ошибка есть — исправь и только потом выводи.

==================================================
ФОРМАТ ВЫВОДА
==================================================

Верни только обработанный текст.
Без комментариев. Без кода. Без пояснений.

"""

print("Обработка текста нейросетью...")

try:
    completion = client.chat.completions.create(
        model="qwen3-esper3-reasoning-coder-instruct-12b-brainstorm20x-128k-ctx",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.1,
    )

    # Печатаем весь объект, если что-то пошло не так
    if not completion.choices:
        print("\n!!! СЕРВЕР ВЕРНУЛ ПУСТОЙ ОТВЕТ !!!")
        print(f"Полный ответ от LM Studio: {completion}")
        exit()

    msg = completion.choices[0].message
    content_body = msg.content if msg.content else ""

    # Если модель "думала", отрезаем мысли и берем только результат
    if "<think>" in content_body:
        content_body = content_body.split("</think>")[-1].strip()

    # Очистка от Markdown
    content_body = content_body.replace("```latex", "").replace("```", "").strip()

    # Склейка и сохранение
    full_latex_document = latex_template_start + content_body + latex_template_end
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_latex_document)
    print(f"\nУспешно сохранено в {output_file}")

except Exception as e:
    print(f"\nКритическая ошибка: {e}")
