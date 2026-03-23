import os
from openai import OpenAI

# Настройка клиента
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1", 
    api_key="lm-studio"
)

# Читаем результат предыдущего шага и сохраняем в него же или в финальный файл
input_file = "result.tex"
output_file = "final_lecture.tex"

if not os.path.exists(input_file):
    print(f"Ошибка: {input_file} не найден. Сначала запустите qwen.py.")
    exit()

with open(input_file, "r", encoding="utf-8") as f:
    latex_content = f.read()

system_instruction = r"""
Ты — робот-верстальщик LaTeX. Твоя единственная роль: расставить переносы строк и выделить заголовки.

 ГЛАВНЫЕ ЗАПРЕТЫ (ШТРАФ):
1. ЗАПРЕЩЕНО добавлять новые слова, даже если предложение кажется незаконченным.
2. ЗАПРЕЩЕНО менять формулировки. Если в тексте ошибка (например, перепутаны индексы), ОСТАВЬ ЕЁ КАК ЕСТЬ. Ты не учитель, ты верстальщик.
3. ЗАПРЕЩЕНО менять порядок предложений.
4. ЗАПРЕЩЕНО выбрасывать любые части текста (номера вопросов, вводные слова).

 ЧТО НУЖНО СДЕЛАТЬ:
1. ТЕКСТ: Весь текст из входного файла должен перекочевать в выходной слово в слово.
2. АБЗАЦЫ: Между логическими блоками (например, после вопроса, перед ответом) вставляй ДВЕ пустые строки.
3. ЖИРНЫЙ ШРИФТ: Выделяй только ключевые сущности, которые УЖЕ есть в тексте: "Вопрос №", "Ответ", "Определение", "Примеры".
4. МАТЕМАТИКА: Оставляй $...$ и \[ ... \] нетронутыми.

 
 ФОРМАТ ВЫВОДА:
Выдай только чистый LaTeX код. Без комментариев, без ```latex.
"""

print("Стилистическая правка текста...")

try:
    completion = client.chat.completions.create(
        model="qwen3-esper3-reasoning-coder-instruct-12b-brainstorm20x-i1@q4_k_m", # Или qwen2.5-7b
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": latex_content}
        ],
        temperature=0.1,
    )

    final_content = completion.choices[0].message.content
    
    # Очистка от мыслей и markdown
    if "<think>" in final_content:
        final_content = final_content.split("</think>")[-1].strip()
    final_content = final_content.replace("```latex", "").replace("```", "").strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    print(f"Готово! Стилизованный файл: {output_file}")

except Exception as e:
    print(f"Ошибка стилиста: {e}")