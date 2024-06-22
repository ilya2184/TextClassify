import re
import Levenshtein

def clean_text(text):
    # Заменить все символы, кроме русских букв, английских букв и цифр, на пробел
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', text)
    # Удалить повторяющиеся пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_best_match(string_list, text):
    cleaned_text = clean_text(text)
    best_match = None
    best_score = float('inf')
    length_penalty_factor = -0.1  # Коэффициент штрафа за длину

    for current_string in string_list:
        cleaned_current_string = clean_text(current_string)
        if len(cleaned_current_string) == 0:
            continue
        for i in range(len(cleaned_text) - len(cleaned_current_string) + 1):
            substring = cleaned_text[i:i+len(cleaned_current_string)]
            distance = Levenshtein.distance(cleaned_current_string, substring)
            # Нормализуем расстояние делением на длину строки и добавляем штраф за длину
            normalized_distance = (distance / len(cleaned_current_string)) + (length_penalty_factor * len(cleaned_current_string))
            
            # Используем нормализованное расстояние для определения лучшего совпадения
            if normalized_distance < best_score:
                best_score = normalized_distance
                best_match = current_string
    return best_match
