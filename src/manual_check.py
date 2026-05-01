from src.manipulation_features import extract_manipulation_features

text = """
Срочно!!! Эксперты считают, что стране грозит катастрофа.
Врачи предупреждают: медлить нельзя, действовать нужно немедленно.
Все знают, что другого выбора нет.
"""

result = extract_manipulation_features(text)

print(result)