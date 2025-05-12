import speech_recognition as sr
from datetime import datetime

# Importamos las bibliotecas necesarias

# Función para convertir palabras relacionadas con el tiempo en valores numéricos
def parse_time_expression(expression):
    """
    Convierte expresiones de tiempo como 'dos horas' o 'treinta minutos' en segundos.
    """
    time_units = {
        "segundo": 1,
        "segundos": 1,
        "minuto": 60,
        "minutos": 60,
        "hora": 3600,
        "horas": 3600
    }
    words = expression.split()
    total_seconds = 0
    for i, word in enumerate(words):
        if word.isdigit():  # Si la palabra es un número
            value = int(word)
            if i + 1 < len(words) and words[i + 1] in time_units:
                total_seconds += value * time_units[words[i + 1]]
    return total_seconds

# Configuración del reconocimiento de voz
def recognize_speech():
    """
    Captura el audio del micrófono y lo convierte en texto utilizando la biblioteca SpeechRecognition.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Por favor, hable ahora...")
        try:
            audio = recognizer.listen(source)  # Escucha el audio del micrófono
            text = recognizer.recognize_google(audio, language="es-ES")  # Convierte el audio a texto
            print(f"Texto reconocido: {text}")
            return text
        except sr.UnknownValueError:
            print("No se pudo entender el audio.")
        except sr.RequestError as e:
            print(f"Error con el servicio de reconocimiento de voz: {e}")
    return None

# Función principal para realizar razonamiento de tiempo
def main():
    """
    Captura una expresión de tiempo hablada, la convierte a segundos y realiza un cálculo probabilístico.
    """
    print("Bienvenido al sistema de razonamiento de tiempo basado en reconocimiento de habla.")
    speech_text = recognize_speech()  # Captura el texto hablado
    if speech_text:
        # Intentamos interpretar la expresión de tiempo
        seconds = parse_time_expression(speech_text)
        if seconds > 0:
            print(f"El tiempo interpretado es de {seconds} segundos.")
            # Ejemplo de razonamiento probabilístico: probabilidad de un evento en ese tiempo
            probability = 1 - (0.5 ** (seconds / 60))  # Ejemplo: probabilidad de un evento en función del tiempo
            print(f"La probabilidad calculada para el evento en {seconds} segundos es: {probability:.2f}")
        else:
            print("No se pudo interpretar una expresión de tiempo válida.")
    else:
        print("No se recibió entrada de voz válida.")

# Ejecutamos el programa principal
if __name__ == "__main__":
    main()