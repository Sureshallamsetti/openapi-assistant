import speech_recognition as sr

def recognize_speech_from_mic():
    # Create a Recognizer instance
    recognizer = sr.Recognizer()

    # Use the default microphone as source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)

        print("Listening... Speak now!")
        audio = recognizer.listen(source)

        print("Recognizing...")

        try:
            # Convert speech to text
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Request error from Google Speech Recognition service: {e}")
        return None

def main():
    print("=== Speech to Text Conversion ===")
    transcription = recognize_speech_from_mic()

    if transcription:
        # Save transcription to file
        with open("transcription.txt", "w") as file:
            file.write(transcription)
        print("Transcription saved to 'transcription.txt'.")
    else:
        print("No transcription was generated.")

if __name__ == "__main__":
    main()
