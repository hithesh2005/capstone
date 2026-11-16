import ctypes
import os
import time
import tempfile
import wave

import audioop

import speech_recognition as sr
from gtts import gTTS


# ============================================================
# WINDOWS MICROPHONE RECORDER
# ============================================================

class WindowsRecorder:

    def __init__(self):

        self.winmm = ctypes.WinDLL("winmm.dll")

        self.winmm.mciSendStringW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_void_p
        ]

        self.winmm.mciSendStringW.restype = ctypes.c_uint

    def command(self, command):

        error = self.winmm.mciSendStringW(
            command,
            None,
            0,
            None
        )

        if error != 0:

            raise RuntimeError(
                f"MCI command failed: {command}\n"
                f"Error code: {error}"
            )

    def record(self, output_file, duration=8):

        output_file = os.path.abspath(output_file)

        if os.path.exists(output_file):
            os.remove(output_file)

        print()
        print("=" * 60)
        print("MICROPHONE RECORDING")
        print("=" * 60)

        print(f"Recording for {duration} seconds...")
        print("Speak clearly after recording starts.")
        print()

        try:

            # Open Windows microphone
            self.command(
                "open new type waveaudio alias mediassist"
            )

            # IMPORTANT:
            # Do not use the "set" command.
            # Some Windows drivers return error 282.

            self.command(
                "record mediassist"
            )

            for remaining in range(duration, 0, -1):

                print(
                    f"\rRecording... {remaining} seconds remaining ",
                    end="",
                    flush=True
                )

                time.sleep(1)

            print()

            self.command(
                "stop mediassist"
            )

            self.command(
                f'save mediassist "{output_file}"'
            )

            self.command(
                "close mediassist"
            )

            if not os.path.exists(output_file):

                raise RuntimeError(
                    "WAV file was not created."
                )

            size = os.path.getsize(output_file)

            print()
            print("Recording saved:")
            print(output_file)

            print(
                f"Original audio size: "
                f"{size / 1024:.2f} KB"
            )

            return output_file

        except Exception as e:

            try:
                self.command("stop mediassist")
            except:
                pass

            try:
                self.command("close mediassist")
            except:
                pass

            raise e


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio(input_file, output_file):

    """
    Convert Windows MCI audio:

        11025 Hz
        8-bit
        mono

    into:

        16000 Hz
        16-bit
        mono

    Better format for speech recognition.
    """

    print()
    print("=" * 60)
    print("AUDIO CONVERSION")
    print("=" * 60)

    with wave.open(input_file, "rb") as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()

        frames = wav.readframes(
            wav.getnframes()
        )

    print(
        "Original:",
        channels,
        "channel(s),",
        sample_rate,
        "Hz,",
        sample_width * 8,
        "bit"
    )

    # --------------------------------------------------------
    # Ensure mono
    # --------------------------------------------------------

    if channels > 1:

        frames = audioop.tomono(
            frames,
            sample_width,
            1,
            1
        )

        channels = 1

    # --------------------------------------------------------
    # Convert 8-bit unsigned PCM to signed 8-bit PCM
    # --------------------------------------------------------

    if sample_width == 1:

        # WAV 8-bit PCM is unsigned.
        frames = audioop.bias(
            frames,
            1,
            -128
        )

        # Convert 8-bit -> 16-bit
        frames = audioop.lin2lin(
            frames,
            1,
            2
        )

        sample_width = 2

    # --------------------------------------------------------
    # Convert sample rate to 16000 Hz
    # --------------------------------------------------------

    if sample_rate != 16000:

        frames, _ = audioop.ratecv(
            frames,
            2,
            1,
            sample_rate,
            16000,
            None
        )

        sample_rate = 16000

    # --------------------------------------------------------
    # Check audio level
    # --------------------------------------------------------

    rms = audioop.rms(
        frames,
        2
    )

    peak = audioop.max(
        frames,
        2
    )

    print(
        "Converted audio:"
    )

    print(
        "Channels: 1"
    )

    print(
        "Sample rate:",
        sample_rate,
        "Hz"
    )

    print(
        "Sample width: 16-bit"
    )

    print(
        "RMS level:",
        rms
    )

    print(
        "Peak level:",
        peak
    )

    # --------------------------------------------------------
    # Amplify quiet recordings
    # --------------------------------------------------------

    if rms < 1000:

        print(
            "Audio is quiet. Applying amplification..."
        )

        gain = 4.0

        frames = audioop.mul(
            frames,
            2,
            gain
        )

        rms = audioop.rms(
            frames,
            2
        )

        print(
            "New RMS:",
            rms
        )

    elif rms < 2500:

        print(
            "Audio is slightly quiet. "
            "Applying small amplification..."
        )

        gain = 2.0

        frames = audioop.mul(
            frames,
            2,
            gain
        )

    else:

        print(
            "Audio level looks reasonable."
        )

    # --------------------------------------------------------
    # Save converted WAV
    # --------------------------------------------------------

    with wave.open(
        output_file,
        "wb"
    ) as wav:

        wav.setnchannels(1)

        wav.setsampwidth(2)

        wav.setframerate(16000)

        wav.writeframes(frames)

    print()
    print(
        "Converted WAV:"
    )

    print(
        output_file
    )

    return output_file


# ============================================================
# SPEECH TO TEXT
# ============================================================

class SpeechToText:

    LANGUAGE_CODES = {
        "en": "en-IN",
        "te": "te-IN",
        "hi": "hi-IN"
    }

    def __init__(self):

        self.recognizer = sr.Recognizer()

        self.recognizer.dynamic_energy_threshold = True

        self.recognizer.pause_threshold = 0.8

        self.recognizer.phrase_threshold = 0.2

        self.recognizer.non_speaking_duration = 0.5

    def recognize(
        self,
        wav_file,
        language="en"
    ):

        language_code = self.LANGUAGE_CODES.get(
            language,
            "en-IN"
        )

        print()
        print("=" * 60)
        print("SPEECH RECOGNITION")
        print("=" * 60)

        print(
            "Recognition language:",
            language_code
        )

        try:

            with sr.AudioFile(
                wav_file
            ) as source:

                audio = self.recognizer.record(
                    source
                )

            print(
                "Sending audio to Google Speech Recognition..."
            )

            text = self.recognizer.recognize_google(
                audio,
                language=language_code
            )

            return text

        except sr.UnknownValueError:

            print()
            print(
                "Google could not understand the recorded speech."
            )

            return ""

        except sr.RequestError as e:

            print()
            print(
                "Google Speech Recognition service error:"
            )

            print(e)

            return ""

        except Exception as e:

            print()
            print(
                "Speech recognition error:"
            )

            print(e)

            return ""


# ============================================================
# TEXT TO SPEECH
# ============================================================

class TextToSpeech:

    LANGUAGE_CODES = {
        "en": "en",
        "te": "te",
        "hi": "hi"
    }

    def speak(
        self,
        text,
        language="en"
    ):

        if not text:
            return

        language_code = self.LANGUAGE_CODES.get(
            language,
            "en"
        )

        try:

            mp3_file = os.path.join(
                tempfile.gettempdir(),
                "mediassist_response.mp3"
            )

            tts = gTTS(
                text=text,
                lang=language_code,
                slow=False
            )

            tts.save(mp3_file)

            print(
                "Playing voice response..."
            )

            os.startfile(
                mp3_file
            )

        except Exception as e:

            print(
                "Text-to-speech error:",
                e
            )


# ============================================================
# COMPLETE LISTEN FUNCTION
# ============================================================

def listen(
    language="en",
    duration=8
):

    recorder = WindowsRecorder()

    speech = SpeechToText()

    temp_dir = tempfile.gettempdir()

    original_wav = os.path.join(
        temp_dir,
        "mediassist_original.wav"
    )

    converted_wav = os.path.join(
        temp_dir,
        "mediassist_16khz.wav"
    )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    recorder.record(
        original_wav,
        duration
    )

    # --------------------------------------------------------
    # CONVERT
    # --------------------------------------------------------

    convert_audio(
        original_wav,
        converted_wav
    )

    # --------------------------------------------------------
    # SPEECH RECOGNITION
    # --------------------------------------------------------

    text = speech.recognize(
        converted_wav,
        language
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RECOGNIZED SPEECH")
    print("=" * 60)

    if text:

        print(
            text
        )

    else:

        print(
            "No speech recognized."
        )

    print("=" * 60)

    # Keep WAV files for debugging
    print()
    print("Debug files:")
    print(original_wav)
    print(converted_wav)

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MEDIASSIST VOICE TEST")
    print("=" * 60)

    print()
    print("Select language:")

    print(
        "1. English"
    )

    print(
        "2. Telugu"
    )

    print(
        "3. Hindi"
    )

    choice = input(
        "\nChoose language [1]: "
    ).strip()

    if choice == "2":

        language = "te"

    elif choice == "3":

        language = "hi"

    else:

        language = "en"

    print()
    print(
        "Selected language:",
        language
    )

    # --------------------------------------------------------
    # LISTEN
    # --------------------------------------------------------

    text = listen(
        language=language,
        duration=8
    )

    # --------------------------------------------------------
    # TTS TEST
    # --------------------------------------------------------

    if text:

        print()
        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        print(
            "User said:",
            text
        )

        print(
            "=" * 60
        )

        answer = input(
            "\nPlay the recognized text? (yes/no): "
        ).strip().lower()

        if answer in {
            "yes",
            "y"
        }:

            tts = TextToSpeech()

            tts.speak(
                text,
                language
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()