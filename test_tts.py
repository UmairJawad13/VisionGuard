"""
Simple TTS test to verify pyttsx3 is working
"""
import pyttsx3

print("Initializing TTS engine...")
engine = pyttsx3.init()

# Get current settings
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
voices = engine.getProperty('voices')

print(f"Rate: {rate}")
print(f"Volume: {volume}")
print(f"Available voices: {len(voices)}")
for i, voice in enumerate(voices):
    print(f"  {i}: {voice.name}")

# Test speech
print("\nTesting speech output...")
print("You should hear: 'Testing one two three'")
engine.say("Testing one two three")
engine.runAndWait()

print("\nTesting longer text...")
print("You should hear: 'I found 3 text items. Reading now: Vaseline. Deep. Restore'")
engine.say("I found 3 text items. Reading now: Vaseline. Deep. Restore")
engine.runAndWait()

print("\nIf you heard both messages, TTS is working correctly.")
print("If not, there may be an audio driver or output device issue.")
