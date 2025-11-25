"""
Test audio output with volume control
"""
import pyttsx3
import time

print("Testing TTS with different volumes...")
print("=" * 50)

for volume in [0.5, 0.7, 0.9, 1.0]:
    print(f"\nTesting with volume: {volume}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', volume)
    
    text = f"Testing volume at {int(volume * 100)} percent"
    print(f"Speaking: {text}")
    
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    del engine
    
    time.sleep(0.5)

print("\n" + "=" * 50)
print("Audio test complete.")
print("\nIf you heard ANY of these messages, audio is working.")
print("If you heard NONE, check:")
print("1. Windows Volume Mixer (search for 'Volume Mixer' in Start menu)")
print("2. Right-click speaker icon > Open Volume mixer")
print("3. Make sure Python is not muted")
print("4. Check your default playback device (speaker icon > Sounds > Playback)")
