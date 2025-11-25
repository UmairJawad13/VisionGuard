"""
Audio output module using pyttsx3 for offline text-to-speech
"""

import pyttsx3
import threading
from queue import Queue
from utils.config import TTS_RATE, TTS_VOLUME, TTS_VOICE_INDEX


class AudioManager:
    """Manages text-to-speech audio output with queue for non-blocking speech"""
    
    def __init__(self):
        self.engine = None
        self.speech_queue = Queue()
        self.is_speaking = False
        self.enabled = True
        self._initialize_engine()
        
        # Start speech worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 engine with configured settings"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self.engine.setProperty('rate', TTS_RATE)
            self.engine.setProperty('volume', TTS_VOLUME)
            
            # Try to set voice (0 = default, 1 = alternate, usually female)
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > TTS_VOICE_INDEX:
                self.engine.setProperty('voice', voices[TTS_VOICE_INDEX].id)
            
            print(f"[AUDIO] TTS engine initialized successfully")
            print(f"[AUDIO] Rate: {TTS_RATE} WPM, Volume: {TTS_VOLUME}")
            
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to initialize TTS engine: {e}")
            self.enabled = False
    
    def _speech_worker(self):
        """Worker thread that processes speech queue"""
        while True:
            try:
                text = self.speech_queue.get()
                if text is None:  # Sentinel value to stop thread
                    break
                
                if self.enabled and self.engine:
                    self.is_speaking = True
                    print(f"[AUDIO] Speaking: {text[:50]}...")  # Debug output
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.is_speaking = False
                    print(f"[AUDIO] Finished speaking")  # Debug output
                
                self.speech_queue.task_done()
            except Exception as e:
                print(f"[AUDIO ERROR] Speech worker error: {e}")
                self.is_speaking = False
    
    def speak(self, text, priority=False):
        """
        Add text to speech queue
        
        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if not self.enabled:
            print(f"[AUDIO DISABLED] Would say: {text}")
            return
        
        print(f"[AUDIO] Queuing speech (priority={priority}): {text[:50]}...")  # Debug
        
        if priority:
            # Clear queue for priority messages
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except:
                    break
            # Stop current speech if possible
            if self.engine and self.is_speaking:
                try:
                    self.engine.stop()
                except:
                    pass
        
        self.speech_queue.put(text)
    
    def speak_blocking(self, text):
        """
        Speak text immediately and block until finished (for critical messages)
        
        Args:
            text: Text to speak
        """
        if not self.enabled:
            print(f"[AUDIO DISABLED] Would say: {text}")
            return
        
        try:
            # Create a new engine instance for blocking speech (thread-safe on Windows)
            print(f"[AUDIO] Speaking (blocking): {text[:50]}...")
            temp_engine = pyttsx3.init()
            temp_engine.setProperty('rate', TTS_RATE)
            temp_engine.setProperty('volume', 1.0)  # Max volume
            
            # Flag to track if speech is done
            speech_done = [False]
            
            def on_end(name, completed):
                speech_done[0] = True
            
            # Connect the event
            temp_engine.connect('finished-utterance', on_end)
            
            temp_engine.say(text)
            temp_engine.startLoop(False)
            
            # Manually pump the event loop
            while not speech_done[0]:
                temp_engine.iterate()
                import time
                time.sleep(0.01)
            
            temp_engine.endLoop()
            del temp_engine
            
            print(f"[AUDIO] Finished blocking speech")
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to speak: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def clear_queue(self):
        """Clear pending speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break
    
    def enable(self):
        """Enable audio output"""
        self.enabled = True
        self.speak("Audio enabled")
    
    def disable(self):
        """Disable audio output"""
        print("[AUDIO] Audio disabled")
        self.enabled = False
    
    def toggle(self):
        """Toggle audio on/off"""
        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled
    
    def set_rate(self, rate):
        """Change speech rate"""
        if self.engine:
            self.engine.setProperty('rate', rate)
            print(f"[AUDIO] Speech rate set to {rate} WPM")
    
    def set_volume(self, volume):
        """Change volume (0.0 to 1.0)"""
        if self.engine:
            volume = max(0.0, min(1.0, volume))  # Clamp to valid range
            self.engine.setProperty('volume', volume)
            print(f"[AUDIO] Volume set to {volume}")
    
    def list_voices(self):
        """List available voices"""
        if self.engine:
            voices = self.engine.getProperty('voices')
            print(f"\n[AUDIO] Available voices ({len(voices)}):")
            for i, voice in enumerate(voices):
                print(f"  {i}: {voice.name} ({voice.id})")
            return voices
        return []
    
    def shutdown(self):
        """Cleanup and shutdown audio engine"""
        self.speech_queue.put(None)  # Signal worker thread to stop
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        print("[AUDIO] Audio manager shutdown")


# Singleton instance
_audio_manager = None

def get_audio_manager():
    """Get or create the singleton AudioManager instance"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
    return _audio_manager
