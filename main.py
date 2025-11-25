"""
VisionGuard - Main Application
Computer Vision Assistance for Visually Impaired Users
"""

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from modules.navigator import Navigator
from modules.assistant import Assistant
from modules.reader import Reader
from modules.audio import get_audio_manager
from utils.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, UI_BG_COLOR, UI_FG_COLOR,
    UI_BUTTON_COLOR, UI_BUTTON_HOVER, CAMERA_INDEX
)


class VisionGuardApp:
    """Main application class for VisionGuard"""
    
    def __init__(self):
        # Initialize main window
        self.window = ctk.CTk()
        self.window.title("VisionGuard - Navigation Assistant")
        self.window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        # Initialize modules
        print("[MAIN] Initializing modules...")
        self.audio = get_audio_manager()
        self.navigator = Navigator(use_finetuned=False)  # Will auto-use finetuned if exists
        self.assistant = Assistant()
        self.reader = Reader()
        
        # State variables
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.show_debug = True
        self.last_warning_time = 0
        self.warning_cooldown = 2.0  # Seconds between warnings
        self.pause_warnings = False  # Pause warnings during text reading or scene description
        
        # UI components
        self.video_label = None
        self.status_label = None
        self.info_text = None
        
        # Setup UI
        self._setup_ui()
        
        # Bind keyboard shortcuts
        self._setup_keyboard_bindings()
        
        # Announce startup
        self.audio.speak("VisionGuard starting up", priority=True)
        
        print("[MAIN] Application initialized successfully")
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ctk.CTkFrame(self.window, fg_color=UI_BG_COLOR)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="VisionGuard - Navigation Assistant",
            font=("Arial", 24, "bold"),
            text_color=UI_FG_COLOR
        )
        title.pack(pady=10)
        
        # Video frame
        video_frame = ctk.CTkFrame(main_frame, fg_color="#1a1a1a")
        video_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.video_label = ctk.CTkLabel(video_frame, text="Camera feed will appear here")
        self.video_label.pack(fill="both", expand=True)
        
        # Control buttons frame
        button_frame = ctk.CTkFrame(main_frame, fg_color=UI_BG_COLOR)
        button_frame.pack(pady=10)
        
        # Start/Stop button
        self.start_button = ctk.CTkButton(
            button_frame,
            text="Start Camera",
            command=self.toggle_camera,
            width=150,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#00AA00",
            hover_color="#00DD00"
        )
        self.start_button.pack(side="left", padx=5)
        
        # Scene description button
        scene_button = ctk.CTkButton(
            button_frame,
            text="Describe Scene (SPACE)",
            command=self.describe_scene,
            width=200,
            height=40,
            font=("Arial", 14),
            fg_color=UI_BUTTON_COLOR,
            hover_color=UI_BUTTON_HOVER
        )
        scene_button.pack(side="left", padx=5)
        
        # Read text button
        read_button = ctk.CTkButton(
            button_frame,
            text="Read Text (R)",
            command=self.read_text,
            width=150,
            height=40,
            font=("Arial", 14),
            fg_color=UI_BUTTON_COLOR,
            hover_color=UI_BUTTON_HOVER
        )
        read_button.pack(side="left", padx=5)
        
        # Debug toggle button
        debug_button = ctk.CTkButton(
            button_frame,
            text="Toggle Debug (D)",
            command=self.toggle_debug,
            width=150,
            height=40,
            font=("Arial", 14),
            fg_color=UI_BUTTON_COLOR,
            hover_color=UI_BUTTON_HOVER
        )
        debug_button.pack(side="left", padx=5)
        
        # Status bar
        status_frame = ctk.CTkFrame(main_frame, fg_color="#1a1a1a")
        status_frame.pack(pady=5, fill="x")
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Status: Ready | Press START to begin",
            font=("Arial", 12),
            text_color=UI_FG_COLOR
        )
        self.status_label.pack(pady=5)
        
        # Info panel
        info_frame = ctk.CTkFrame(main_frame, fg_color="#1a1a1a")
        info_frame.pack(pady=5, fill="both", expand=False)
        
        info_title = ctk.CTkLabel(
            info_frame,
            text="Keyboard Controls:",
            font=("Arial", 12, "bold"),
            text_color=UI_FG_COLOR
        )
        info_title.pack(pady=5)
        
        controls_text = """
        Q - Quit Application  |  SPACE - Describe Scene  |  R - Read Text
        D - Toggle Debug Mode  |  A - Toggle Audio  |  L - Toggle Logging
        """
        
        self.info_text = ctk.CTkLabel(
            info_frame,
            text=controls_text,
            font=("Arial", 10),
            text_color="#AAAAAA"
        )
        self.info_text.pack(pady=5)
    
    def _setup_keyboard_bindings(self):
        """Setup keyboard shortcuts"""
        self.window.bind('<q>', lambda e: self.quit_app())
        self.window.bind('<Q>', lambda e: self.quit_app())
        self.window.bind('<space>', lambda e: self.describe_scene())
        self.window.bind('<r>', lambda e: self.read_text())
        self.window.bind('<R>', lambda e: self.read_text())
        self.window.bind('<d>', lambda e: self.toggle_debug())
        self.window.bind('<D>', lambda e: self.toggle_debug())
        self.window.bind('<a>', lambda e: self.toggle_audio())
        self.window.bind('<A>', lambda e: self.toggle_audio())
        self.window.bind('<l>', lambda e: self.toggle_logging())
        self.window.bind('<L>', lambda e: self.toggle_logging())
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera and processing"""
        print("[MAIN] Starting camera...")
        
        try:
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            
            if not self.camera.isOpened():
                self.audio.speak("Error: Cannot open camera")
                self.update_status("Error: Camera not accessible")
                return
            
            self.is_running = True
            self.start_button.configure(text="Stop Camera", fg_color="#AA0000", hover_color="#DD0000")
            self.update_status("Status: Camera active - Scanning environment")
            self.audio.speak("Camera started. System ready.")
            
            # Start video processing thread
            video_thread = threading.Thread(target=self._process_video, daemon=True)
            video_thread.start()
            
            print("[MAIN] Camera started successfully")
            
        except Exception as e:
            print(f"[MAIN ERROR] Failed to start camera: {e}")
            self.audio.speak("Error starting camera")
            self.update_status(f"Error: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera"""
        print("[MAIN] Stopping camera...")
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_button.configure(text="Start Camera", fg_color="#00AA00", hover_color="#00DD00")
        self.update_status("Status: Camera stopped")
        self.audio.speak("Camera stopped")
        
        # Clear video display
        self.video_label.configure(image=None, text="Camera feed stopped")
        
        print("[MAIN] Camera stopped")
    
    def _process_video(self):
        """Main video processing loop (runs in separate thread)"""
        while self.is_running:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("[MAIN ERROR] Failed to read frame")
                    break
                
                # Store current frame for other operations
                self.current_frame = frame.copy()
                
                # Process frame with navigator
                processed_frame, detections, analysis = self.navigator.process_frame(
                    frame, show_debug=self.show_debug
                )
                
                # Handle warnings
                self._handle_warnings(analysis)
                
                # Update display
                self._update_video_display(processed_frame)
                
                # Small delay to control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"[MAIN ERROR] Video processing error: {e}")
                break
        
        print("[MAIN] Video processing stopped")
    
    def _handle_warnings(self, analysis):
        """Handle audio warnings for detected hazards"""
        # Skip warnings if paused (during text reading or scene description)
        if self.pause_warnings:
            return
            
        current_time = time.time()
        
        # Check if enough time has passed since last warning
        if current_time - self.last_warning_time < self.warning_cooldown:
            return
        
        # Get priority warning
        priority_warning = analysis.get('priority_warning')
        
        if priority_warning:
            self.audio.speak(priority_warning, priority=True)
            self.last_warning_time = current_time
            self.update_status(f"Warning: {priority_warning}")
    
    def _update_video_display(self, frame):
        """Update the video display in the GUI"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display area
            display_height = 400
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_width = int(display_height * aspect_ratio)
            rgb_frame = cv2.resize(rgb_frame, (display_width, display_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"[MAIN ERROR] Display update error: {e}")
    
    def describe_scene(self):
        """Request scene description from LLaVA"""
        if self.current_frame is None:
            self.audio.speak("No camera feed available")
            return
        
        self.pause_warnings = True  # Pause hazard warnings
        self.audio.speak("Analyzing scene, please wait")
        self.update_status("Status: Analyzing scene...")
        
        # Run in separate thread to avoid blocking UI
        def analyze():
            description = self.assistant.describe_scene(self.current_frame)
            print(f"[MAIN] LLaVA response: {description[:100]}...")  # Print first 100 chars
            # Use blocking speech to ensure it completes
            self.audio.speak_blocking(description)
            time.sleep(1)  # Give 1 second buffer before resuming warnings
            self.pause_warnings = False  # Resume warnings
            self.update_status("Status: Scene description complete")
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def read_text(self):
        """Read text from current frame using OCR"""
        if self.current_frame is None:
            self.audio.speak("No camera feed available")
            return
        
        self.pause_warnings = True  # Pause hazard warnings
        self.update_status("Status: Reading text...")
        
        # Run in separate thread
        def read():
            try:
                _, detected_texts = self.reader.read_text(self.current_frame)
                text_message = self.reader.format_text_for_speech(detected_texts)
                print(f"[MAIN] Text detected: {text_message}")
                print(f"[MAIN] About to speak text using blocking method...")
                # Use blocking speech to ensure it completes
                self.audio.speak_blocking(text_message)
                print(f"[MAIN] Finished speaking text")
                time.sleep(1)  # Give 1 second buffer before resuming warnings
                self.pause_warnings = False  # Resume warnings
                self.update_status("Status: Text reading complete")
            except Exception as e:
                print(f"[MAIN ERROR] Text reading failed: {e}")
                import traceback
                traceback.print_exc()
                self.pause_warnings = False
        
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
    
    def toggle_debug(self):
        """Toggle debug visualization"""
        self.show_debug = not self.show_debug
        status = "enabled" if self.show_debug else "disabled"
        self.audio.speak(f"Debug mode {status}")
        self.update_status(f"Status: Debug mode {status}")
    
    def toggle_audio(self):
        """Toggle audio output"""
        enabled = self.audio.toggle()
        status = "enabled" if enabled else "disabled"
        self.update_status(f"Status: Audio {status}")
    
    def toggle_logging(self):
        """Toggle debug logging"""
        enabled = self.navigator.debug_logger.toggle()
        status = "enabled" if enabled else "disabled"
        self.audio.speak(f"Logging {status}")
        self.update_status(f"Status: Debug logging {status}")
    
    def update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.configure(text=message)
    
    def quit_app(self):
        """Cleanup and quit application"""
        print("[MAIN] Shutting down...")
        self.audio.speak("Shutting down VisionGuard")
        
        self.stop_camera()
        
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass  # Ignore OpenCV GUI errors on Windows
        self.audio.shutdown()
        
        # Save performance report
        try:
            import os
            report_path = os.path.join("logs", "performance_report.json")
            self.navigator.save_performance_report(report_path)
            print(f"[MAIN] Performance report saved to {report_path}")
        except Exception as e:
            print(f"[MAIN] Could not save performance report: {e}")
        
        time.sleep(1)  # Give time for final speech
        self.window.quit()
        self.window.destroy()
    
    def run(self):
        """Start the application"""
        print("[MAIN] Starting GUI...")
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.window.mainloop()


def main():
    """Main entry point"""
    print("=" * 60)
    print("VisionGuard - Navigation Assistant for Visually Impaired")
    print("=" * 60)
    
    app = VisionGuardApp()
    app.run()


if __name__ == "__main__":
    main()
