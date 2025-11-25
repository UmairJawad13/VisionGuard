"""
Visual Question Answering module using Ollama + LLaVA
"""

import requests
import base64
import cv2
import numpy as np
import time
from utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL, VQA_PROMPT
from utils.logger import PerformanceLogger


class Assistant:
    """VQA assistant for deep scene understanding using LLaVA via Ollama"""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.perf_logger = PerformanceLogger()
        self.last_response = None
        
        # Check if Ollama is running
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                print(f"[ASSISTANT] Connected to Ollama successfully")
                print(f"[ASSISTANT] Available models: {model_names}")
                
                # Check if our model is available
                if any(self.model in name for name in model_names):
                    print(f"[ASSISTANT] Model '{self.model}' is available")
                else:
                    print(f"[ASSISTANT WARNING] Model '{self.model}' not found!")
                    print(f"[ASSISTANT] Please run: ollama pull {self.model}")
            else:
                print(f"[ASSISTANT WARNING] Ollama responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[ASSISTANT ERROR] Cannot connect to Ollama at {self.base_url}")
            print(f"[ASSISTANT] Please ensure Ollama is running:")
            print(f"[ASSISTANT]   1. Install Ollama from https://ollama.ai")
            print(f"[ASSISTANT]   2. Run: ollama serve")
            print(f"[ASSISTANT]   3. Run: ollama pull {self.model}")
        except Exception as e:
            print(f"[ASSISTANT ERROR] Connection check failed: {e}")
    
    def _encode_image(self, frame):
        """
        Encode frame to base64 for API
        
        Args:
            frame: OpenCV image (numpy array)
        
        Returns:
            Base64 encoded string
        """
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    
    def describe_scene(self, frame, custom_prompt=None):
        """
        Generate detailed scene description for visually impaired user
        
        Args:
            frame: Input image frame (numpy array)
            custom_prompt: Optional custom prompt (uses default VQA_PROMPT if None)
        
        Returns:
            str: Scene description or error message
        """
        start_time = time.time()
        
        try:
            # Encode image
            base64_image = self._encode_image(frame)
            
            # Prepare prompt
            prompt = custom_prompt if custom_prompt else VQA_PROMPT
            
            # Prepare API request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            print("[ASSISTANT] Sending request to LLaVA...")
            
            # Send request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180  # LLaVA can be slow, especially on first run
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', 'No response generated')
                
                # Log inference time
                inference_time = time.time() - start_time
                self.perf_logger.log_inference_time("llm", inference_time)
                
                print(f"[ASSISTANT] Scene described in {inference_time:.2f}s")
                self.last_response = description
                
                return description
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                print(f"[ASSISTANT ERROR] {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out. LLaVA might be processing slowly."
            print(f"[ASSISTANT ERROR] {error_msg}")
            return error_msg
            
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama. Please ensure it's running."
            print(f"[ASSISTANT ERROR] {error_msg}")
            return error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[ASSISTANT ERROR] {error_msg}")
            return error_msg
    
    def answer_question(self, frame, question):
        """
        Answer a specific question about the scene
        
        Args:
            frame: Input image frame
            question: User's question
        
        Returns:
            str: Answer to the question
        """
        custom_prompt = f"You are assisting a visually impaired person. Answer this question about the image: {question}"
        return self.describe_scene(frame, custom_prompt)
    
    def identify_hazards(self, frame):
        """
        Focus specifically on identifying hazards
        
        Args:
            frame: Input image frame
        
        Returns:
            str: Hazard description
        """
        hazard_prompt = """You are assisting a visually impaired person. 
        Focus ONLY on potential hazards and obstacles in this scene:
        - People or vehicles that might be in the path
        - Stairs, curbs, or elevation changes
        - Obstacles like furniture, poles, or barriers
        - Any dangerous elements
        
        Be concise and prioritize the most immediate hazards."""
        
        return self.describe_scene(frame, hazard_prompt)
    
    def read_environment(self, frame):
        """
        Describe the general environment and atmosphere
        
        Args:
            frame: Input image frame
        
        Returns:
            str: Environment description
        """
        env_prompt = """You are assisting a visually impaired person.
        Describe the general environment:
        - Indoor or outdoor?
        - Type of location (street, room, store, etc.)
        - Lighting conditions
        - General atmosphere and surroundings
        
        Keep it brief and useful for navigation."""
        
        return self.describe_scene(frame, env_prompt)
    
    def get_last_response(self):
        """Get the last generated response"""
        return self.last_response
    
    def is_available(self):
        """Check if Ollama service is currently available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
