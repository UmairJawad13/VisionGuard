"""
VisionGuard Web App - Streamlit Version
Computer Vision Assistance for Visually Impaired Users
Public demo without camera access or TTS
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules with error handling
try:
    from modules.navigator import Navigator
    from modules.reader import Reader
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error loading modules: {e}")
    st.info("Please ensure all required files are uploaded to GitHub")
    MODULES_LOADED = False

# Page configuration
st.set_page_config(
    page_title="VisionGuard - AI Vision Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #FFE5E5;
        border-left: 5px solid #FF0000;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load AI models (cached to avoid reloading)"""
    if not MODULES_LOADED:
        st.error("Cannot load models - modules import failed")
        st.stop()
        
    with st.spinner("Loading AI models... This may take a minute on first run."):
        navigator = Navigator(use_finetuned=False)
        reader = Reader()
    return navigator, reader

def process_image_detection(image, navigator, confidence_threshold, show_debug):
    """Process image for object detection"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process with navigator
    processed_frame, detections, analysis = navigator.process_frame(
        img_bgr, 
        show_debug=show_debug
    )
    
    # Convert back to RGB for display
    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_rgb, detections, analysis

def process_image_ocr(image, reader):
    """Process image for text detection"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process with reader
    processed_frame, detected_texts = reader.read_text(img_bgr, show_debug=True)
    
    # Convert back to RGB for display
    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_rgb, detected_texts

def main():
    # Header
    st.markdown('<div class="main-header">👁️ VisionGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Vision Assistance for Everyone</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>🌟 About VisionGuard:</strong><br>
        This application uses advanced AI to help visually impaired users navigate their environment safely.
        Upload an image to:
        <ul>
            <li><strong>Detect Objects:</strong> Identify hazards, people, vehicles, and obstacles</li>
            <li><strong>Read Text:</strong> Extract and read text from images (signs, labels, documents)</li>
            <li><strong>Analyze Scenes:</strong> Get detailed descriptions of your surroundings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Object Detection", "Text Reading (OCR)", "Both"],
            index=0
        )
        
        st.markdown("---")
        
        # Detection settings
        if mode in ["Object Detection", "Both"]:
            st.subheader("Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher = fewer but more confident detections"
            )
            show_debug = st.checkbox("Show Debug Info", value=True)
        else:
            confidence_threshold = 0.5
            show_debug = True
        
        st.markdown("---")
        
        # About section
        st.subheader("📋 About")
        st.markdown("""
        **VisionGuard** is an AI-powered computer vision system designed to assist visually impaired users.
        
        **Technologies:**
        - YOLOv8 (Object Detection)
        - EasyOCR (Text Recognition)
        - PyTorch (Deep Learning)
        
        **Created for:** UOW Image Processing & Computer Vision Assignment
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            st.info(f"📐 Image size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None:
            # Load models
            try:
                navigator, reader = load_models()
                
                # Process based on mode
                if mode == "Object Detection":
                    with st.spinner("Detecting objects..."):
                        processed_img, detections, analysis = process_image_detection(
                            image, navigator, confidence_threshold, show_debug
                        )
                    
                    # Display processed image
                    st.image(processed_img, caption="Detected Objects", use_column_width=True)
                    
                    # Display detection results
                    if detections:
                        st.success(f"✅ Detected {len(detections)} objects")
                        
                        # Show warnings
                        if analysis.get('priority_warning'):
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>⚠️ Warning:</strong> {analysis['priority_warning']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detection details
                        with st.expander("📊 Detection Details"):
                            for i, det in enumerate(detections):
                                st.write(f"**{i+1}. {det['class_name'].upper()}**")
                                st.write(f"   - Confidence: {det['confidence']:.2%}")
                                st.write(f"   - Position: {det['position']}")
                                st.write(f"   - Distance: {det['distance']}")
                    else:
                        st.warning("No objects detected in this image.")
                
                elif mode == "Text Reading (OCR)":
                    with st.spinner("Reading text..."):
                        processed_img, detected_texts = process_image_ocr(image, reader)
                    
                    # Display processed image
                    st.image(processed_img, caption="Detected Text", use_column_width=True)
                    
                    # Display OCR results
                    if detected_texts:
                        st.success(f"✅ Found {len(detected_texts)} text regions")
                        
                        # Show text
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("**📝 Extracted Text:**")
                        for i, text_item in enumerate(detected_texts):
                            st.write(f"{i+1}. {text_item['text']} (Confidence: {text_item['confidence']:.2%})")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Full text
                        full_text = " ".join([t['text'] for t in detected_texts])
                        with st.expander("📄 Full Text"):
                            st.text_area("", full_text, height=150)
                    else:
                        st.warning("No text detected in this image.")
                
                else:  # Both modes
                    with st.spinner("Analyzing image..."):
                        # Object detection
                        processed_img_det, detections, analysis = process_image_detection(
                            image, navigator, confidence_threshold, show_debug
                        )
                        
                        # OCR
                        processed_img_ocr, detected_texts = process_image_ocr(image, reader)
                    
                    # Display both results
                    tab1, tab2 = st.tabs(["🎯 Objects", "📝 Text"])
                    
                    with tab1:
                        st.image(processed_img_det, use_column_width=True)
                        
                        if detections:
                            st.success(f"✅ Detected {len(detections)} objects")
                            
                            if analysis.get('priority_warning'):
                                st.markdown(f"""
                                <div class="warning-box">
                                    <strong>⚠️ Warning:</strong> {analysis['priority_warning']}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with st.expander("📊 Detection Details"):
                                for i, det in enumerate(detections):
                                    st.write(f"**{i+1}. {det['class_name'].upper()}**")
                                    st.write(f"   - Confidence: {det['confidence']:.2%}")
                                    st.write(f"   - Position: {det['position']}")
                                    st.write(f"   - Distance: {det['distance']}")
                        else:
                            st.warning("No objects detected.")
                    
                    with tab2:
                        st.image(processed_img_ocr, use_column_width=True)
                        
                        if detected_texts:
                            st.success(f"✅ Found {len(detected_texts)} text regions")
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown("**📝 Extracted Text:**")
                            for i, text_item in enumerate(detected_texts):
                                st.write(f"{i+1}. {text_item['text']} (Confidence: {text_item['confidence']:.2%})")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("No text detected.")
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)
        else:
            st.info("👆 Upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>VisionGuard © 2025 | Built with ❤️ using Streamlit, YOLOv8, and EasyOCR</p>
        <p>For academic purposes | University of Wollongong</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
