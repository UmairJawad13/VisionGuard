"""Simple test without PyTorch to check other components"""
print("Testing basic imports...")

# Test imports that don't need PyTorch
try:
    import cv2
    print("✓ OpenCV")
except:
    print("✗ OpenCV")

try:
    import pyttsx3
    print("✓ pyttsx3")
except:
    print("✗ pyttsx3")

try:
    import customtkinter
    print("✓ customtkinter")
except:
    print("✗ customtkinter")

try:
    import numpy
    print("✓ numpy")
except:
    print("✗ numpy")

try:
    import matplotlib
    print("✓ matplotlib")
except:
    print("✗ matplotlib")

try:
    import requests
    print("✓ requests")
except:
    print("✗ requests")

print("\nYou need to install Visual C++ Redistributable for PyTorch to work.")
print("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
