#!/usr/bin/env python3
"""
Installation script for Real-Time Screen Analyzer
Run this first to install all dependencies
"""

import subprocess
import sys

def run_command(cmd):
    """Run shell command and check result"""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Command failed: {result.stderr}")
    else:
        print("✅ Success")
    return result.returncode == 0

def main():
    """Install all required dependencies"""
    print("🚀 Installing Real-Time Screen Analyzer Dependencies")
    print("="*60)
    
    # Update pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Core dependencies
    core_packages = [
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "pillow",
        "keyboard",
        "mss",
        "pyautogui"
    ]
    
    print("\n📦 Installing core packages...")
    for package in core_packages:
        run_command(f"{sys.executable} -m pip install {package}")
    
    # Face detection
    print("\n👤 Installing face detection packages...")
    face_packages = [
        "ultralytics",  # YOLOv8
        "mediapipe"     # Face landmarks
    ]
    for package in face_packages:
        run_command(f"{sys.executable} -m pip install {package}")
    
    # Deep learning (optional)
    print("\n🤖 Installing deep learning packages...")
    print("Note: PyTorch installation varies by system")
    print("For CPU-only:")
    print(f"  {sys.executable} -m pip install torch torchvision torchaudio")
    print("\nFor CUDA 11.8:")
    print(f"  {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    choice = input("\nInstall PyTorch? (y/n): ")
    if choice.lower() == 'y':
        run_command(f"{sys.executable} -m pip install torch torchvision")
    
    # Audio for lip-sync (optional)
    print("\n👄 Installing audio packages for lip-sync...")
    audio_packages = [
        "pyaudio",
        "sounddevice",
        "librosa"
    ]
    for package in audio_packages:
        run_command(f"{sys.executable} -m pip install {package}")
    
    print("\n" + "="*60)
    print("✅ Installation complete!")
    print("\n🎯 To run the analyzer:")
    print("   python real_time_analyzer.py")
    print("\n🎮 Controls:")
    print("   ESC: Exit")
    print("   P: Pause/Resume")
    print("   O: Toggle overlay")
    print("   S: Save screenshot")
    print("   R: Reset statistics")
    print("="*60)

if __name__ == "__main__":
    main()