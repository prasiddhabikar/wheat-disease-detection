# 🌾 Early Wheat Leaf Disease Detection Engine

An AI-driven web application designed to help farmers and agronomists perform real-time diagnostic tissue analysis on wheat crops. This system uses a deep convolutional neural network to identify early-stage leaf diseases instantly through a secure web workspace.

## 🚀 System Architecture
- **Frontend Workspace**: A responsive user dashboard with integrated local focus/clarity validation guardrails, authenticated via Firebase Google OAuth, and deployed globally via Firebase Hosting.
- **AI Core Backend**: A high-performance FastAPI server running inference engines over a customized, lightweight PyTorch MobileNetV3 small architecture.

## 📊 Target Diagnostics
The neural network evaluates crop tissue and classifies images into one of four distinct categories:
1. **Healthy**: Disease-free, optimal structural integrity.
2. **Leaf Rust**: Early-to-late stage fungal spore colonization.
3. **Powdery Mildew**: Fungal mycelium surface patches.
4. **Yellow Rust**: Striated pustule leaf tissue disruption.

## 🛠️ Technology Stack
- **Languages**: Python, JavaScript, HTML5, CSS3
- **Deep Learning Framework**: PyTorch, Torchvision
- **Backend Architecture**: FastAPI, Uvicorn Server
- **Cloud Infrastructure**: Firebase (Auth & Hosting)
-
