# RGB-IR Human Detection Demo

A Flask-based web application developed as part of my undergraduate thesis to compare RGB, Thermal (IR), and Early Fusion YOLO models for human detection under low-light conditions.

---

## 📌 Overview

This project was developed as part of my undergraduate thesis in Computer Engineering at Düzce University.

The application provides an interactive web interface for evaluating and comparing multiple human detection models trained on RGB, Thermal (IR), and fused image modalities under low-light conditions.

Users can upload images, select different detection models, adjust confidence thresholds, and visually compare detection results.

---

## ✨ Features

- Human detection on RGB and Thermal (IR) images
- Multiple YOLO-based model selection
- Early Fusion model support
- Fine-tuned model comparison
- Adjustable confidence threshold
- Interactive Flask web interface
- Visualization of detection results
- Comparative analysis between different models

---

## 🧠 Supported Models

- YOLOv5s RGB
- YOLOv5s IR
- YOLOv26s RGB
- YOLOv26s IR
- Early Fusion
- Early Fusion (Fine-Tuned)

---

## 🛠 Technologies

- Python
- Flask
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```
RGB-IR-Human-Detection-Demo/
│
├── app.py
├── requirements.txt
├── models/
├── static/
├── templates/
├── uploads/
├── results/
└── README.md
```

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/rgb-ir-human-detection-demo.git

cd rgb-ir-human-detection-demo
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
python app.py
```

Open your browser.

```
http://127.0.0.1:5000
```

---

## 📖 Usage

1. Select a detection model.
2. Upload an RGB or Thermal image.
3. Adjust the confidence threshold.
4. Start the detection process.
5. Analyze and compare the results.

---

## 📷 Screenshots

### Home Page
<p align="center">
  <img src="https://github.com/user-attachments/assets/b19beb5c-bf64-4cb5-93d2-a7198fe4762c" alt="Home Page" width="600" style="max-width: 100%; height: auto; display: block; margin: 0 auto;"/>
</p>

---

### Detection & Model Comparison
<p align="center">
  <table align="center" border="0" cellspacing="10" cellpadding="0">
    <tr>
      <td align="center" valign="top" width="33%">
        <p><b>Detection Result</b></p>
        <img src="https://github.com/user-attachments/assets/e255baf8-70a1-48f3-b993-e035fa63a5b7" alt="Detection Result" width="100%" style="border-radius: 5px;"/>
      </td>
      <td align="center" valign="top" width="33%">
        <p><b>Model Comparison 1</b></p>
        <img src="https://github.com/user-attachments/assets/d2b1d33c-5dba-48b0-abe2-86dfc83c7901" alt="Model Comparison 1" width="100%" style="border-radius: 5px;"/>
      </td>
      <td align="center" valign="top" width="33%">
        <p><b>Model Comparison 2</b></p>
        <img src="https://github.com/user-attachments/assets/f1dde6f5-2872-4ef0-9eba-375120ea8132" alt="Model Comparison 2" width="100%" style="border-radius: 5px;"/>
      </td>
    </tr>
  </table>
</p>

> 

---

## 🎓 Thesis

This application was developed as part of my undergraduate thesis:

**Human Detection Under Low-Light Conditions Using RGB and Thermal Images**

The project investigates the effectiveness of RGB, Thermal, and Early Fusion approaches for robust human detection under challenging illumination conditions.

---

## 📊 Future Improvements

- Real-time webcam support
- Video-based detection
- ONNX deployment
- TensorRT optimization
- Docker support
- Additional multimodal datasets

---

## 👩‍💻 Author

**Gizem Efe**

Computer Engineer

- GitHub: https://github.com/gizembm
- LinkedIn: https://linkedin.com/in/gizemefe

---

## 📄 License

This project was developed for academic and research purposes.

Commercial use is not permitted without permission.
