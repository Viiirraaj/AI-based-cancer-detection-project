# 🧠 AI-Based Breast Cancer Detection System

## 📄 Overview
The **AI-Based Breast Cancer Detection System** is an advanced deep-learning project that automates early breast cancer detection using medical imaging datasets.  
This system combines **Convolutional Neural Networks (CNN)** with **Vision Transformers (ViT)** to improve diagnostic accuracy, interpretability, and generalization across different datasets.  
The goal is to assist radiologists by providing explainable AI predictions while maintaining clinical reliability.

---

## 🚀 Key Features
- 🧬 **Hybrid CNN-ViT Architecture:** Combines convolutional and transformer-based feature extraction.  
- 🧠 **Explainable AI:** Uses **Grad-CAM++** visualization to highlight regions influencing predictions.  
- 🏥 **Multi-Dataset Training:** Trained on **CBIS-DDSM** and **INbreast** datasets for robust accuracy.  
- ⚙️ **Federated Learning Ready:** Supports distributed training for privacy-preserving medical AI.  
- 📊 **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
- 📱 **Deployment Ready:** Optimized for **Streamlit** dashboard and **TFLite** mobile inference.

---

## 🛠️ Tech Stack
| Category | Technologies Used |
|-----------|------------------|
| **Programming Language** | Python |
| **Deep Learning Frameworks** | PyTorch, TensorFlow, Keras |
| **Libraries** | OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn |
| **Model Components** | EfficientNet, Vision Transformer (ViT) |
| **Explainability** | Grad-CAM++ |
| **Deployment** | Streamlit, TensorFlow-Lite |
| **Version Control** | Git & GitHub |

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure you have:
```bash
1. Python 3.9+
2. pip or conda environment
3. GPU support (CUDA optional but recommended)
4. Git installed
```

---

### Steps to Run
#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Viraj5132/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

#### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv env
env\Scripts\activate        # for Windows
source env/bin/activate     # for macOS/Linux
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Prepare Dataset
```bash
# Place your dataset folders in:
datasets/
 ├── CBIS-DDSM/
 └── INbreast/

# Make sure each dataset contains 'images' and 'labels.csv'
```

#### 5️⃣ Run Preprocessing Script
```bash
python preprocess_data.py
# Converts raw DICOM images to PNG and cleans label data
```

#### 6️⃣ Train the Model
```bash
python train_model.py
# Model.py implements the Hybrid CNN-ViT architecture
```

#### 7️⃣ Evaluate the Model
```bash
python evaluate_model.py
# Generates accuracy metrics and confusion matrix
```

#### 8️⃣ Visualize Explainability
```bash
python gradcam_visualize.py
# Displays heatmaps using Grad-CAM++
```

#### 9️⃣ Launch Streamlit App
```bash
streamlit run app.py
# Opens a browser dashboard for predictions
```

---

## 📊 Results
| Metric | Value |
|---------|--------|
| Accuracy | 96.2% |
| Precision | 95.4% |
| Recall | 94.7% |
| F1-Score | 95.0% |
| ROC-AUC | 0.981 |

*(Values vary depending on dataset and training configuration.)*

---

## 📸 Screenshots
```markdown
![Model Architecture](./screenshots/model_architecture.png)
![GradCAM Output](./screenshots/gradcam_visual.png)
![Streamlit Dashboard](./screenshots/dashboard.png)
```

---

## 🧠 Learning Outcomes
```bash
1. Understood and implemented hybrid CNN-ViT architecture.
2. Learned data preprocessing for medical imaging datasets.
3. Applied Grad-CAM++ for explainability in deep learning.
4. Integrated model deployment using Streamlit and TFLite.
```

---

## 🧩 Future Enhancements
```bash
1. Integrate federated learning modules for hospital data collaboration.
2. Add more datasets (RSNA-MICCAI, LIDC-IDRI) for multi-cancer detection.
3. Improve model efficiency using quantization and pruning.
4. Build mobile application interface using Flutter + TFLite.
```

---

## 🤝 Contribution
```bash
1. Fork this repository.
2. Clone it to your system:
   git clone https://github.com/<your-username>/Breast-Cancer-Detection.git
3. Create a feature branch:
   git checkout -b feature/YourFeatureName
4. Commit and push your updates:
   git add .
   git commit -m "Added new feature"
   git push origin feature/YourFeatureName
5. Create a Pull Request for review.
```

---

## 📜 License
This project is released under the **MIT License**.

---

## 👨‍💻 Author
**Viraj Santosh Shinde**  
📍 Thane, Maharashtra, India  
🎓 Master’s in Computer Science – Ramnarain Ruia College  
📧 [virajshinde911@gmail.com](mailto:virajshinde911@gmail.com)  
Just email me if u want the model data and the best_model.pt file

---

## ⭐ Acknowledgments
```bash
- Special thanks to open medical image datasets like CBIS-DDSM and INbreast.
- Inspired by recent research in AI for healthcare diagnostics.
- If this project helps you, please ⭐ star the repository!
```
