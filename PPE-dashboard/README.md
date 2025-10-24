# 🦺 PPE Detection Dashboard

A comprehensive Streamlit dashboard for **Automated Detection of Personal Protective Equipment for Construction Safety** using YOLOv8.

## 📋 Project Overview

This dashboard provides a complete monitoring solution for Personal Protective Equipment (PPE) compliance on construction sites. It leverages computer vision and deep learning to automatically detect:

- 🪖 **Helmets** (Hard Hats)
- 🦺 **Safety Vests** (Reflective Vests)
- 🧤 **Gloves** (Work Gloves)
- 👢 **Safety Boots** (Steel-toe Boots)

## 🚀 Features

### 🏗️ Site Overview
- Real-time safety metrics and compliance rates
- Live worker monitoring with violation alerts
- Site-wise performance comparison
- Historical compliance trends

### 🎯 Model Performance
- YOLOv8 detection metrics (mAP, Precision, Recall)
- Confusion matrix visualization
- Training history and performance curves
- Speed vs accuracy analysis

### 📊 Dataset Analysis
- Comprehensive dataset statistics (43,054 images from 9 datasets)
- Class distribution analysis
- Data quality metrics and preprocessing pipeline
- Challenge identification and solutions

### 🔍 Live Detection
- Real-time image upload and analysis
- Configurable detection parameters
- Worker compliance assessment
- Live camera feed integration (placeholder)

### 📈 Analytics & Reports
- Automated safety report generation
- Predictive analytics for violation patterns
- AI-generated safety recommendations
- Multiple export formats (PDF, Excel, CSV)

### ⚙️ System Settings
- Model configuration and parameters
- Alert and notification settings
- Camera and monitoring configuration
- Data storage and retention policies

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd PEE-Detection/PEE-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard:**
```bash
streamlit run app.py
```

4. **Open your browser and navigate to:**
```
http://localhost:8501
```

## 📊 Dataset Information

The system utilizes a comprehensive dataset compiled from 9 different sources:

| Dataset | Images | Size | Classes | Format |
|---------|--------|------|---------|--------|
| SH17 PPE Dataset | 8,095 | 13.0 GB | 17 | TXT |
| HuggingFace PPE | 11,978 | 2.1 GB | 4 | COCO |
| Harvard Dataverse PPE | 7,063 | 262 MB | 8 | XML |
| Deteksi APD | 3,958 | 112 MB | 4 | TXT |
| Hard Hat Detection | 5,000 | 1.3 GB | 1 | XML |
| Mendeley PPE | 2,286 | 229 MB | 4 | TXT |
| CHVG Dataset | 1,699 | 429 MB | 8 | TXT |
| SoDaConstruction | 1,559 | 163 MB | 4 | TXT |
| PPE Kit Detection | 1,416 | 174 MB | 6 | TXT |

**Total: 43,054 images, 19.9 GB**

## 🎯 Model Architecture

- **Base Model:** YOLOv8 (You Only Look Once)
- **Input Resolution:** 640x640 pixels
- **Classes:** 4 unified PPE categories
- **Performance:** 89.1% mAP@0.5, 45ms inference time
- **Framework:** PyTorch with Ultralytics

## 📈 Key Metrics

- **Overall Compliance Rate:** 87.3%
- **Detection Accuracy:** 94.2% (Helmets), 91.8% (Vests), 78.5% (Gloves), 82.1% (Boots)
- **Processing Speed:** 45ms per image
- **Real-time Capability:** 30 FPS on GPU

## 🔧 Technical Requirements

- **Python:** 3.8+
- **RAM:** 8GB minimum (16GB recommended)
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)
- **Storage:** 2GB for application, additional space for data storage

## 📚 Usage

1. **Navigate through the sidebar** to access different sections
2. **Upload images** in the Live Detection page for instant analysis
3. **Configure settings** in the System Settings page
4. **Generate reports** from the Analytics & Reports section
5. **Monitor real-time compliance** on the Site Overview page

## 🏗️ Project Structure

```
PEE-dashboard/
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── README.md                                # This documentation
├── DPL302m_ADPPECS_Project_Report.pdf      # Technical project report
└── assets/                                  # (Future: images, models, etc.)
```

## 🔮 Future Enhancements

- **Real-time video stream processing**
- **Multi-camera support**
- **Integration with existing security systems**
- **Mobile app companion**
- **Advanced analytics with ML predictions**
- **IoT sensor integration**

## 👥 Team Members

- **Le Nguyen Gia Hung** (SE194127)
- **Huynh Quoc Viet** (SE194225)
- **Vo Tan Phat** (SE194484)
- **Ngo Hoai Nam** (SE194190)

**Lecturer:** Mr. Ho Le Minh Toan

## 📄 License

This project is developed as part of the DPL302m course (2025) at FPT University.

## 🆘 Support

For technical support or questions about the project, please contact the development team or refer to the technical report included in this repository.

---

**⚠️ Safety Note:** This system is designed to assist with safety monitoring but should not replace human safety supervision and adherence to construction safety protocols.