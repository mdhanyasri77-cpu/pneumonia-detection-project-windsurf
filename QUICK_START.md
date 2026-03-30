# 🚀 Quick Start Guide

Get your Pneumonia Detection System running in minutes!

## ⚡ One-Command Setup

```bash
# Clone and setup
git clone <repository-url>
cd pneumonia_detection_system

# Install dependencies
pip install -r requirements.txt

# Train model and run app
python src/train.py && streamlit run app.py
```

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Kaggle API configured (for dataset download)
- [ ] GPU recommended for training (optional)

## 🔧 Kaggle API Setup (2 minutes)

1. Go to [Kaggle.com](https://kaggle.com) → Account → Create New API Token
2. Download `kaggle.json`
3. Move to: `C:\Users\<username>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux)

## 🏥 System Overview

```
Upload X-ray → AI Analysis → Medical Report
    ↓              ↓            ↓
  Image        Prediction    PDF Download
```

## 📊 Expected Performance

- **Training Time**: 30-60 minutes
- **Accuracy**: 92-95%
- **Inference**: <1 second per image

## 🎯 Usage Steps

1. **Train Model**: `python src/train.py`
2. **Launch App**: `streamlit run app.py`
3. **Upload Image**: Drag & drop chest X-ray
4. **Get Results**: View prediction and confidence
5. **Generate Report**: Download medical PDF

## 🆘 Troubleshooting

**Dataset Download Issues?**
- Check Kaggle API credentials
- Ensure internet connection
- Try manual download

**Model Training Slow?**
- Use GPU: `pip install tensorflow-gpu`
- Reduce epochs in `config.py`

**App Not Loading?**
- Check Streamlit version: `pip install --upgrade streamlit`
- Clear cache: `streamlit cache clear`

## 📞 Need Help?

- Check [README.md](README.md) for detailed docs
- Create an Issue for bugs
- Review training logs for errors

---
**Ready in 5 minutes! 🎉**
