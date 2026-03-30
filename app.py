import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import PneumoniaDetector
from report_generator import ReportGenerator
from config import Config

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        detector = PneumoniaDetector()
        model_path = os.path.join('models', 'pneumonia_detector.h5')
        if os.path.exists(model_path):
            detector.load_model(model_path)
            return detector
        else:
            st.error("❌ Model not found. Please train the model first by running: python src/train.py")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Ensure image has 3 dimensions (height, width, channels)
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif len(img_array.shape) == 4:  # RGBA image
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Resize to 224x224
    img_resized = tf.image.resize(img_array, (224, 224))
    
    # Normalize to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_array

def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_prediction_pie(prediction, confidence):
    """Create a pie chart showing prediction breakdown"""
    if prediction == "Pneumonia":
        values = [confidence * 100, (1 - confidence) * 100]
        labels = ["Pneumonia", "Normal"]
        colors = ["#ff6b6b", "#51cf66"]
    else:
        values = [(1 - confidence) * 100, confidence * 100]
        labels = ["Normal", "Pneumonia"]
        colors = ["#51cf66", "#ff6b6b"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        font=dict(size=14),
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = None
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(Config().DISCLAIMER, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model
    detector = load_model()
    if detector is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 System Options")
    
    # Model information
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info("""
    **Model Architecture**: MobileNetV2 (Transfer Learning)
    
    **Input Size**: 224×224 pixels
    
    **Classes**: Normal, Pneumonia
    
    **Training Dataset**: Chest X-Ray Images
    """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Predict", "📋 Medical Guidance", "📄 Reports"])
    
    with tab1:
        st.header("📸 Upload Chest X-Ray Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Image information
                st.info(f"""
                **Image Details:**
                - Size: {image.size}
                - Format: {image.format}
                - Mode: {image.mode}
                """)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                # Preprocess and predict
                with st.spinner("Analyzing image..."):
                    img_batch, original_img = preprocess_image(image)
                    result = detector.predict_single_image(img_batch)
                
                # Display prediction
                if result['prediction'] == 'Pneumonia':
                    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                    st.error(f"## 🚨 {result['prediction']} Detected")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"## ✅ {result['prediction']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence score
                st.metric("Confidence Score", f"{result['confidence']:.2%}")
                
                # Visualizations
                st.subheader("📊 Confidence Analysis")
                
                # Gauge chart
                fig_gauge = create_confidence_gauge(result['confidence'])
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Pie chart
                fig_pie = create_prediction_pie(result['prediction'], result['confidence'])
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Raw probability
                st.info(f"Raw Probability: {result['raw_probability']:.4f}")
        
        else:
            st.info("👆 Please upload a chest X-ray image to begin analysis.")
    
    with tab2:
        st.header("📋 Medical Guidance & Recommendations")
        
        # Patient information form
        with st.form("patient_form"):
            st.subheader("👤 Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                patient_name = st.text_input("Patient Name")
                patient_age = st.number_input("Age", min_value=0, max_value=150, value=30)
            
            with col2:
                patient_gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
                patient_notes = st.text_area("Additional Notes")
            
            submitted = st.form_submit_button("Generate Recommendations")
            
            if submitted:
                if patient_name and patient_gender != "Select":
                    st.success("✅ Patient information recorded!")
                    
                    # Display general precautions
                    st.subheader("🛡️ General Pneumonia Precautions")
                    for i, precaution in enumerate(Config().PNEUMONIA_PRECAUTIONS, 1):
                        st.write(f"{i}. {precaution}")
                    
                    # Display diet recommendations
                    st.subheader("🥗 Recommended Diet Plan")
                    for i, diet_item in enumerate(Config().RECOMMENDED_DIETS, 1):
                        st.write(f"{i}. {diet_item}")
                    
                    # Store patient info for report generation
                    st.session_state.patient_info = {
                        'name': patient_name,
                        'age': patient_age,
                        'gender': patient_gender,
                        'notes': patient_notes
                    }
                else:
                    st.error("❌ Please fill in all required fields.")
    
    with tab3:
        st.header("📄 Report Generation")
        
        if 'patient_info' not in st.session_state:
            st.warning("⚠️ Please complete patient information in the Medical Guidance tab first.")
        else:
            st.info("📋 Patient information available for report generation.")
            
            if st.button("📄 Generate Medical Report", type="primary"):
                with st.spinner("Generating report..."):
                    # Create a sample prediction for demo
                    sample_result = {
                        'prediction': 'Normal',
                        'confidence': 0.85,
                        'raw_probability': 0.15
                    }
                    
                    # Generate report
                    report_gen = ReportGenerator()
                    report_path = report_gen.generate_medical_report(
                        st.session_state.patient_info,
                        sample_result
                    )
                    
                    st.success(f"✅ Report generated successfully!")
                    st.info(f"Report saved to: {report_path}")
                    
                    # Provide download link
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Report",
                            data=file.read(),
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Pneumonia Detection System | AI-Powered Medical Imaging Analysis</p>
        <p>Developed with ❤️ using TensorFlow, Streamlit, and Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
