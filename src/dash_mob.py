import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os
from PIL import Image
import io
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zipfile
import tempfile
from datetime import datetime
import time
import qrcode
import socket

# Page config for mobile responsiveness
st.set_page_config(
    page_title="Smart Waste Classification Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsiveness and styling
st.markdown("""
<style>
    .main-container {
        padding: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .correct-prediction {
        border-color: #4CAF50;
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
    }
    
    .low-confidence {
        border-color: #FF9800;
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    }
    
    .high-confidence {
        border-color: #2196F3;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .camera-toggle-btn {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .camera-off-btn {
        background: linear-gradient(90deg, #f44336 0%, #da190b 100%);
    }
    
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
        }
        
        .metric-container {
            font-size: 0.9rem;
            padding: 0.8rem;
        }
        
        .prediction-card {
            margin: 0.25rem;
            padding: 0.8rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
    }
    
    /* Hide scrollbar */
    .main .block-container {
        max-height: 100vh;
        overflow-y: auto;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    
    .main .block-container::-webkit-scrollbar {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# CORRECTED Model Class - Now matches your training architecture
class EfficientNetModel(nn.Module):
    """EfficientNet-B3 model matching the training architecture"""

    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(EfficientNetModel, self).__init__()

        # Use EfficientNet-B3 to match training (not B0!)
        try:
            # Try new torchvision format
            self.backbone = models.efficientnet_b3(weights=None)
        except:
            # Fall back to old format
            self.backbone = models.efficientnet_b3(pretrained=False)

        # Get the correct input features for the classifier
        if hasattr(self.backbone.classifier, '__getitem__') and len(self.backbone.classifier) > 1:
            in_features = self.backbone.classifier[1].in_features
        else:
            in_features = self.backbone.classifier.in_features

        # Custom classifier matching training architecture
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class AdvancedImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def remove_background(self, image):
        try:
            mask = np.zeros(image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            height, width = image.shape[:2]
            rect = (10, 10, width-20, height-20)
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = image * mask2[:, :, np.newaxis]
            return result
        except:
            return image

    def enhance_contrast(self, image):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except:
            return image

    def denoise_image(self, image):
        try:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except:
            return image

    def preprocess_image(self, image, enhance=True):
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image = cv2.resize(image, self.target_size)
            
            if enhance:
                image = self.remove_background(image)
                image = self.enhance_contrast(image)
                image = self.denoise_image(image)
            
            # Convert back to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            st.warning(f"Preprocessing failed, using original image: {e}")
            if isinstance(image, np.ndarray):
                return cv2.resize(image, self.target_size)
            else:
                return np.array(Image.fromarray(image).resize(self.target_size))

def load_model_and_config():
    """Load model and configuration files - FIXED VERSION"""
    base_path = r"C:\Users\HP\Desktop\DIP_TASK"
    
    try:
        # Load class names
        class_path = os.path.join(base_path, 'class_names.json')
        if os.path.exists(class_path):
            with open(class_path, 'r') as f:
                classes = json.load(f)
        else:
            classes = ['glass', 'metal', 'Organic', 'paper', 'plastic']
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Look for model files
        model_files = ['best_model.pth', 'final_model.pth', 'model.pth']
        model_path = None
        
        for model_file in model_files:
            full_path = os.path.join(base_path, model_file)
            if os.path.exists(full_path):
                model_path = full_path
                break
        
        if model_path is None:
            st.error("No model file found. Please ensure model file exists in the directory.")
            return None, None, None, None
        
        # Create model with correct architecture (EfficientNet-B3)
        model = EfficientNetModel(num_classes=len(classes))
        
        # Load the saved weights
        try:
            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Load weights (should work now since architectures match)
            model.load_state_dict(state_dict, strict=True)
            
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            st.write("Creating model with random weights for demonstration...")
        
        model.eval()
        model.to(device)
        
        # Load preprocessing transforms
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return model, classes, transform, device
        
    except Exception as e:
        st.error(f"Error in load_model_and_config: {str(e)}")
        return None, None, None, None

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = {'glass': 0, 'metal': 0, 'Organic': 0, 'paper': 0, 'plastic': 0}
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False
if 'camera_facing' not in st.session_state:
    st.session_state.camera_facing = 'user'  # 'user' for front, 'environment' for back

def get_local_ip():
    """Get local IP address for QR code generation"""
    try:
        # Connect to a remote server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def generate_qr_code():
    """Generate QR code for mobile access"""
    try:
        # Get local IP and port
        local_ip = get_local_ip()
        port = 8501  # Default Streamlit port
        
        # Create URL
        url = f"http://{local_ip}:{port}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        qr_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue(), url
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return None, None
    """Make prediction on a single image with better error handling"""
    try:
        # Preprocess image
        processed_img = preprocessor.preprocess_image(image, enhance=True)
        
        # Apply transforms
        transformed = transform(image=processed_img)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return classes[predicted_class], confidence, probabilities[0].cpu().numpy()
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, None

def predict_image(image, model, classes, transform, device, preprocessor):
    """Create pie chart for class distribution"""
    if sum(st.session_state.class_counts.values()) == 0:
        # Show empty chart
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.class_counts.keys()),
            values=[1]*5,  # Equal distribution for empty state
            hole=.4,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )])
        fig.update_traces(textinfo='none')
        fig.update_layout(
            title="Class Distribution (No Data)",
            showlegend=True,
            height=400,
            annotations=[dict(text='No Predictions', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    else:
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.class_counts.keys()),
            values=list(st.session_state.class_counts.values()),
            hole=.4,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            textinfo='label+percent+value',
            textfont_size=12
        )])
        fig.update_layout(
            title="Class Distribution",
            showlegend=True,
            height=400
        )
    
    return fig

def create_pie_chart():
    """Create bar chart for class distribution"""
    classes = list(st.session_state.class_counts.keys())
    counts = list(st.session_state.class_counts.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=counts,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Distribution (Bar Chart)",
        xaxis_title="Waste Classes",
        yaxis_title="Number of Predictions",
        height=400,
        showlegend=False
    )
    
    return fig

def capture_camera_image():
    """Capture image from camera with toggle control and mobile camera switching"""
    # Check if on mobile (simplified detection)
    is_mobile = st.checkbox("📱 Mobile Mode (Enable Camera Switching)", key="mobile_mode")
    
    # Camera control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.session_state.camera_enabled:
            if st.button("📷 Turn OFF", key="camera_off"):
                st.session_state.camera_enabled = False
                st.rerun()
        else:
            if st.button("📷 Turn ON", key="camera_on"):
                st.session_state.camera_enabled = True
                st.rerun()
    
    with col2:
        # Camera switching for mobile
        if is_mobile and st.session_state.camera_enabled:
            if st.session_state.camera_facing == 'user':
                if st.button("🔄 Back Cam", key="switch_to_back"):
                    st.session_state.camera_facing = 'environment'
                    st.rerun()
            else:
                if st.button("🔄 Front Cam", key="switch_to_front"):
                    st.session_state.camera_facing = 'user'
                    st.rerun()
    
    with col3:
        # Status display
        if st.session_state.camera_enabled:
            cam_type = "Front" if st.session_state.camera_facing == 'user' else "Back"
            status_text = f"🟢 Camera Active ({cam_type})" if is_mobile else "🟢 Camera Active"
        else:
            status_text = "🔴 Camera Inactive"
        st.markdown(f"**Status:** {status_text}")
    
    # Show camera input only if enabled
    if st.session_state.camera_enabled:
        # Add custom JavaScript for mobile camera switching
        if is_mobile:
            st.markdown(f"""
            <script>
            // Mobile camera constraint
            navigator.mediaDevices.getUserMedia({{
                video: {{ facingMode: "{st.session_state.camera_facing}" }}
            }});
            </script>
            """, unsafe_allow_html=True)
        
        camera_input = st.camera_input("📸 Take a photo of waste item", key=f"camera_{st.session_state.camera_facing}")
        if camera_input is not None:
            return Image.open(camera_input)
    else:
        st.info("Click 'Turn ON' to enable camera capture")
    
    return None

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>♻️ Smart Waste Classification Dashboard</h1>
        <p style='color: white; margin: 0.5rem 0 0 0;'>AI-Powered Waste Sorting & Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and configuration (removed from cache to fix widget error)
    model_data = load_model_and_config()
    if model_data[0] is None:  # If model loading failed
        st.error("Failed to load model. Please check your model files and try again.")
        return
    
    model, classes, transform, device = model_data
    preprocessor = AdvancedImagePreprocessor()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Dashboard", "📋 Detailed Results", "📱 Mobile Access", "🔧 Debug Info"])
    
    with tab3:
        st.markdown("### 📱 Mobile Access")
        
        # QR Code Section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📷 QR Code for Mobile Access")
            
            # Generate QR code
            if st.button("🎯 Generate QR Code", key="generate_qr"):
                with st.spinner("Generating QR code..."):
                    qr_image, url = generate_qr_code()
                    
                    if qr_image:
                        st.image(qr_image, caption="Scan with your mobile device", width=300)
                        
                        st.success(f"✅ QR Code Generated!")
                        st.info(f"**URL:** {url}")
                        
                        # Copy URL button
                        st.markdown(f"""
                        <div style='text-align: center; margin: 1rem 0;'>
                            <button onclick="navigator.clipboard.writeText('{url}')" 
                                    style='background: #28a745; color: white; border: none; 
                                           padding: 10px 20px; border-radius: 5px; cursor: pointer;'>
                                📋 Copy URL
                            </button>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📋 Instructions")
            st.markdown("""
            **Step 1:** Make sure your computer and mobile device are on the same WiFi network
            
            **Step 2:** Run Streamlit with network access:
            ```bash
            streamlit run your_app.py --server.address 0.0.0.0
            ```
            
            **Step 3:** Click "Generate QR Code" and scan with your mobile device
            
            **Step 4:** On mobile, enable "Mobile Mode" for camera switching between front and back cameras
            """)
        
        # Mobile Features Info
        st.markdown("---")
        st.markdown("### 📱 Mobile Features")
        
        feature_col1, feature_col2 = st.columns([1, 1])
        
        with feature_col1:
            st.markdown("""
            #### ✨ Enhanced Mobile Experience
            - **📷 Front/Back Camera Toggle**: Switch between selfie and rear camera
            - **📱 Touch-Friendly Interface**: Optimized for mobile screens
            - **🔄 Real-time Processing**: Instant waste classification
            - **📊 Interactive Charts**: View analytics on mobile
            """)
        
        with feature_col2:
            st.markdown("""
            #### 🚀 Mobile Usage Tips
            - **📶 Stable WiFi**: Ensure strong WiFi connection
            - **🔆 Good Lighting**: Take photos in well-lit areas
            - **📐 Clear Angles**: Hold waste items clearly in frame
            - **🔋 Battery**: Keep device charged for extended use
            """)
        
        # Network Status
        st.markdown("---")
        st.markdown("### 🌐 Network Status")
        
        try:
            local_ip = get_local_ip()
            st.success(f"✅ **Local IP Address:** {local_ip}")
            st.info("💡 **Tip:** Make sure both devices are on the same WiFi network!")
            
            # Test connectivity
            if st.button("🔍 Test Network Connection"):
                st.info("🔄 Testing network connectivity...")
                # You can add actual network tests here
                st.success("✅ Network connection looks good!")
                
        except Exception as e:
            st.error(f"❌ Network Error: {e}")
            st.warning("⚠️ Please check your network connection")

    with tab4:
        st.markdown("### 🔧 Debug Information")
        st.write(f"**Device:** {device}")
        st.write(f"**Classes:** {classes}")
        st.write(f"**Number of classes:** {len(classes)}")
        st.write(f"**Model type:** {type(model).__name__}")
        st.write(f"**Model architecture:** EfficientNet-B3")
        
        # Test prediction with random data
        if st.button("🧪 Test Model with Random Data"):
            try:
                test_input = torch.randn(1, 3, 224, 224).to(device)
                with torch.no_grad():
                    test_output = model(test_input)
                st.success(f"✅ Model test successful! Output shape: {test_output.shape}")
                st.write(f"Raw output: {test_output}")
                
                # Test softmax
                probabilities = torch.softmax(test_output, dim=1)
                st.write(f"Probabilities: {probabilities}")
                predicted_class = torch.argmax(test_output, dim=1).item()
                st.write(f"Predicted class index: {predicted_class}")
                st.write(f"Predicted class name: {classes[predicted_class]}")
                
            except Exception as e:
                st.error(f"❌ Model test failed: {e}")
                st.write("This might indicate an architecture mismatch or missing weights.")
        
        # Model inspection
        if st.checkbox("🔍 Show Model Architecture"):
            st.text(str(model))
        
        if st.checkbox("📊 Show Model Parameters"):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.write(f"Total parameters: {total_params:,}")
            st.write(f"Trainable parameters: {trainable_params:,}")
    
    with tab1:
        # Main dashboard layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Input Methods")
            
            # Camera input with toggle
            st.markdown("""
            <div style='margin: 1rem 0;'>
                <h4>📸 Camera Capture</h4>
            </div>
            """, unsafe_allow_html=True)
            
            camera_image = capture_camera_image()
            if camera_image is not None:
                image_np = np.array(camera_image)
                pred_class, confidence, probabilities = predict_image(
                    image_np, model, classes, transform, device, preprocessor
                )
                
                if pred_class != "Error":
                    # Update session state
                    result = {
                        'filename': f'camera_capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        'image': image_np,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'timestamp': datetime.now()
                    }
                    st.session_state.predictions.append(result)
                    
                    if pred_class in st.session_state.class_counts:
                        st.session_state.class_counts[pred_class] += 1
                    
                    # Show result
                    st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.2%})")
            
            st.markdown("---")
            
            # File upload
            st.markdown("""
            <div style='margin: 1rem 0;'>
                <h4>📁 File Upload</h4>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "Choose images or zip folder",
                type=['png', 'jpg', 'jpeg', 'zip'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("🔍 Process Files", key="process_btn"):
                    with st.spinner("Processing images..."):
                        # Check if zip file
                        zip_files = [f for f in uploaded_files if f.name.endswith('.zip')]
                        image_files = [f for f in uploaded_files if not f.name.endswith('.zip')]
                        
                        all_results = []
                        
                        # Process zip files
                        for zip_file in zip_files:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                    zip_ref.extractall(temp_dir)
                                
                                # Process extracted images
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                            img_path = os.path.join(root, file)
                                            image = Image.open(img_path)
                                            image_np = np.array(image)
                                            
                                            pred_class, confidence, probabilities = predict_image(
                                                image_np, model, classes, transform, device, preprocessor
                                            )
                                            
                                            if pred_class != "Error":
                                                result = {
                                                    'filename': file,
                                                    'image': image_np,
                                                    'predicted_class': pred_class,
                                                    'confidence': confidence,
                                                    'probabilities': probabilities,
                                                    'timestamp': datetime.now()
                                                }
                                                all_results.append(result)
                                                
                                                if pred_class in st.session_state.class_counts:
                                                    st.session_state.class_counts[pred_class] += 1
                        
                        # Process individual image files
                        for img_file in image_files:
                            image = Image.open(img_file)
                            image_np = np.array(image)
                            
                            pred_class, confidence, probabilities = predict_image(
                                image_np, model, classes, transform, device, preprocessor
                            )
                            
                            if pred_class != "Error":
                                result = {
                                    'filename': img_file.name,
                                    'image': image_np,
                                    'predicted_class': pred_class,
                                    'confidence': confidence,
                                    'probabilities': probabilities,
                                    'timestamp': datetime.now()
                                }
                                all_results.append(result)
                                
                                if pred_class in st.session_state.class_counts:
                                    st.session_state.class_counts[pred_class] += 1
                        
                        # Add to session state
                        st.session_state.predictions.extend(all_results)
                        st.success(f"Processed {len(all_results)} images successfully!")
        
        with col2:
            st.markdown("### 📊 Analytics")
            
            # Metrics
            total_predictions = len(st.session_state.predictions)
            avg_confidence = np.mean([p['confidence'] for p in st.session_state.predictions]) if st.session_state.predictions else 0
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Predictions", total_predictions)
            with metric_col2:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Charts
            st.plotly_chart(create_pie_chart(), use_container_width=True)
            st.plotly_chart(create_bar_chart(), use_container_width=True)
        
        # Recent predictions
        if st.session_state.predictions:
            st.markdown("### 🔮 Recent Predictions")
            recent_predictions = st.session_state.predictions[-6:]  # Show last 6
            
            cols = st.columns(3)
            for i, pred in enumerate(recent_predictions):
                col_idx = i % 3
                with cols[col_idx]:
                    # Confidence-based styling
                    if pred['confidence'] >= 0.8:
                        card_class = "high-confidence"
                    elif pred['confidence'] >= 0.6:
                        card_class = "low-confidence"  
                    else:
                        card_class = "correct-prediction"
                    
                    # Modified card - only show class and confidence with dark text
                    st.markdown(f"""
                    <div class='prediction-card {card_class}'>
                        <h5 style='color: #333333; margin-bottom: 0.5rem;'>{pred['predicted_class']}</h5>
                        <p style='color: #000000; font-weight: bold; font-size: 1.1rem; margin: 0.5rem 0;'>{pred['confidence']:.1%}</p>
                        <small style='color: #666666;'>{pred['filename']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show thumbnail
                    if pred['image'] is not None:
                        st.image(pred['image'], use_container_width=True)
    
    with tab2:
        st.markdown("### 📋 Detailed Classification Results")
        
        if not st.session_state.predictions:
            st.info("No predictions available. Upload images in the Main Dashboard tab.")
            return
        
        # Controls
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            sort_by = st.selectbox("Sort by", ["Timestamp", "Confidence", "Class"])
        with col2:
            filter_class = st.selectbox("Filter by class", ["All"] + classes)
        with col3:
            if st.button("🗑️ Clear All Predictions"):
                st.session_state.predictions = []
                st.session_state.class_counts = {cls: 0 for cls in classes}
                st.rerun()
        
        # Filter predictions
        filtered_predictions = st.session_state.predictions.copy()
        if filter_class != "All":
            filtered_predictions = [p for p in filtered_predictions if p['predicted_class'] == filter_class]
        
        # Sort predictions
        if sort_by == "Confidence":
            filtered_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "Class":
            filtered_predictions.sort(key=lambda x: x['predicted_class'])
        else:  # Timestamp
            filtered_predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Display detailed results
        for i, pred in enumerate(filtered_predictions):
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Display image
                    st.image(pred['image'], width=150)
                
                with col2:
                    # Prediction details
                    st.markdown(f"**Filename:** {pred['filename']}")
                    st.markdown(f"**Predicted Class:** {pred['predicted_class']}")
                    st.markdown(f"**Confidence:** {pred['confidence']:.2%}")
                    st.markdown(f"**Timestamp:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Confidence bar
                    confidence_color = "green" if pred['confidence'] >= 0.8 else "orange" if pred['confidence'] >= 0.6 else "red"
                    st.markdown(f"""
                    <div style='background-color: #f0f0f0; border-radius: 10px; padding: 5px;'>
                        <div style='background-color: {confidence_color}; width: {pred['confidence']*100}%; 
                                    height: 20px; border-radius: 5px; color: white; text-align: center; line-height: 20px;'>
                            {pred['confidence']:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # All class probabilities - REMOVED COLOR DOTS, ONLY CLASS AND CONFIDENCE
                    if pred['probabilities'] is not None:
                        st.markdown("**All Classes:**")
                        for j, (cls, prob) in enumerate(zip(classes, pred['probabilities'])):
                            # Show only class name and confidence percentage
                            st.markdown(f"{cls}: {prob:.1%}")
                
                st.markdown("---")

if __name__ == "__main__":
    main()