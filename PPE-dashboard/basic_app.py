import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PPE Detection - Basic Dashboard",
    page_icon="🦺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .detection-result {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .violation-alert {
        background-color: #ffe8e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🦺 PPE Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Upload an image to detect Personal Protective Equipment**")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.1, 1.0, 0.5, 0.05,
            help="Minimum confidence for detections"
        )
        
        model_size = st.selectbox(
            "Model Version",
            ["YOLOv8n (Fast)", "YOLOv8s (Balanced)", "YOLOv8m (Accurate)"],
            index=0
        )
        
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        
        st.header("📊 Quick Stats")
        st.metric("Model Accuracy", "89.1%")
        st.metric("Processing Speed", "45ms")
        st.metric("Total Classes", "4")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📈 Model Performance", "📊 Prediction Analytics"])
    
    with tab1:
        show_detection_tab(confidence_threshold, show_confidence, show_boxes)
    
    with tab2:
        show_model_performance_tab()
    
    with tab3:
        show_prediction_analytics_tab()

def show_detection_tab(confidence_threshold, show_confidence, show_boxes):
    st.header("🔍 PPE Detection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a construction site image for PPE detection"
    )
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate detection process
                status_text.text("Loading model...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Processing image...")
                progress_bar.progress(50)
                time.sleep(1)
                
                status_text.text("Running inference...")
                progress_bar.progress(80)
                time.sleep(0.5)
                
                status_text.text("Generating results...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Mock detection results
                detection_results = generate_mock_detection()
                
                # Display annotated image (simulated)
                st.image(image, caption="Detection Results (Simulated)", use_column_width=True)
                
                # Detection summary
                display_detection_summary(detection_results, confidence_threshold)
    
    else:
        st.info("👆 Please upload an image to start detection")
        
        # Sample images
        st.subheader("📋 Sample Images")
        st.write("Try these sample construction site scenarios:")
        
        sample_cols = st.columns(3)
        with sample_cols[0]:
            st.write("**Complete PPE**")
            st.write("Worker with helmet, vest, gloves, and boots")
        
        with sample_cols[1]:
            st.write("**Missing Helmet**")
            st.write("Safety violation scenario")
        
        with sample_cols[2]:
            st.write("**Multiple Workers**")
            st.write("Complex scene detection")

def generate_mock_detection():
    """Generate realistic mock detection results"""
    num_workers = np.random.randint(1, 4)
    
    results = {
        'workers': num_workers,
        'detections': [],
        'total_objects': 0,
        'processing_time': np.random.uniform(35, 55),
        'violations': []
    }
    
    for i in range(num_workers):
        worker_detections = {}
        worker_id = f"Worker_{i+1}"
        
        # Helmet detection
        helmet_conf = np.random.uniform(0.6, 0.98)
        has_helmet = helmet_conf > 0.5
        worker_detections['helmet'] = {'detected': has_helmet, 'confidence': helmet_conf}
        
        # Vest detection  
        vest_conf = np.random.uniform(0.5, 0.96)
        has_vest = vest_conf > 0.4
        worker_detections['vest'] = {'detected': has_vest, 'confidence': vest_conf}
        
        # Gloves detection (less common)
        gloves_conf = np.random.uniform(0.3, 0.89)
        has_gloves = gloves_conf > 0.6
        worker_detections['gloves'] = {'detected': has_gloves, 'confidence': gloves_conf}
        
        # Boots detection
        boots_conf = np.random.uniform(0.4, 0.94)
        has_boots = boots_conf > 0.5
        worker_detections['boots'] = {'detected': has_boots, 'confidence': boots_conf}
        
        results['detections'].append({
            'worker_id': worker_id,
            'equipment': worker_detections
        })
        
        # Count violations
        missing_items = []
        if not has_helmet:
            missing_items.append('Helmet')
        if not has_vest:
            missing_items.append('Vest')
        if not has_gloves and np.random.random() > 0.3:  # Not always required
            missing_items.append('Gloves')
        if not has_boots:
            missing_items.append('Boots')
        
        if missing_items:
            results['violations'].append({
                'worker_id': worker_id,
                'missing_items': missing_items
            })
    
    # Count total detected objects
    total_detected = 0
    for detection in results['detections']:
        for item, data in detection['equipment'].items():
            if data['detected']:
                total_detected += 1
    
    results['total_objects'] = total_detected
    
    return results

def display_detection_summary(results, confidence_threshold):
    """Display detection results in a structured way"""
    
    # Processing info
    st.success(f"✅ Detection completed in {results['processing_time']:.1f}ms")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Workers Detected", results['workers'])
    
    with col2:
        st.metric("PPE Items Found", results['total_objects'])
    
    with col3:
        violations_count = len(results['violations'])
        st.metric("Violations", violations_count, delta_color="inverse")
    
    with col4:
        compliance_rate = ((results['workers'] - violations_count) / results['workers'] * 100) if results['workers'] > 0 else 0
        st.metric("Compliance", f"{compliance_rate:.0f}%")
    
    # Detailed results
    st.subheader("🔍 Detailed Detection Results")
    
    for detection in results['detections']:
        worker_id = detection['worker_id']
        equipment = detection['equipment']
        
        # Check if worker is compliant
        is_compliant = worker_id not in [v['worker_id'] for v in results['violations']]
        
        if is_compliant:
            st.markdown(f'<div class="detection-result"><strong>{worker_id}</strong> - ✅ COMPLIANT</div>', unsafe_allow_html=True)
        else:
            violation = next(v for v in results['violations'] if v['worker_id'] == worker_id)
            missing_str = ", ".join(violation['missing_items'])
            st.markdown(f'<div class="violation-alert"><strong>{worker_id}</strong> - ⚠️ VIOLATION: Missing {missing_str}</div>', unsafe_allow_html=True)
        
        # Equipment details
        cols = st.columns(4)
        equipment_items = ['helmet', 'vest', 'gloves', 'boots']
        equipment_icons = ['🪖', '🦺', '🧤', '👢']
        
        for i, (item, icon) in enumerate(zip(equipment_items, equipment_icons)):
            with cols[i]:
                detected = equipment[item]['detected']
                confidence = equipment[item]['confidence']
                
                if detected and confidence >= confidence_threshold:
                    st.write(f"{icon} **{item.title()}**")
                    st.write(f"✅ Detected ({confidence:.2f})")
                else:
                    st.write(f"{icon} **{item.title()}**")
                    if detected:
                        st.write(f"⚠️ Low Conf. ({confidence:.2f})")
                    else:
                        st.write("❌ Not Detected")
        
        st.write("---")
    
    # Violation summary
    if results['violations']:
        st.subheader("🚨 Safety Violations Summary")
        
        violation_data = []
        for violation in results['violations']:
            violation_data.append({
                'Worker': violation['worker_id'],
                'Missing PPE': ", ".join(violation['missing_items']),
                'Severity': 'High' if 'Helmet' in violation['missing_items'] else 'Medium',
                'Action Required': 'Immediate' if 'Helmet' in violation['missing_items'] else 'Standard'
            })
        
        violation_df = pd.DataFrame(violation_data)
        st.dataframe(violation_df, use_container_width=True)

def show_model_performance_tab():
    st.header("📈 Model Performance")
    
    # Model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Version", "YOLOv8n")
        st.metric("Model Size", "6.2 MB")
    
    with col2:
        st.metric("Training Images", "43,054")
        st.metric("Classes", "4")
    
    with col3:
        st.metric("mAP@0.5", "89.1%")
        st.metric("Inference Speed", "45ms")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Per-Class Performance")
        
        performance_data = pd.DataFrame({
            'PPE Class': ['Helmet', 'Vest', 'Gloves', 'Boots'],
            'Precision': [0.915, 0.859, 0.798, 0.924],
            'Recall': [0.931, 0.916, 0.873, 0.916],
            'mAP@0.5': [0.923, 0.887, 0.834, 0.920]
        })
        
        fig = px.bar(performance_data, x='PPE Class', y=['Precision', 'Recall', 'mAP@0.5'],
                    title="Model Performance by PPE Class", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(performance_data, use_container_width=True)
    
    with col2:
        st.subheader("Speed vs Accuracy")
        
        model_comparison = pd.DataFrame({
            'Model': ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l'],
            'Speed (ms)': [45, 68, 95, 142],
            'mAP@0.5': [0.891, 0.913, 0.936, 0.948],
            'Size (MB)': [6.2, 21.5, 49.7, 83.7]
        })
        
        fig = px.scatter(model_comparison, x='Speed (ms)', y='mAP@0.5', 
                        size='Size (MB)', hover_name='Model',
                        title="Model Trade-offs")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(model_comparison, use_container_width=True)
    
    # Training metrics
    st.subheader("Training Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate training curve data
        epochs = list(range(1, 101))
        train_loss = [0.8 - 0.6 * (1 - np.exp(-i/20)) + np.random.normal(0, 0.02) for i in epochs]
        val_loss = [0.85 - 0.55 * (1 - np.exp(-i/25)) + np.random.normal(0, 0.025) for i in epochs]
        
        training_loss_df = pd.DataFrame({
            'Epoch': epochs * 2,
            'Loss': train_loss + val_loss,
            'Dataset': ['Training'] * 100 + ['Validation'] * 100
        })
        
        fig = px.line(training_loss_df, x='Epoch', y='Loss', color='Dataset',
                     title="Training Loss")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # mAP progression
        train_map = [0.3 + 0.6 * (1 - np.exp(-i/15)) + np.random.normal(0, 0.01) for i in epochs]
        val_map = [0.25 + 0.65 * (1 - np.exp(-i/18)) + np.random.normal(0, 0.015) for i in epochs]
        
        training_map_df = pd.DataFrame({
            'Epoch': epochs * 2,
            'mAP': train_map + val_map,
            'Dataset': ['Training'] * 100 + ['Validation'] * 100
        })
        
        fig = px.line(training_map_df, x='Epoch', y='mAP', color='Dataset',
                     title="Training mAP")
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Model Confusion Matrix")
    
    classes = ['Helmet', 'Vest', 'Gloves', 'Boots']
    confusion_matrix = np.array([
        [2156, 23, 15, 8],    # Helmet
        [31, 1987, 42, 18],   # Vest
        [45, 67, 1534, 29],   # Gloves
        [12, 24, 35, 1876]    # Boots
    ])
    
    fig = px.imshow(confusion_matrix, 
                   text_auto=True,
                   aspect="auto",
                   title="Confusion Matrix",
                   labels=dict(x="Predicted", y="Actual"),
                   x=classes, y=classes,
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_analytics_tab():
    st.header("📊 Prediction Analytics")
    
    # Generate some sample prediction history
    st.subheader("Recent Predictions")
    
    # Create sample prediction data
    prediction_history = []
    for i in range(20):
        timestamp = datetime.now().replace(
            hour=np.random.randint(6, 18),
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
        workers = np.random.randint(1, 5)
        violations = np.random.randint(0, workers + 1)
        compliance = ((workers - violations) / workers * 100) if workers > 0 else 0
        
        prediction_history.append({
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Image_ID': f'IMG_{1000+i:04d}',
            'Workers_Detected': workers,
            'Violations': violations,
            'Compliance_Rate': f"{compliance:.0f}%",
            'Processing_Time': f"{np.random.uniform(35, 65):.1f}ms",
            'Model_Confidence': f"{np.random.uniform(0.7, 0.98):.3f}"
        })
    
    prediction_df = pd.DataFrame(prediction_history)
    st.dataframe(prediction_df, use_container_width=True)
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Compliance Distribution")
        
        compliance_values = [float(x.strip('%')) for x in prediction_df['Compliance_Rate']]
        compliance_bins = pd.cut(compliance_values, 
                               bins=[0, 50, 75, 90, 100], 
                               labels=['Poor (<50%)', 'Fair (50-75%)', 'Good (75-90%)', 'Excellent (90-100%)'])
        
        compliance_dist = pd.DataFrame({
            'Compliance_Category': compliance_bins.value_counts().index,
            'Count': compliance_bins.value_counts().values
        })
        
        fig = px.pie(compliance_dist, values='Count', names='Compliance_Category',
                    title="Compliance Rate Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Processing Performance")
        
        processing_times = [float(x.strip('ms')) for x in prediction_df['Processing_Time']]
        
        fig = px.histogram(processing_times, nbins=10,
                          title="Processing Time Distribution",
                          labels={'value': 'Processing Time (ms)', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Detection statistics
    st.subheader("Detection Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_workers = prediction_df['Workers_Detected'].mean()
        st.metric("Avg Workers/Image", f"{avg_workers:.1f}")
    
    with col2:
        total_violations = prediction_df['Violations'].sum()
        st.metric("Total Violations", total_violations)
    
    with col3:
        avg_processing = np.mean(processing_times)
        st.metric("Avg Processing Time", f"{avg_processing:.1f}ms")
    
    with col4:
        avg_confidence = np.mean([float(x) for x in prediction_df['Model_Confidence']])
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    # PPE Detection Trends
    st.subheader("PPE Detection Trends")
    
    # Generate sample PPE detection data
    ppe_trend_data = []
    for i in range(10):
        ppe_trend_data.append({
            'Image': f'IMG_{i+1:02d}',
            'Helmet': np.random.uniform(0.8, 1.0),
            'Vest': np.random.uniform(0.7, 0.95),
            'Gloves': np.random.uniform(0.5, 0.85),
            'Boots': np.random.uniform(0.6, 0.9)
        })
    
    ppe_df = pd.DataFrame(ppe_trend_data)
    
    fig = px.line(ppe_df.melt(id_vars=['Image'], var_name='PPE_Type', value_name='Detection_Rate'),
                 x='Image', y='Detection_Rate', color='PPE_Type',
                 title="PPE Detection Rates Across Recent Images")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()