import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="PPE Detection Dashboard",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .safety-alert {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .compliance-good {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .ppe-card {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🦺 PPE Detection & Construction Safety Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Automated Detection of Personal Protective Equipment for Construction Safety**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Navigation")
        page = st.selectbox("Select Page", [
            "🏗️ Site Overview", 
            "🎯 Model Performance", 
            "📊 Dataset Analysis", 
            "🔍 Live Detection", 
            "📈 Analytics & Reports",
            "⚙️ System Settings"
        ])
        
        st.header("🔍 Filters")
        site_filter = st.multiselect(
            "Construction Sites",
            ["Site A - Building Complex", "Site B - Bridge", "Site C - Highway", "Site D - Industrial"],
            default=["Site A - Building Complex", "Site B - Bridge"]
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            key="date_filter"
        )
        
        st.header("🦺 PPE Categories")
        ppe_filter = st.multiselect(
            "PPE Items",
            ["Helmet", "Vest", "Gloves", "Boots"],
            default=["Helmet", "Vest", "Gloves", "Boots"]
        )
    
    # Main content based on selected page
    if "Site Overview" in page:
        show_site_overview()
    elif "Model Performance" in page:
        show_model_performance()
    elif "Dataset Analysis" in page:
        show_dataset_analysis()
    elif "Live Detection" in page:
        show_live_detection()
    elif "Analytics & Reports" in page:
        show_analytics_reports()
    elif "System Settings" in page:
        show_system_settings()

def show_site_overview():
    st.header("🏗️ Construction Site Safety Overview")
    
    # Alert Banner
    col1, col2 = st.columns([3, 1])
    with col1:
        compliance_rate = 87.3
        if compliance_rate < 90:
            st.markdown(f'<div class="safety-alert">⚠️ Safety Alert: Overall PPE compliance at {compliance_rate}% - Below target of 90%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="compliance-good">✅ Good Compliance: Overall PPE compliance at {compliance_rate}%</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Active Workers", "247", "12 today")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👷 Workers Monitored",
            value="247",
            delta="12 new today",
            help="Total number of workers currently on site"
        )
    
    with col2:
        st.metric(
            label="🦺 PPE Compliance Rate",
            value="87.3%",
            delta="-2.1%",
            delta_color="inverse",
            help="Percentage of workers with complete PPE compliance"
        )
    
    with col3:
        st.metric(
            label="🚨 Safety Violations",
            value="31",
            delta="5 today",
            delta_color="inverse",
            help="Number of PPE violations detected"
        )
    
    with col4:
        st.metric(
            label="⚡ Detection Speed",
            value="45ms",
            delta="-3ms",
            help="Average processing time per frame"
        )
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PPE Compliance by Category")
        # Sample data for PPE compliance
        ppe_data = pd.DataFrame({
            'PPE_Type': ['Helmet', 'Vest', 'Gloves', 'Boots'],
            'Compliance_Rate': [94.2, 91.8, 78.5, 82.1],
            'Required_Count': [247, 247, 180, 247],  # Not all workers need gloves
            'Detected_Count': [233, 227, 141, 203]
        })
        
        fig = px.bar(ppe_data, x='PPE_Type', y='Compliance_Rate', 
                     title="Current PPE Compliance Rates",
                     color='Compliance_Rate',
                     color_continuous_scale=['red', 'yellow', 'green'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Hourly Violation Trends")
        # Sample hourly data
        hours = list(range(6, 19))  # 6 AM to 6 PM
        violations = [2, 1, 4, 7, 5, 3, 8, 6, 4, 2, 5, 7, 3]
        hourly_data = pd.DataFrame({
            'Hour': [f"{h}:00" for h in hours],
            'Violations': violations[:len(hours)]
        })
        
        fig = px.line(hourly_data, x='Hour', y='Violations',
                     title="Safety Violations by Hour",
                     markers=True)
        fig.update_traces(line_color='#FF6B35', marker_color='#FF6B35')
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Site-wise Compliance")
        site_data = pd.DataFrame({
            'Site': ['Site A - Building', 'Site B - Bridge', 'Site C - Highway', 'Site D - Industrial'],
            'Workers': [89, 67, 54, 37],
            'Compliance': [91.2, 85.1, 88.9, 83.7]
        })
        
        fig = px.scatter(site_data, x='Workers', y='Compliance', 
                        size='Workers', color='Compliance',
                        hover_name='Site',
                        title="Worker Count vs Compliance Rate",
                        color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Weekly Compliance Trend")
        dates = pd.date_range(start='2024-01-15', periods=7)
        weekly_data = pd.DataFrame({
            'Date': dates,
            'Compliance': [85.2, 87.1, 89.3, 86.7, 88.2, 87.8, 87.3],
            'Target': [90] * 7
        })
        
        fig = px.line(weekly_data, x='Date', y=['Compliance', 'Target'],
                     title="7-Day Compliance Trend")
        fig.update_traces(selector=dict(name="Target"), line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Violations Table
    st.subheader("Recent Safety Violations")
    violations_data = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-22 14:30', periods=8, freq='23min'),
        'Worker_ID': [f'W{1500+i}' for i in range(8)],
        'Site': np.random.choice(['Site A', 'Site B', 'Site C'], 8),
        'Missing_PPE': ['Helmet', 'Gloves', 'Vest', 'Boots', 'Helmet, Gloves', 'Vest', 'Boots', 'Gloves'],
        'Severity': np.random.choice(['High', 'Medium', 'Low'], 8),
        'Status': np.random.choice(['Pending', 'Resolved', 'Escalated'], 8)
    })
    
    # Color code the status
    def color_status(val):
        if val == 'Resolved':
            return 'background-color: #d4edda'
        elif val == 'Escalated':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #fff3cd'
    
    styled_df = violations_data.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

def show_model_performance():
    st.header("🎯 YOLOv8 Model Performance Metrics")
    
    # Model Info Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Model**: YOLOv8n\n**Input Size**: 640x640\n**Classes**: 4 (Helmet, Vest, Gloves, Boots)")
    
    with col2:
        st.success("**Training Images**: 43,054\n**Validation Split**: 20%\n**Test Split**: 10%")
    
    with col3:
        st.warning("**Inference Speed**: 45ms\n**Model Size**: 6.2MB\n**GPU Memory**: 2.1GB")
    
    # Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Metrics")
        metrics_data = pd.DataFrame({
            'Metric': ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score'],
            'Overall': [0.891, 0.654, 0.874, 0.908, 0.891],
            'Helmet': [0.923, 0.712, 0.915, 0.931, 0.923],
            'Vest': [0.887, 0.625, 0.859, 0.916, 0.887],
            'Gloves': [0.834, 0.567, 0.798, 0.873, 0.834],
            'Boots': [0.920, 0.712, 0.924, 0.916, 0.920]
        })
        
        fig = px.bar(metrics_data, x='Metric', y=['Helmet', 'Vest', 'Gloves', 'Boots'],
                    title="Per-Class Performance Metrics",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(metrics_data, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        # Sample confusion matrix for 4 classes
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
                       x=classes, y=classes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy per class
        accuracy_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        accuracy_df = pd.DataFrame({
            'Class': classes,
            'Accuracy': accuracy_per_class
        })
        st.dataframe(accuracy_df, use_container_width=True)
    
    # Training History
    st.subheader("Training Progress")
    epochs = list(range(1, 101))
    
    # Simulated training curves
    train_loss = [0.8 - 0.6 * (1 - np.exp(-i/20)) + np.random.normal(0, 0.02) for i in epochs]
    val_loss = [0.85 - 0.55 * (1 - np.exp(-i/25)) + np.random.normal(0, 0.025) for i in epochs]
    train_map = [0.3 + 0.6 * (1 - np.exp(-i/15)) + np.random.normal(0, 0.01) for i in epochs]
    val_map = [0.25 + 0.65 * (1 - np.exp(-i/18)) + np.random.normal(0, 0.015) for i in epochs]
    
    training_data = pd.DataFrame({
        'Epoch': epochs * 4,
        'Value': train_loss + val_loss + train_map + val_map,
        'Metric': ['Train Loss'] * 100 + ['Val Loss'] * 100 + ['Train mAP'] * 100 + ['Val mAP'] * 100,
        'Type': ['Loss', 'Loss', 'mAP', 'mAP'] * 100
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        loss_data = training_data[training_data['Type'] == 'Loss']
        fig = px.line(loss_data, x='Epoch', y='Value', color='Metric',
                     title="Training & Validation Loss")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        map_data = training_data[training_data['Type'] == 'mAP']
        fig = px.line(map_data, x='Epoch', y='Value', color='Metric',
                     title="Training & Validation mAP")
        st.plotly_chart(fig, use_container_width=True)
    
    # Inference Speed Analysis
    st.subheader("Inference Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batch_sizes = [1, 2, 4, 8, 16, 32]
        inference_times = [45, 52, 68, 89, 134, 203]
        
        speed_df = pd.DataFrame({
            'Batch Size': batch_sizes,
            'Inference Time (ms)': inference_times
        })
        
        fig = px.line(speed_df, x='Batch Size', y='Inference Time (ms)',
                     title="Batch Size vs Inference Time",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        resolutions = ['320x320', '416x416', '512x512', '640x640', '832x832']
        accuracy = [0.834, 0.867, 0.883, 0.891, 0.896]
        speed = [28, 35, 41, 45, 67]
        
        res_df = pd.DataFrame({
            'Resolution': resolutions,
            'mAP': accuracy,
            'Speed (ms)': speed
        })
        
        fig = px.scatter(res_df, x='Speed (ms)', y='mAP', 
                        hover_name='Resolution',
                        title="Speed vs Accuracy Trade-off",
                        size=[1]*5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.metric("Best mAP@0.5", "0.891", "at 640x640")
        st.metric("Fastest Inference", "28ms", "at 320x320")
        st.metric("Optimal Balance", "640x640", "45ms, 0.891 mAP")

def show_dataset_analysis():
    st.header("📊 Dataset Analysis & Statistics")
    
    # Dataset Overview
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", "43,054")
    with col2:
        st.metric("Total Datasets", "9")
    with col3:
        st.metric("Dataset Size", "19.9 GB")
    with col4:
        st.metric("Annotation Format", "YOLO TXT")
    
    # Dataset Sources
    st.subheader("Dataset Sources")
    dataset_info = pd.DataFrame({
        'Dataset': ['SH17 PPE Dataset', 'HuggingFace PPE', 'Harvard Dataverse PPE', 
                   'Deteksi APD', 'Hard Hat Detection', 'Mendeley PPE', 
                   'CHVG Dataset', 'SoDaConstruction', 'PPE Kit Detection'],
        'Images': [8095, 11978, 7063, 3958, 5000, 2286, 1699, 1559, 1416],
        'Size (MB)': [13312, 2150, 262, 112, 1310, 229, 429, 163, 174],
        'Classes': [17, 4, 8, 4, 1, 4, 8, 4, 6],
        'Format': ['TXT', 'COCO', 'XML', 'TXT', 'XML', 'TXT', 'TXT', 'TXT', 'TXT']
    })
    
    st.dataframe(dataset_info, use_container_width=True)
    
    # Class Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Unified Class Distribution")
        class_data = pd.DataFrame({
            'PPE_Class': ['Helmet', 'Vest', 'Gloves', 'Boots'],
            'Image_Count': [35210, 24880, 6540, 7150],
            'BBox_Count': [81550, 54120, 8980, 9630]
        })
        
        fig = px.bar(class_data, x='PPE_Class', y=['Image_Count', 'BBox_Count'],
                    title="Class Distribution (Images vs Bounding Boxes)",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Size Distribution")
        fig = px.pie(dataset_info, values='Images', names='Dataset',
                    title="Images per Dataset")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Quality Metrics
    st.subheader("Data Quality Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Image Resolution Distribution")
        resolution_data = pd.DataFrame({
            'Resolution': ['320x320', '416x416', '512x512', '640x640', '832x832', 'Variable'],
            'Count': [5430, 8920, 12340, 15210, 980, 174]
        })
        fig = px.bar(resolution_data, x='Resolution', y='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Lighting Conditions")
        lighting_data = pd.DataFrame({
            'Condition': ['Daylight', 'Overcast', 'Indoor', 'Low Light', 'Night'],
            'Percentage': [65, 20, 10, 3, 2]
        })
        fig = px.pie(lighting_data, values='Percentage', names='Condition')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Scene Complexity")
        complexity_data = pd.DataFrame({
            'Workers_per_Image': ['1', '2-3', '4-6', '7-10', '>10'],
            'Count': [18500, 15200, 6800, 2200, 354]
        })
        fig = px.bar(complexity_data, x='Workers_per_Image', y='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Challenges and Limitations
    st.subheader("Dataset Challenges & Solutions")
    
    challenges = pd.DataFrame({
        'Challenge': ['Class Imbalance', 'Format Inconsistency', 'Limited Diversity', 'Duplicate Images'],
        'Impact': ['High', 'Medium', 'Medium', 'Low'],
        'Solution': [
            'Data augmentation, weighted loss functions',
            'Unified conversion to YOLO format',
            'Synthetic data generation, diverse collection',
            'Deduplication algorithms, hash comparison'
        ],
        'Status': ['Implemented', 'Completed', 'In Progress', 'Completed']
    })
    
    st.dataframe(challenges, use_container_width=True)
    
    # Data Preprocessing Pipeline
    st.subheader("Data Preprocessing Pipeline")
    
    with st.expander("View Preprocessing Steps"):
        st.markdown("""
        **1. Data Collection & Integration**
        - Merge 9 different datasets
        - Handle various annotation formats (YOLO, Pascal VOC, COCO)
        
        **2. Format Unification**
        - Convert all annotations to YOLO TXT format
        - Standardize image formats and naming conventions
        
        **3. Class Mapping**
        - Map dataset-specific classes to 4 unified classes
        - Handle class synonyms and variations
        
        **4. Quality Control**
        - Remove corrupted or duplicate images
        - Validate annotation consistency
        - Check bounding box coordinates
        
        **5. Data Augmentation**
        - Horizontal flips, rotations
        - Brightness/contrast adjustments
        - Mosaic augmentation for small objects
        
        **6. Train/Val/Test Split**
        - 70% Training, 20% Validation, 10% Testing
        - Stratified split to maintain class distribution
        """)

def show_live_detection():
    st.header("🔍 Live PPE Detection System")
    
    # Detection Settings
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
        
        detection_classes = st.multiselect(
            "PPE Classes to Detect",
            ["Helmet", "Vest", "Gloves", "Boots"],
            default=["Helmet", "Vest", "Gloves", "Boots"]
        )
        
        real_time_mode = st.checkbox("Real-time Processing", value=False)
        save_results = st.checkbox("Save Detection Results", value=True)
    
    with col1:
        st.subheader("Image Upload & Detection")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Construction Site Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image from a construction site for PPE detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col_b:
                st.subheader("Detection Results")
                
                if st.button("Run PPE Detection", type="primary"):
                    with st.spinner("Running YOLOv8 detection..."):
                        import time
                        time.sleep(2)  # Simulate processing time
                        
                        # Mock detection results
                        detections = {
                            "workers_detected": np.random.randint(1, 6),
                            "helmets": np.random.randint(0, 5),
                            "vests": np.random.randint(0, 5),
                            "gloves": np.random.randint(0, 3),
                            "boots": np.random.randint(0, 4)
                        }
                        
                        # Simulate annotated image (in real implementation, this would be the actual detection result)
                        st.image(image, caption="Detection Result (Simulated)", use_column_width=True)
                        
                        # Detection summary
                        st.success("Detection completed!")
                        
                        col1_det, col2_det = st.columns(2)
                        with col1_det:
                            st.metric("Workers Detected", detections["workers_detected"])
                            st.metric("Helmets Detected", detections["helmets"])
                        
                        with col2_det:
                            st.metric("Vests Detected", detections["vests"])
                            st.metric("Safety Boots", detections["boots"])
                        
                        # Compliance analysis
                        compliance_rate = (detections["helmets"] + detections["vests"]) / (2 * detections["workers_detected"]) * 100
                        
                        if compliance_rate >= 90:
                            st.success(f"✅ Good Compliance: {compliance_rate:.1f}%")
                        elif compliance_rate >= 70:
                            st.warning(f"⚠️ Moderate Compliance: {compliance_rate:.1f}%")
                        else:
                            st.error(f"🚨 Poor Compliance: {compliance_rate:.1f}%")
                        
                        # Detailed detection results
                        with st.expander("Detailed Detection Results"):
                            detection_details = pd.DataFrame({
                                'Object': ['Worker 1', 'Worker 2', 'Worker 3'][:detections["workers_detected"]],
                                'Helmet': np.random.choice(['✅ Detected', '❌ Missing'], detections["workers_detected"]),
                                'Vest': np.random.choice(['✅ Detected', '❌ Missing'], detections["workers_detected"]),
                                'Gloves': np.random.choice(['✅ Detected', '❌ Missing', 'N/A'], detections["workers_detected"]),
                                'Boots': np.random.choice(['✅ Detected', '❌ Missing'], detections["workers_detected"]),
                                'Compliance': np.random.choice(['Compliant', 'Non-Compliant'], detections["workers_detected"])
                            })
                            st.dataframe(detection_details, use_container_width=True)
    
    # Live Camera Stream (Placeholder)
    st.subheader("📹 Live Camera Feed")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if real_time_mode:
            st.info("🔴 Live camera stream would be displayed here in a real implementation")
            st.image("https://via.placeholder.com/800x450/cccccc/000000?text=Live+Camera+Feed", 
                    caption="Live PPE Detection Stream (Placeholder)")
        else:
            st.info("Enable 'Real-time Processing' to activate live camera feed")
    
    with col2:
        st.subheader("Live Stats")
        if real_time_mode:
            st.metric("FPS", "30")
            st.metric("Latency", "45ms")
            st.metric("Workers in Frame", "3")
            st.metric("Violations", "1")
        else:
            st.metric("FPS", "—")
            st.metric("Latency", "—")
            st.metric("Workers in Frame", "—")
            st.metric("Violations", "—")

def show_analytics_reports():
    st.header("📈 Analytics & Safety Reports")
    
    # Report Generation Section
    st.subheader("📋 Generate Safety Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Daily Safety Summary", "Weekly Compliance Report", "Monthly Analysis", 
             "Incident Report", "Site Comparison", "Custom Analysis"]
        )
        
        report_format = st.selectbox(
            "Export Format",
            ["PDF Report", "Excel Spreadsheet", "CSV Data", "PowerPoint Presentation"]
        )
    
    with col2:
        include_charts = st.checkbox("Include Charts & Graphs", value=True)
        include_images = st.checkbox("Include Detection Images", value=False)
        include_recommendations = st.checkbox("Include Safety Recommendations", value=True)
        
        anonymize_data = st.checkbox("Anonymize Worker Data", value=True)
    
    with col3:
        st.write("**Report Period**")
        report_start = st.date_input("From", value=datetime.now() - timedelta(days=7))
        report_end = st.date_input("To", value=datetime.now())
        
        sites_to_include = st.multiselect(
            "Sites to Include",
            ["Site A - Building", "Site B - Bridge", "Site C - Highway", "Site D - Industrial"],
            default=["Site A - Building", "Site B - Bridge"]
        )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating safety report..."):
            import time
            time.sleep(3)
        
        st.success("✅ Report generated successfully!")
        
        # Mock report preview
        st.subheader("Report Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Executive Summary**")
            st.write(f"- Report Period: {report_start} to {report_end}")
            st.write(f"- Sites Covered: {len(sites_to_include)} construction sites")
            st.write(f"- Total Workers Monitored: 247")
            st.write(f"- Overall Compliance Rate: 87.3%")
            st.write(f"- Total Violations: 31")
            st.write(f"- Critical Issues: 3")
        
        with col2:
            st.write("**Key Findings**")
            st.write("- Helmet compliance highest at 94.2%")
            st.write("- Gloves compliance needs improvement (78.5%)")
            st.write("- Peak violation times: 10-11 AM, 2-3 PM")
            st.write("- Site A shows best compliance rates")
            st.write("- Weather correlation: compliance drops 5% in rain")
        
        st.download_button(
            label=f"📥 Download {report_format}",
            data="Mock report data - would contain actual report in real implementation",
            file_name=f"safety_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    
    # Historical Analytics
    st.subheader("📊 Historical Safety Analytics")
    
    # Compliance Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("30-Day Compliance Trend")
        dates = pd.date_range(start='2024-01-01', periods=30)
        compliance_trend = pd.DataFrame({
            'Date': dates,
            'Helmet': np.random.normal(94, 2, 30),
            'Vest': np.random.normal(90, 3, 30),
            'Gloves': np.random.normal(78, 5, 30),
            'Boots': np.random.normal(82, 4, 30)
        })
        
        fig = px.line(compliance_trend, x='Date', 
                     y=['Helmet', 'Vest', 'Gloves', 'Boots'],
                     title="PPE Compliance Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Violation Patterns")
        violation_patterns = pd.DataFrame({
            'Hour': list(range(6, 19)),
            'Violations': [2, 1, 4, 7, 5, 3, 8, 6, 4, 2, 5, 7, 3],
            'Severity': ['Low', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High', 
                        'Medium', 'Medium', 'Low', 'Medium', 'High', 'Medium']
        })
        
        fig = px.bar(violation_patterns, x='Hour', y='Violations', color='Severity',
                    title="Hourly Violation Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Analytics
    st.subheader("🔮 Predictive Safety Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Violations (Next 7 Days)",
            value="23",
            delta="Based on historical patterns"
        )
    
    with col2:
        st.metric(
            label="Risk Score",
            value="Medium",
            delta="Weather & activity factors"
        )
    
    with col3:
        st.metric(
            label="Recommended Inspections",
            value="Site B",
            delta="Highest risk probability"
        )
    
    # Safety Recommendations
    st.subheader("🎯 AI-Generated Safety Recommendations")
    
    recommendations = pd.DataFrame({
        'Priority': ['High', 'Medium', 'Medium', 'Low'],
        'Recommendation': [
            'Increase gloves compliance training - currently at 78.5%',
            'Schedule additional safety briefings during peak violation hours (10-11 AM)',
            'Install additional PPE reminder signage at Site B entrance',
            'Consider weather-resistant PPE options for rainy season'
        ],
        'Expected Impact': ['15% improvement', '8% improvement', '5% improvement', '3% improvement'],
        'Implementation Cost': ['Low', 'Medium', 'Low', 'High']
    })
    
    st.dataframe(recommendations, use_container_width=True)

def show_system_settings():
    st.header("⚙️ System Configuration & Settings")
    
    # Model Settings
    st.subheader("🎯 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detection Parameters**")
        
        confidence_threshold = st.slider(
            "Global Confidence Threshold", 
            0.1, 1.0, 0.5, 0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold (NMS)", 
            0.1, 1.0, 0.45, 0.05,
            help="Intersection over Union threshold for Non-Maximum Suppression"
        )
        
        max_detections = st.number_input(
            "Max Detections per Image", 
            min_value=10, max_value=1000, value=100,
            help="Maximum number of detections per image"
        )
        
        model_size = st.selectbox(
            "Model Size",
            ["YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)", "YOLOv8l (Large)", "YOLOv8x (Extra Large)"],
            index=0,
            help="Larger models are more accurate but slower"
        )
    
    with col2:
        st.write("**Processing Settings**")
        
        input_resolution = st.selectbox(
            "Input Resolution",
            ["320x320", "416x416", "512x512", "640x640", "832x832"],
            index=3,
            help="Higher resolution improves accuracy but increases processing time"
        )
        
        batch_size = st.number_input(
            "Batch Size", 
            min_value=1, max_value=32, value=1,
            help="Number of images processed simultaneously"
        )
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=True)
        
        enable_tracking = st.checkbox(
            "Enable Multi-Object Tracking", 
            value=False,
            help="Track workers across video frames"
        )
    
    # Alert Settings
    st.subheader("🚨 Alert & Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Alert Thresholds**")
        
        compliance_threshold = st.slider(
            "Compliance Alert Threshold (%)", 
            50, 100, 90, 5,
            help="Send alert when compliance drops below this percentage"
        )
        
        violation_threshold = st.number_input(
            "Daily Violation Alert Threshold", 
            min_value=1, max_value=100, value=10,
            help="Send alert when daily violations exceed this number"
        )
        
        critical_ppe = st.multiselect(
            "Critical PPE Items",
            ["Helmet", "Vest", "Gloves", "Boots"],
            default=["Helmet"],
            help="Missing these items trigger immediate alerts"
        )
    
    with col2:
        st.write("**Notification Channels**")
        
        email_alerts = st.checkbox("Email Notifications", value=True)
        if email_alerts:
            email_recipients = st.text_area(
                "Email Recipients",
                value="safety@company.com\nmanager@company.com",
                help="One email per line"
            )
        
        sms_alerts = st.checkbox("SMS Notifications", value=False)
        if sms_alerts:
            sms_numbers = st.text_area(
                "SMS Numbers",
                value="+1234567890",
                help="One number per line"
            )
        
        webhook_alerts = st.checkbox("Webhook Integration", value=False)
        if webhook_alerts:
            webhook_url = st.text_input("Webhook URL")
    
    # Camera Settings
    st.subheader("📹 Camera & Monitoring Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Camera Configuration**")
        
        camera_fps = st.slider("Camera FPS", 1, 60, 30)
        camera_resolution = st.selectbox(
            "Camera Resolution",
            ["1280x720", "1920x1080", "2560x1440", "3840x2160"],
            index=1
        )
        
        recording_enabled = st.checkbox("Enable Recording", value=False)
        if recording_enabled:
            recording_duration = st.number_input(
                "Recording Duration (hours)", 
                min_value=1, max_value=168, value=24
            )
    
    with col2:
        st.write("**Monitoring Zones**")
        
        zone_detection = st.checkbox("Enable Zone-based Detection", value=False)
        if zone_detection:
            st.info("Configure detection zones on the Live Detection page")
        
        schedule_monitoring = st.checkbox("Scheduled Monitoring", value=True)
        if schedule_monitoring:
            start_time = st.time_input("Start Time", value=datetime.strptime("06:00", "%H:%M").time())
            end_time = st.time_input("End Time", value=datetime.strptime("18:00", "%H:%M").time())
    
    # Database Settings
    st.subheader("💾 Data Storage & Retention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Storage Settings**")
        
        save_detections = st.checkbox("Save Detection Results", value=True)
        save_images = st.checkbox("Save Original Images", value=False)
        save_annotated = st.checkbox("Save Annotated Images", value=True)
        
        retention_period = st.selectbox(
            "Data Retention Period",
            ["7 days", "30 days", "90 days", "1 year", "Indefinite"],
            index=2
        )
    
    with col2:
        st.write("**Export Settings**")
        
        auto_backup = st.checkbox("Automatic Backup", value=True)
        if auto_backup:
            backup_frequency = st.selectbox(
                "Backup Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=1
            )
        
        export_format = st.multiselect(
            "Default Export Formats",
            ["CSV", "JSON", "Excel", "PDF"],
            default=["CSV", "PDF"]
        )
    
    # Save Settings
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save All Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
            st.balloons()
    
    # System Status
    st.subheader("🖥️ System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online")
    
    with col2:
        st.metric("GPU Memory", "2.1/8.0 GB")
    
    with col3:
        st.metric("CPU Usage", "23%")
    
    with col4:
        st.metric("Storage Used", "145/500 GB")

if __name__ == "__main__":
    main()