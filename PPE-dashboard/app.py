import streamlit as st
from pathlib import Path
from datetime import datetime
import queue
import logging
import pandas as pd
import cv2
import threading # Need this for threading.Event
from collections import deque
import plotly.express as px # <--- ADDED THIS IMPORT
import time # <--- ADDED THIS IMPORT

# Import constants
from config import *

# Import all helpers, classes, and workers from our new utils.py
from utils import (
    WorkerTracker, CameraWorker, InferenceWorker,
    list_models, load_model,
    generate_session_report, get_table_download_link, # <--- REMOVED create_gauge_chart
    save_detection, save_violation_clip,
    annotate_frame_with_workers, # We need this for snapshots
)

# --- Page Config and Paths (Main App) ---

# Configure logging
logging.basicConfig(level=logging.INFO)
# Reduce Streamlit warnings
try:
    logging.getLogger("streamlit").setLevel(logging.ERROR)
except Exception:
    pass

# Page config
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .alert-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .compliant-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .violation-badge {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    /* Hide plotly modebar */
    .plotly .modebar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Main App ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦺 PPE Detection System</h1>
        <p>Construction Site Safety Monitoring | DPL302m Deep Learning Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'camera_running' not in st.session_state:
        st.session_state['camera_running'] = False
    if 'tracker' not in st.session_state:
        st.session_state['tracker'] = WorkerTracker()
    if 'session_stats' not in st.session_state:
        st.session_state['session_stats'] = {
            'start_time': None, 'total_detections': 0,
            'total_violations': 0, 'violations_log': []
        }
    if 'violation_buffer' not in st.session_state:
        st.session_state['violation_buffer'] = deque(maxlen=150)  # 5 seconds @ 30fps
    if 'last_payload' not in st.session_state:
        st.session_state['last_payload'] = None # For snapshot
    
    if 'last_chart_update' not in st.session_state:
        st.session_state['last_chart_update'] = 0
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("🎯 Model Settings")
        
        models = list_models() # Uses MODELS_DIR from config
        if models:
            model_key = st.selectbox("Select Model", options=list(models.keys()))
            
            if st.button("🔄 Load Model"):
                with st.spinner("Loading model..."):
                    st.session_state['model'] = load_model(models[model_key])
                    if st.session_state['model']:
                        st.success(f"✅ Loaded: {model_key}")
                        st.rerun() # Rerun to update button state
                    else:
                        st.error("Model failed to load. Check console for logs.")
        else:
            st.error(f"No models found in {MODELS_DIR} directory")
        
        st.markdown("---")
        
        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
        
        st.markdown("---")
        st.subheader("🦺 PPE to Monitor")
        monitor_helmet = st.checkbox("Helmet", value=True)
        monitor_vest = st.checkbox("Vest", value=True)
        monitor_gloves = st.checkbox("Gloves", value=True)
        monitor_boots = st.checkbox("Boots", value=True)
        
        ppe_to_monitor = []
        if monitor_helmet: ppe_to_monitor.append('helmet')
        if monitor_vest: ppe_to_monitor.append('vest')
        if monitor_gloves: ppe_to_monitor.append('gloves')
        if monitor_boots: ppe_to_monitor.append('boots')
        
        st.markdown("---")
        st.subheader("🎬 Recording Settings")
        auto_record = st.checkbox("Auto-record violations", value=True)
        save_detections_csv = st.checkbox("Save detections to CSV", value=True)
        
        st.markdown("---")
        st.subheader("📊 Session Info")
        start_time = st.session_state['session_stats']['start_time']
        if start_time:
            duration = datetime.now() - start_time
            st.metric("Session Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")
        st.metric("Total Detections", st.session_state['session_stats']['total_detections'])
        st.metric("Total Violations", st.session_state['session_stats']['total_violations'])
    
    # --- Main Layout ---
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            start_btn = st.button(
                "🔴 Start Camera", 
                type="primary", 
                disabled=st.session_state['model'] is None 
            )
        with col_btn2:
            stop_btn = st.button("⏹️ Stop Camera")
        with col_btn3:
            snapshot_btn = st.button("📸 Snapshot")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        fps_metric = col_s1.empty()
        workers_metric = col_s2.empty()
        compliant_metric = col_s3.empty()
        violations_metric = col_s4.empty()
    
    with col_right:
        st.subheader("📊 Live Statistics")
        stats_placeholder = st.empty() # <--- FIX: Changed to st.empty()
        
        st.markdown("---")
        st.subheader("🚨 Active Alerts")
        alerts_container = st.empty() # <--- FIX: Changed to st.empty()
    
    # --- Bottom Tabs ---
    tab1, tab2, tab3 = st.tabs(["📈 Real-time Analytics", "📋 Violation History", "📊 Session Summary"])
    
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            compliance_chart = st.empty()
        with col_chart2:
            timeline_chart = st.empty()
    
    with tab2:
        violations_table = st.empty() # <--- FIX: Changed to st.empty()
    
    with tab3:
        summary_container = st.empty() # <--- FIX: Changed to st.empty()
        report_container = st.container() 

        if st.button("📄 Generate Report", key='generate_report_btn'):
            if st.session_state['session_stats']['start_time'] is None:
                st.warning("Please run the camera first to generate session data.")
            else:
                with st.spinner("Generating report..."):
                    violations_log_df = pd.DataFrame()
                    if len(st.session_state['session_stats']['violations_log']) > 0:
                        violations_log_df = pd.DataFrame(st.session_state['session_stats']['violations_log'])

                    report_content = generate_session_report(st.session_state['session_stats'], violations_log_df)
                    report_filename = f"ppe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    
                    (OUT_REPORTS / report_filename).write_text(report_content)
                    
                    with report_container:
                        st.markdown(
                            get_table_download_link(report_content, report_filename, f"📥 Download Report: {report_filename}"),
                            unsafe_allow_html=True
                        )
                    st.success(f"✅ Report saved to {OUT_REPORTS / report_filename}")
    
    # --- Button Logic ---
    
    if start_btn:
        st.session_state['camera_running'] = True
        st.session_state['session_stats'] = {
            'start_time': datetime.now(),
            'total_detections': 0,
            'total_violations': 0,
            'violations_log': []
        }
        st.session_state['tracker'] = WorkerTracker()
        st.session_state['last_payload'] = None
        st.session_state['last_chart_update'] = 0 
        st.rerun() 
        
    if stop_btn:
        st.session_state['camera_running'] = False
        if 'stop_event' in st.session_state:
            st.session_state['stop_event'].set()
        st.session_state['last_payload'] = None
        st.rerun()
    
    if snapshot_btn:
        if st.session_state['last_payload']:
            payload = st.session_state['last_payload']
            frame = payload['annotated'] 
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = OUT_FRAMES / f"snapshot_{ts}.png"
            img_bgr = cv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filename), img_bgr)
            st.success(f"📸 Snapshot saved to {filename}")
        else:
            st.warning("No camera feed to snapshot. Please start the camera.")

    # --- Main Camera Loop ---
    
    if st.session_state['camera_running']:
        # Initialize resources
        if 'frame_q' not in st.session_state:
            st.session_state['frame_q'] = queue.Queue(maxsize=3)
        if 'result_q' not in st.session_state:
            st.session_state['result_q'] = queue.Queue(maxsize=2)
        if 'stop_event' not in st.session_state or st.session_state['stop_event'].is_set():
            st.session_state['stop_event'] = threading.Event()

        # Start camera worker
        if 'camera_thread' not in st.session_state or not st.session_state['camera_thread'].is_alive():
            cam_worker = CameraWorker(0, st.session_state['frame_q'], st.session_state['stop_event'])
            st.session_state['camera_thread'] = cam_worker
            cam_worker.start()

        # Start inference worker
        if 'inference_thread' not in st.session_state or not st.session_state['inference_thread'].is_alive():
            
            model_object = st.session_state.get('model')
            if model_object is None:
                st.error("Model not loaded! Stopping camera.")
                st.session_state['camera_running'] = False
                st.rerun()
                return 
            
            inf_worker = InferenceWorker(
                st.session_state['frame_q'],
                st.session_state['result_q'],
                model_object, 
                lambda: conf_thresh,
                lambda: ppe_to_monitor,
                st.session_state['stop_event'],
                st.session_state['tracker'],
                st.session_state['violation_buffer']
            )
            st.session_state['inference_thread'] = inf_worker
            inf_worker.start()
        
        while st.session_state.camera_running:
            try:
                # 1. Wait for a result from the inference thread
                payload = st.session_state['result_q'].get(timeout=0.1) 
                st.session_state['last_payload'] = payload
                
                annotated = payload['annotated']
                workers = payload['workers']
                fps = payload['fps']
                frame_count = payload['frame_count']
                required_ppe = payload['required_ppe']
                
                current_time = time.time()
                update_charts = False
                if (current_time - st.session_state['last_chart_update']) > 1.0: # Update charts every 1 second
                    st.session_state['last_chart_update'] = current_time
                    update_charts = True
                
                # ==============================================
                # --- HIGH-FREQUENCY UPDATES (Every Frame) ---
                # ==============================================
                
                # Update UI
                video_placeholder.image(annotated, use_container_width=True)
                
                fps_metric.metric("FPS", f"{fps:.1f}")
                workers_metric.metric("👷 Workers", len(workers))
                
                # Calculate compliance
                compliant_count = 0
                violations_count = 0
                
                if workers:
                    for w in workers:
                        missing = [p for p in required_ppe if not w['ppe'].get(p, False)]
                        if not missing:
                            compliant_count += 1
                    violations_count = len(workers) - compliant_count
                
                if len(workers) > 0:
                    compliance_rate = (compliant_count / len(workers)) * 100
                    compliant_metric.metric("✅ Compliant", f"{compliant_count} ({compliance_rate:.0f}%)")
                    violations_metric.metric("⚠️ Violations", violations_count)
                else:
                    compliant_metric.metric("✅ Compliant", "0")
                    violations_metric.metric("⚠️ Violations", "0")
                
                # Update session stats
                if 'last_processed_frame' not in st.session_state:
                    st.session_state['last_processed_frame'] = -1
                
                if frame_count > st.session_state['last_processed_frame']:
                    st.session_state['last_processed_frame'] = frame_count
                    st.session_state['session_stats']['total_detections'] += len(workers)
                    
                    # Process violations
                    for worker in workers:
                        missing = [p for p in required_ppe if not worker['ppe'].get(p, False)]
                        
                        if missing:
                            st.session_state['session_stats']['total_violations'] += 1
                            
                            alert_data = {
                                'Time': payload['ts'],
                                'Worker': f"Worker #{worker['id']}",
                                'Missing PPE': [p.upper() for p in missing],
                                'Frame': frame_count
                            }
                            st.session_state['session_stats']['violations_log'].append(alert_data)
                            
                            # Auto-record violation clip
                            if auto_record and len(st.session_state['violation_buffer']) >= 150:
                                last_log = st.session_state['session_stats']['violations_log']
                                if len(last_log) > 1:
                                    try:
                                        last_ts = datetime.fromisoformat(last_log[-2]['Time'])
                                        if (datetime.fromisoformat(payload['ts']) - last_ts).total_seconds() < 10:
                                            continue
                                    except Exception:
                                        pass
                                
                                save_violation_clip(
                                    list(st.session_state['violation_buffer']),
                                    worker['id'],
                                    missing
                                )
                        
                        # Save individual detection
                        if save_detections_csv:
                            detection_rec = {
                                'timestamp': payload['ts'],
                                'session_id': st.session_state['session_stats']['start_time'].strftime('%Y%m%d_%H%M%S'),
                                'worker_id': worker['id'],
                                'helmet': worker['ppe'].get('helmet', False),
                                'vest': worker['ppe'].get('vest', False),
                                'gloves': worker['ppe'].get('gloves', False),
                                'boots': worker['ppe'].get('boots', False),
                                'compliance_status': 'compliant' if not missing else 'violation',
                                'frame_number': frame_count
                            }
                            save_detection(detection_rec)
                
                # Update Active Alerts (Every Frame)
                # --- FIX: Replaced .empty() with .container() ---
                with alerts_container.container():
                    alerts_from_payload = []
                    for worker in workers:
                         missing = [p for p in required_ppe if not worker['ppe'].get(p, False)]
                         if missing:
                             alerts_from_payload.append({
                                'Worker': f"Worker #{worker['id']}",
                                'Missing PPE': [p.upper() for p in missing],
                                'Time': payload['ts']
                             })

                    if alerts_from_payload:
                        for alert in alerts_from_payload[-5:]:
                            st.markdown(f"""
                            <div class="alert-card">
                                <strong>⚠️ {alert['Worker']}</strong><br>
                                <span class="violation-badge">Missing: {', '.join(alert['Missing PPE'])}</span><br>
                                <small>🕐 {alert['Time']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="compliant-card">
                            <strong>✅ All Workers Compliant</strong><br>
                            <small>No active violations detected</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # =============================================
                # --- LOW-FREQUENCY UPDATES (Throttled) ---
                # =============================================
                
                if update_charts:
                    # Update Stats (Throttled)
                    # --- FIX: Replaced .empty() with .container() ---
                    with stats_placeholder.container():
                        if len(workers) > 0:
                            ppe_stats = {'helmet': 0, 'vest': 0, 'gloves': 0, 'boots': 0}
                            for worker in workers:
                                for ppe_item in ppe_stats:
                                    if worker['ppe'].get(ppe_item, False):
                                        ppe_stats[ppe_item] += 1
                            
                            ppe_percentages = {k: (v / len(workers)) * 100 for k, v in ppe_stats.items()}
                            
                            col_g1, col_g2 = st.columns(2)
                            col_g3, col_g4 = st.columns(2)
                            
                            with col_g1:
                                st.metric("🪖 Helmet", f"{ppe_percentages.get('helmet', 0):.0f}%")
                            with col_g2:
                                st.metric("🦺 Vest", f"{ppe_percentages.get('vest', 0):.0f}%")
                            with col_g3:
                                st.metric("🧤 Gloves", f"{ppe_percentages.get('gloves', 0):.0f}%")
                            with col_g4:
                                st.metric("👢 Boots", f"{ppe_percentages.get('boots', 0):.0f}%")
                        else:
                            st.info("Stats will appear here when workers are detected.")
                    
                    # Update Analytics Tabs (Throttled)
                    violations_log_df = pd.DataFrame()
                    if len(st.session_state['session_stats']['violations_log']) > 0:
                        violations_log_df = pd.DataFrame(st.session_state['session_stats']['violations_log'])
                    
                    if len(workers) > 0:
                        # This logic seems fine, ppe_percentages is defined above
                        ppe_data = pd.DataFrame({
                            'PPE Item': ['Helmet', 'Vest', 'Gloves', 'Boots'],
                            'Compliance %': [
                                ppe_percentages.get('helmet', 0),
                                ppe_percentages.get('vest', 0),
                                ppe_percentages.get('gloves', 0),
                                ppe_percentages.get('boots', 0)
                            ]
                        })
                        
                        fig = px.bar(
                            ppe_data, x='PPE Item', y='Compliance %',
                            title='Real-time PPE Compliance',
                            color='Compliance %',
                            color_continuous_scale=[(0, '#ef4444'), (0.7, '#f97316'), (0.9, '#fde047'), (1, '#22c55e')],
                            range_color=[0, 100]
                        )
                        fig.update_layout(height=300, showlegend=False, yaxis_range=[0,100])
                        fig.update_layout(datarevision=time.time())
                        compliance_chart.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    if not violations_log_df.empty:
                        violations_log_df['Time'] = pd.to_datetime(violations_log_df['Time'])
                        violations_by_time = violations_log_df.groupby(violations_log_df['Time'].dt.floor('10s')).size().reset_index(name='Violations')
                            
                        fig = px.line(
                            violations_by_time, x='Time', y='Violations',
                                title='Violations Timeline', markers=True
                        )
                        fig.update_traces(line_color='#ef4444')
                        fig.update_layout(height=300)
                        fig.update_layout(datarevision=time.time())
                        timeline_chart.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Update Violation Table and Summary (Throttled)
                    # --- FIX: Replaced .empty() with .container() ---
                    with violations_table.container():
                        if not violations_log_df.empty:
                            violations_display = violations_log_df.tail(10).sort_index(ascending=False)
                            violations_display['Missing PPE'] = violations_display['Missing PPE'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                            st.dataframe(violations_display, use_container_width=True, hide_index=True)
                        else:
                            st.info("No violations recorded in this session yet")
                    
                    # --- FIX: Replaced .empty() with .container() ---
                    with summary_container.container():
                        if st.session_state['session_stats']['start_time']:
                            duration = datetime.now() - st.session_state['session_stats']['start_time']
                            duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m {duration.seconds % 60}s"
                            
                            total_det = st.session_state['session_stats']['total_detections']
                            total_viol = st.session_state['session_stats']['total_violations']
                            compliance_rate = ((total_det - total_viol) / total_det * 100) if total_det > 0 else 100
                            
                            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                            with col_sum1:
                                st.metric("Session Duration", duration_str)
                            with col_sum2:
                                st.metric("Total Detections", total_det)
                            with col_sum3:
                                st.metric("Total Violations", total_viol)
                            with col_sum4:
                                st.metric("Compliance Rate", f"{compliance_rate:.1f}%")

            except queue.Empty:
                if not st.session_state.camera_running:
                    break 
                pass
            
            except Exception as e:
                st.error(f"An error occurred in the main loop: {e}")
                logging.error(f"Main loop error: {e}", exc_info=True)
                st.session_state['camera_running'] = False
                break

        # --- Loop exited ---
        if 'stop_event' in st.session_state:
            st.session_state['stop_event'].set()
        
        video_placeholder.info("Camera stopping...")
    
    else:
        # Camera not running - show idle screen
        
        if 'stop_event' in st.session_state and not st.session_state['stop_event'].is_set():
            st.session_state['stop_event'].set()
            video_placeholder.info("Camera stopping...")
        else:
            video_placeholder.info(f"""
            ### 📹 Camera Controls
            1. **Select a model** from the sidebar.
            2. Click **Load Model** button.
            3. Adjust **Confidence Threshold** if needed.
            4. Select the **PPE to Monitor** using the checkboxes.
            5. Click **🔴 Start Camera** to begin monitoring.
            
            **Status:** {"✅ Model Loaded! Ready to start." if st.session_state['model'] else "⚠️ Model Not Loaded. Please load a model."}
            """)
        
        # --- FIX: Replaced .empty() with .container() ---
        with alerts_container.container():
            st.markdown("""
            <div class.compliant-card">
                <strong>💤 System Idle</strong><br>
                <small>Start camera to begin monitoring</small>
            </div>
            """, unsafe_allow_html=True)
        
        # --- FIX: Replaced .empty() with .container() ---
        with stats_placeholder.container():
            st.info("Stats will appear here when the camera is running.")

if __name__ == "__main__":
    main()

