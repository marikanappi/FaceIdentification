import socket
import pyrealsense2 as rs
import torch
from landmarks import landmarks_dist
from model import FaceClassifier
from predict import predict_identity
import numpy as np
import json  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FaceClassifier(input_dim=22, num_classes=111)  
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def capture_and_process():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Align depth to color frame
    align = rs.align(rs.stream.color)  
    pipeline.start(config)

    try:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned frames")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        print("Processing landmarks...")
        features = landmarks_dist(color_image, depth_image, "unknown", "neutral")    
        identity, confidence = predict_identity(features)
        print(f"✅ Prediction: {identity} (confidence={confidence:.2f})")
        
        # Return both identity and confidence
        return {"identity": identity, "confidence": float(confidence)}
    
    except Exception as e:
        print(f"❌ Error in capture_and_process: {str(e)}")
        return {"error": str(e)}
    finally:
        pipeline.stop()

def run_server():
    host = "127.0.0.1"
    port = 12345

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # NEW: Allow quick restart
        s.bind((host, port))
        s.listen()
        print(f"Python server started on {host}:{port}. Waiting for Unity...")
        
        while True:  
            conn, addr = s.accept()
            print(f"Connected to Unity: {addr}")
            
            try:
                with conn:
                    while True:
                        data = conn.recv(1024).decode().strip()
                        if not data:
                            break

                        if data == "CAPTURE":
                            result = capture_and_process()
                            # Send JSON for structured data
                            conn.sendall(json.dumps(result).encode() + b'\n')  
                        elif data == "EXIT":
                            print("Unity requested disconnect.")
                            break
            except ConnectionResetError:
                print("Unity disconnected abruptly")
            except Exception as e:
                print(f"Server error: {str(e)}")

'''if _name_ == "_main_":
    run_server()'''