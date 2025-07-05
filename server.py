import socket
import struct
from landmarks import landmarks_dist, show_overlay
import torch
from model import FaceClassifier
import joblib
import numpy as np
import os
from tempfile import NamedTemporaryFile
from predict import predict_identity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceClassifier(input_dim=40, num_classes=111)  
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

HOST = '0.0.0.0'
PORT = 5005
BUFFER_SIZE = 65536
IDLE_TIMEOUT = None
PROCESSING_TIMEOUT = 30.0

def recv_exact(sock, n):
    sock.settimeout(IDLE_TIMEOUT)
    data = b''
    while len(data) < n:
        remaining = n - len(data)
        packet = sock.recv(min(remaining, BUFFER_SIZE))
        if not packet:
            raise ConnectionError("Connection closed prematurely")
        data += packet
    return data

def handle_client(conn, addr):
    try:
        print(f"\nConnection accepted from {addr}")
        
        while True:
            try:
                conn.settimeout(IDLE_TIMEOUT)

                png_len_bytes = recv_exact(conn, 4)
                if not png_len_bytes:
                    print(f"Client {addr} gracefully disconnected")
                    break
                    
                png_len = struct.unpack('<I', png_len_bytes)[0]
                print(f"Receiving PNG ({png_len/1024:.1f} KB)...")
                
                conn.settimeout(PROCESSING_TIMEOUT)
                png_data = recv_exact(conn, png_len)

                from PIL import Image
                import io

                raw_len = struct.unpack('<I', recv_exact(conn, 4))[0]
                print(f"Receiving RAW ({raw_len/1024:.1f} KB)...")
                raw_data = recv_exact(conn, raw_len)

                with NamedTemporaryFile(delete=False, suffix='.png') as png_file, \
                     NamedTemporaryFile(delete=False, suffix='.raw') as raw_file:
                    png_file.write(png_data)
                    raw_file.write(raw_data)
                    png_path = png_file.name
                    raw_path = raw_file.name

                try:
                    print("Processing landmarks...")
                    features = landmarks_dist(png_path, raw_path, "unknown", "neutral")
                    '''show_overlay(png_path, raw_path, 640, 480)'''
                    predicted_label, confidence = predict_identity(features)

                    print(f"Prediction: {predicted_label} (confidence={confidence:.2f})")
                    conn.sendall(predicted_label.encode('utf-8') + b'\x00')

                finally:

                    os.unlink(png_path)
                    os.unlink(raw_path)

            except socket.timeout:
                print(f"Processing timeout with {addr}")
                conn.sendall(b"Error: Processing timeout\x00")
                continue
                
            except struct.error:
                print(f"Client {addr} sent malformed data length")
                break
                
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                conn.sendall(f"Error: {str(e)}\x00".encode())
                continue

    except ConnectionError as e:
        print(f"Connection error with {addr}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with {addr}: {str(e)}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed")

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server started on {HOST}:{PORT} (Processing timeout: {PROCESSING_TIMEOUT}s)")
        
        while True:
            try:
                conn, addr = s.accept()
                handle_client(conn, addr)
            except KeyboardInterrupt:
                print("\nServer shutting down...")
                break
            except Exception as e:
                print(f"Error accepting connection: {str(e)}")
                continue