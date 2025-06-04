import socket
import struct
from landmarks import landmarks_dist

HOST = '0.0.0.0'
PORT = 5005
BUFFER_SIZE = 65536
IDLE_TIMEOUT = None  # No timeout while waiting for first data
PROCESSING_TIMEOUT = 30.0  # 30-second timeout during processing

def recv_exact(sock, n):
    """Receive exactly n bytes with no timeout"""
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
        
        while True:  # Main loop to keep connection alive
            try:
                # Phase 1: Wait indefinitely for first data
                conn.settimeout(IDLE_TIMEOUT)
                
                # Receive PNG length (blocks forever until data arrives)
                png_len_bytes = recv_exact(conn, 4)
                if not png_len_bytes:  # Client closed connection
                    print(f"Client {addr} gracefully disconnected")
                    break
                    
                png_len = struct.unpack('<I', png_len_bytes)[0]
                print(f"Receiving PNG ({png_len/1024:.1f} KB)...")
                
                # Phase 2: Switch to processing timeout
                conn.settimeout(PROCESSING_TIMEOUT)
                
                # Receive PNG data
                png_data = recv_exact(conn, png_len)
                
                # Receive RAW length
                raw_len = struct.unpack('<I', recv_exact(conn, 4))[0]
                print(f"Receiving RAW ({raw_len/1024:.1f} KB)...")
                
                # Receive RAW data
                raw_data = recv_exact(conn, raw_len)

                # Process data with timeout protection
                print("Processing landmarks...")
                results = landmarks_dist(png_data, raw_data)
                
                print(results)  # Print results for debugging

                # Send response
                conn.sendall(results.encode('utf-8') + b'\x00')
                print("Response sent successfully")
                
                # Reset to idle timeout for next request
                conn.settimeout(IDLE_TIMEOUT)

            except socket.timeout:
                print(f"Processing timeout with {addr} - operation took too long")
                conn.sendall(b"Error: Processing timeout\x00")
                continue  # Continue to next request
                
            except struct.error:
                print(f"Client {addr} sent malformed data length")
                break  # Break connection on protocol errors
                
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                conn.sendall(f"Error: {str(e)}\x00".encode())
                continue  # Continue to next request

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
                # Handle client in same thread (for simplicity)
                handle_client(conn, addr)
            except KeyboardInterrupt:
                print("\nServer shutting down...")
                break
            except Exception as e:
                print(f"Error accepting connection: {str(e)}")
                continue

if __name__ == "__main__":
    run_server()