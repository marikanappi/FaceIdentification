import socket
import struct
import json
from landmarks import landmarks_dist

HOST = '0.0.0.0'
PORT = 5005
BUFFER_SIZE = 65536  # Buffer aumentato a 64KB
TIMEOUT = 10.0       # Timeout aumentato a 10 secondi

def recv_exact(sock, n):
    """Riceve esattamente n byte dal socket."""
    data = b''
    while len(data) < n:
        remaining = n - len(data)
        packet = sock.recv(min(remaining, BUFFER_SIZE))
        if not packet:
            raise ConnectionError("Connessione chiusa prematuramente")
        data += packet
    return data

def handle_client(conn, addr):
    try:
        conn.settimeout(TIMEOUT)
        print(f"\nConnessione accettata da {addr}")

        # 1. Ricevi lunghezza PNG
        png_len = struct.unpack('<I', recv_exact(conn, 4))[0]
        print(f"Ricezione PNG ({png_len/1024:.1f} KB)...")

        # 2. Ricevi dati PNG
        png_data = recv_exact(conn, png_len)
        
        # 3. Ricevi lunghezza RAW
        raw_len = struct.unpack('<I', recv_exact(conn, 4))[0]
        print(f"Ricezione RAW ({raw_len/1024:.1f} KB)...")

        # 4. Ricevi dati RAW
        raw_data = recv_exact(conn, raw_len)

        # Salva i file temporanei (opzionale, per debug)
        with open("received.png", "wb") as f:
            f.write(png_data)
        with open("received.raw", "wb") as f:
            f.write(raw_data)

        # Elaborazione immagini
        print("Elaborazione landmarks...")
        results = landmarks_dist("received.png", "received.raw")  #passare i risultati al modello

        print('distanze',results)

        # Invia risposta con terminatore null
        conn.sendall(results.encode('utf-8') + b'\x00')
        print("Risposta inviata con successo")

    except ConnectionError as e:
        print(f"Errore di connessione: {str(e)}")
        conn.sendall(f"ConnectionError: {str(e)}\x00".encode())
    except Exception as e:
        print(f"Errore nell'elaborazione: {str(e)}")
        conn.sendall(f"ProcessingError: {str(e)}\x00".encode())
    finally:
        conn.close()
        print(f"Connessione con {addr} chiusa")

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server avviato su {HOST}:{PORT} (Timeout: {TIMEOUT}s)")

        while True:
            try:
                conn, addr = s.accept()
                handle_client(conn, addr)
            except Exception as e:
                print(f"Errore nell'accettare connessione: {str(e)}")

if __name__ == "__main__":
    run_server()