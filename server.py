import socket
import struct
import json
from landmarks import landmarks_dist

HOST = '0.0.0.0'
PORT = 5005

def recv_exact(sock, n):
    """Riceve esattamente n byte dal socket."""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server in ascolto su {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connessione da {addr}")

                # Ricevi la lunghezza PNG (4 byte, int)
                data = recv_exact(conn, 4)
                if data is None:
                    print("Connessione chiusa prematuramente")
                    continue
                png_len = struct.unpack('!I', data)[0]  # network byte order

                # Ricevi PNG
                png_bytes = recv_exact(conn, png_len)
                if png_bytes is None:
                    print("Errore ricezione PNG")
                    continue
                with open("input.png", "wb") as f:
                    f.write(png_bytes)

                # Ricevi lunghezza RAW
                data = recv_exact(conn, 4)
                if data is None:
                    print("Connessione chiusa prematuramente")
                    continue
                raw_len = struct.unpack('!I', data)[0]

                # Ricevi RAW
                raw_bytes = recv_exact(conn, raw_len)
                if raw_bytes is None:
                    print("Errore ricezione RAW")
                    continue
                with open("input.raw", "wb") as f:
                    f.write(raw_bytes)

                # Chiama landmarks_dist
                distances = landmarks_dist("input.png", "input.raw")

                if distances is None:
                    response = "No face detected"
                else:
                    response = json.dumps(distances)

                # Invia la risposta come stringa terminata da \0 (per allinearsi allo script Unity)
                conn.sendall(response.encode('utf-8') + b'\0')

                print("Risposta inviata.")


