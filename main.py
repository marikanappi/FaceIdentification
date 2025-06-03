from run import run_training
from test import run_test
from server import run_server  # supponendo che tu metta il codice socket in server.py

if __name__ == "__main__":
    run_training()
    run_test()
    print("Avvio server TCP per ricevere immagini da Unity...")
    run_server()  