from run import run_training
from test import run_test
import numpy as np
from config import config

if __name__ == "__main__":
    seed = 0 
    run_training(seed=seed)  
    run_test() 
    '''print("Avvio server TCP per ricevere immagini da Unity...")
    config()
    from server import run_server  
    run_server()'''