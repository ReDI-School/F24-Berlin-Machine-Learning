# Define the file path
import pandas as pd 
import random


from sklearn.metrics import f1_score
import pandas as pd
from cryptography.fernet import Fernet
import base64
import hashlib
import io
import numpy as np
from datetime import datetime
import json


def encrypt_csv(input_csv: str, password: str) -> None:
    """
    Encrypt a CSV file and save it with a "_pwd.csv" suffix.
    Args:
        input_csv (str): Path to the input CSV file.
        password (str): Password to encrypt the file.
    """
    # Generate the encryption key
    key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
    fernet = Fernet(key)

    # Read the CSV file
    with open(input_csv, "rb") as file:
        data = file.read()

    # Encrypt the CSV content
    encrypted_data = fernet.encrypt(data)

    # Save the encrypted file with "_pwd.csv" suffix
    encrypted_file = input_csv.replace(".csv", "_pwd.csv")
    with open(encrypted_file, "wb") as file:
        file.write(encrypted_data)

pw = 'save_pw123'
paths="Projects/Toxic comments/data/test_labels.csv"
encrypt_csv(paths, pw)