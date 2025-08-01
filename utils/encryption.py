from cryptography.fernet import Fernet
from decouple import config

class EncryptionHelper:
    def __init__(self):
        self.fernet_key = config("FERNET_KEY")
        self.fernet = Fernet(self.fernet_key.encode())

    def decrypt(self, encrypted_text: str) -> str:
        return self.fernet.decrypt(encrypted_text.encode()).decode()

    def encrypt(self, plain_text: str) -> str:
        return self.fernet.encrypt(plain_text.encode()).decode()
