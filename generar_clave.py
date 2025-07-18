#1. Genera el FERNET_KEY
from cryptography.fernet import Fernet

def generate_key():
    key = Fernet.generate_key()
    print("FERNET_KEY:", key.decode())

if __name__ == "__main__":
    generate_key()

#2. test_encrypt.py para encriptar cadena
# from cryptography.fernet import Fernet
# from decouple import config

# class EncryptionHelper:
#     def __init__(self):
#         key = config("FERNET_KEY")
#         self.cipher = Fernet(key)

#     def encrypt(self, plaintext: str) -> str:
#         return self.cipher.encrypt(plaintext.encode()).decode()

#     def decrypt(self, encrypted: str) -> str:
#         return self.cipher.decrypt(encrypted.encode()).decode()

# if __name__ == "__main__":
#     crypto = EncryptionHelper()
    
#     # Valor que deseas encriptar, tomado del .env
#     db_url = config("DATABASE_URL")
#     print(db_url)
    
#     encrypted_url = crypto.encrypt(db_url)
#     print("ENCRYPTED_DATABASE_URL =", encrypted_url)

#3. Script para descifrar cadena
# from cryptography.fernet import Fernet
# from decouple import config

# class EncryptionHelper:
#     def __init__(self):
#         key = config("FERNET_KEY")
#         self.cipher = Fernet(key)

#     def decrypt(self, encrypted: str) -> str:
#         decrypted = self.cipher.decrypt(encrypted.encode()).decode()
#         print("DATABASE_URL desencriptado:", decrypted)  # Línea de prueba
#         return decrypted

# # # Solo para test punto 3 desde línea de comandos
# if __name__ == "__main__":
#     encrypted_url = config("ENCRYPTED_DATABASE_URL")
#     crypto = EncryptionHelper()
#     crypto.decrypt(encrypted_url)