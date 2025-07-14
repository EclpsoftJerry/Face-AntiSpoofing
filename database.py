from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from cryptography.fernet import Fernet
from decouple import config
from utils.encryption import EncryptionHelper

# Instancia del helper
crypto = EncryptionHelper()

# Cargar y descifrar la cadena
ENCRYPTED_DATABASE_URL = config("ENCRYPTED_DATABASE_URL")
DATABASE_URL = crypto.decrypt(ENCRYPTED_DATABASE_URL)
#print(DATABASE_URL)

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    raise RuntimeError(f"Error al conectar a la base de datos: {e}")