from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from models.user import User
from database import SessionLocal
from decouple import config
from logger_config import logger

# Configuración JWT
SECRET_KEY = config("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config("ACCESS_TOKEN_EXPIRE_MINUTES", cast=int, default=60)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Esquema de autenticación (token bearer)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Obtener conexión a BD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Verificar contraseña
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Hashear nueva contraseña
def hash_password(password: str):
    return pwd_context.hash(password)

# Autenticación de usuario real desde DB
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logger.warning("Intento de login fallido: usuario '%s' no encontrado", username)
        return None
    if not verify_password(password, user.hashed_password):
        logger.warning("Intento de login fallido: contraseña incorrecta para usuario '%s'", username)
        return None
    logger.info("Usuario '%s' autenticado exitosamente", username)
    return user

# Crear JWT
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("Token JWT creado para usuario: %s", data.get("sub"))
    return token


# Obtener usuario autenticado desde token
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar el token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token recibido sin campo 'sub'")
            raise credentials_exception
    except JWTError as e:
        logger.error("Error al decodificar el token JWT: %s", str(e))
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        logger.warning("Usuario '%s' del token no existe en la base de datos", username)
        raise credentials_exception
    
    logger.info("Token válido para usuario '%s'", username)
    return user
