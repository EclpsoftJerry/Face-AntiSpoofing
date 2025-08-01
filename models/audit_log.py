from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from database import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    username = Column(String, nullable=True)  # Usuario que hace la solicitud (si aplica)
    endpoint = Column(String, nullable=False)  # Ruta del endpoint, ej: /predict
    method = Column(String, nullable=False)    # Método HTTP: GET, POST, etc.    
    request_data = Column(Text)   # Información enviada en la solicitud (puede ser un resumen o JSON)
    response_data = Column(Text)  # Información devuelta por la API
    status_code = Column(Integer) # Código HTTP (ej. 200, 400, 500)    
    timestamp = Column(DateTime, default=datetime.utcnow)  # Fecha y hora de la solicitud
