from sqlalchemy.orm import Session
from database import SessionLocal
from models.user import User
from security import hash_password

def create_user():
    db: Session = SessionLocal()

    username = "admin"
    plain_password = "admin123"
    full_name = "Administrador"
    email = "admin@eclipsoft.com"
    role = "admin"

    # Verifica si ya existe
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        print(f"El usuario '{username}' ya existe.")
        return

    hashed_pw = hash_password(plain_password)
    new_user = User(
        username=username,
        hashed_password=hashed_pw,
        full_name=full_name,
        email=email,
        role=role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    print(f"Usuario '{username}' creado con éxito.")

if __name__ == "__main__":
    create_user()
