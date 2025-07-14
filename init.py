from database import engine, Base
import models

def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✔ Tablas creadas en PostgreSQL")

if __name__ == "__main__":
    create_tables()