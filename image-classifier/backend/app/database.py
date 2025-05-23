from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# DATABASE_URL will be provided by Render or can be set locally e.g. via a .env file
DATABASE_URL_ENV = os.getenv("DATABASE_URL")

# Define these for potential logging, but DATABASE_URL_ENV will override the connection if set
_POSTGRES_USER_default = "postgres"
_POSTGRES_PASSWORD_default = "525400"
_POSTGRES_SERVER_default = "localhost"
_POSTGRES_PORT_default = "5432"
_POSTGRES_DB_default = "image_classifier_db"

if DATABASE_URL_ENV:
    DATABASE_URL = DATABASE_URL_ENV
    # For logging purposes, try to get POSTGRES_DB if set, otherwise use a generic placeholder
    LOG_DB_NAME = os.getenv("POSTGRES_DB", "the configured database")
else:
    # Fallback to constructing from individual components if DATABASE_URL is not set
    POSTGRES_USER = os.getenv("POSTGRES_USER", _POSTGRES_USER_default) 
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", _POSTGRES_PASSWORD_default) 
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", _POSTGRES_SERVER_default)
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", _POSTGRES_PORT_default)
    POSTGRES_DB = os.getenv("POSTGRES_DB", _POSTGRES_DB_default)
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    LOG_DB_NAME = POSTGRES_DB

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create all tables in the database
def create_db_and_tables():
    try:
        conn = engine.connect()
        conn.close()
        print(f"Successfully connected to {LOG_DB_NAME} using connection: {DATABASE_URL.split('@')[-1]}") # Avoid logging password
    except Exception as e:
        print(f"Database ({LOG_DB_NAME} at {DATABASE_URL.split('@')[-1]}) does not seem to exist or is not accessible: {e}")
        print("Please ensure the database exists and the connection string is correct.")
        print(f"Attempted connection string (sensitive parts might be obscured by Render): {DATABASE_URL_ENV if DATABASE_URL_ENV else 'Constructed from components'}")

    print("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        print(f"Please ensure your PostgreSQL server is running and the database specified in {DATABASE_URL.split('@')[-1]} exists.") 