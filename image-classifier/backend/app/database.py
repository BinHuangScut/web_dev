from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Replace with your actual PostgreSQL connection string
# DATABASE_URL = "postgresql://user:password@host:port/database"
# Example for local PostgreSQL:
# postgresql://your_postgres_user:your_postgres_password@localhost:5432/your_database_name

# It's better to use environment variables for sensitive data
# POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres") # Default to 'postgres' user
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password") # Replace with your actual default or use env
# POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
# POSTGRES_DB = os.getenv("POSTGRES_DB", "image_classifier_db") # Choose a DB name

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "525400"
POSTGRES_SERVER = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "image_classifier_db"

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

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
# This should be called once when the application starts (e.g., in main.py or a startup script)
def create_db_and_tables():
    try:
        # Try to connect to the database to see if it exists
        # A more robust way for DB creation is to handle it outside the app or use Alembic for migrations
        conn = engine.connect()
        conn.close()
        print(f"Successfully connected to the database: {POSTGRES_DB}")
    except Exception as e:
        print(f"Database {POSTGRES_DB} does not seem to exist or is not accessible: {e}")
        print("Please ensure the database exists and the connection string is correct.")
        print(f"Attempted connection string: {DATABASE_URL}")
        # If you are running locally and have permissions, you might try to create it.
        # However, this is generally not recommended for application code.
        # For now, we will just proceed and let Base.metadata.create_all handle table creation.
        # It will fail if the database itself doesn't exist.

    print("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully (if they didn't exist).")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        print("Please ensure your PostgreSQL server is running and the database specified in DATABASE_URL exists.") 