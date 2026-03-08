from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib

# ==========================================
# ⚙️ SQL SERVER CONNECTION CONFIGURATION
# ==========================================
SERVER = "SAM"  # Aapka apna Windows wala SQL Server
DATABASE = "DeepGuard_Enterprise"
DRIVER = "ODBC Driver 17 for SQL Server" 

# FIX: TrustServerCertificate=yes add kiya hai taake connection block na ho
params = urllib.parse.quote_plus(
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
    f"TrustServerCertificate=yes;"
)

SQLALCHEMY_DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={params}"

# Engine aur Session tayyar karna
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Dependency (Ye function routes mein use hoga)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()