import os
from dotenv import load_dotenv # type: ignore

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv()  # Charge les variables d'environnement depuis .env

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'assiya@@@@2004'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Configuration email
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')