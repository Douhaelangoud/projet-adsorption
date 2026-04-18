import smtplib
from config import Config
from getpass import getpass
import sys

def test_smtp_connection():
    print("\n" + "="*50)
    print("Test de connexion SMTP à Gmail - Debug Mode")
    print("="*50)
    
    # Affiche la configuration actuelle
    print("\nConfiguration utilisée:")
    print(f"Server: {Config.MAIL_SERVER}:{Config.MAIL_PORT}")
    print(f"TLS: {Config.MAIL_USE_TLS}, SSL: {Config.MAIL_USE_SSL}")
    print(f"Username: {Config.MAIL_USERNAME}")
    
    # Demande les credentials si manquants
    email = Config.MAIL_USERNAME or input("Email Gmail: ")
    password = Config.MAIL_PASSWORD or getpass("Mot de passe d'application: ")
    
    try:
        print(f"\nTentative de connexion à {Config.MAIL_SERVER}...")
        with smtplib.SMTP(Config.MAIL_SERVER, Config.MAIL_PORT, timeout=15) as server:
            server.set_debuglevel(2)  # Mode debug maximal
            
            print(">>> STARTTLS")
            server.starttls()
            
            print(">>> LOGIN")
            server.login(email, password)
            
            print("\n" + "="*50)
            print("✅ CONNEXION RÉUSSIE!")
            print("="*50)
            return True
            
    except smtplib.SMTPAuthenticationError as e:
        print("\n" + "="*50)
        print(f"❌ ERREUR: {e}")
        print("Solutions possibles:")
        print("- Activez la validation en 2 étapes")
        print("- Créez un mot de passe d'application")
        print("- Vérifiez que vous utilisez le bon port (587 pour TLS)")
        print("="*50)
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ ERREUR INATTENDUE: {type(e).__name__}: {e}")
        print("="*50)
    
    return False

if __name__ == "__main__":
    print("Démarrage du test SMTP...")
    test_smtp_connection()
    input("Appuyez sur Entrée pour quitter...")  # Garde la fenêtre ouverte