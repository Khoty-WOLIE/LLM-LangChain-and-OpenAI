from dotenv import load_dotenv
import os

# Charger la clé API depuis le fichier .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("Clé API chargée avec succès !")
else:
    print("Erreur : Clé API non trouvée.")
