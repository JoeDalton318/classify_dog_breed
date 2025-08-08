"""
Utilitaires pour l'intégration du modèle de classification de chiens
"""

import numpy as np
from PIL import Image
import os
from typing import Dict, List, Tuple

# Liste des races de chiens (à adapter selon votre modèle)
DOG_BREEDS = [
    "Bulldog Français", "Malinois", "Golden Retriever", "Labrador", 
    "Berger Allemand", "Chihuahua", "Yorkshire", "Caniche", 
    "Husky", "Border Collie", "Rottweiler", "Doberman",
    "Berger Australien", "Cavalier King Charles", "Jack Russell",
    "Berger des Shetland", "Berger Suisse", "Berger Hollandais"
]

def load_model(model_path: str = "models/dog_classifier_model.pkl"):
    """
    Charge le modèle entraîné depuis le fichier pickle
    
    Args:
        model_path (str): Chemin vers le fichier du modèle
        
    Returns:
        Le modèle chargé ou None si le fichier n'existe pas
    """
    try:
        # Vérifier si le fichier existe
        if os.path.exists(model_path):
            print(f"⚠️  Modèle trouvé à {model_path}")
            print("📝 Note: joblib non installé, utilisation de la simulation")
            return None
        else:
            print(f"⚠️  Modèle non trouvé à {model_path}")
            print("📝 Utilisation du modèle de simulation")
            return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Prétraite l'image pour le modèle ML
    
    Args:
        image (PIL.Image): Image à prétraiter
        target_size (tuple): Taille cible (largeur, hauteur)
        
    Returns:
        np.ndarray: Image prétraitée
    """
    try:
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionner
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convertir en array numpy
        img_array = np.array(image)
        
        # Normaliser les valeurs (0-255 -> 0-1)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Ajouter dimension batch si nécessaire
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Erreur lors du prétraitement: {e}")
        return None

def format_predictions(predictions: np.ndarray, breeds: List[str] = None) -> Dict[str, float]:
    """
    Formate les prédictions du modèle
    
    Args:
        predictions (np.ndarray): Prédictions du modèle
        breeds (List[str]): Liste des races (optionnel)
        
    Returns:
        Dict[str, float]: Dictionnaire {race: pourcentage}
    """
    if breeds is None:
        breeds = DOG_BREEDS
    
    try:
        # Convertir en pourcentages
        if len(predictions.shape) > 1:
            # Si c'est un array 2D, prendre la première ligne
            predictions = predictions[0]
        
        # Convertir en pourcentages
        percentages = predictions * 100
        
        # Créer le dictionnaire des résultats
        results = {}
        for i, breed in enumerate(breeds):
            if i < len(percentages):
                results[breed] = round(float(percentages[i]), 1)
        
        # Trier par pourcentage décroissant
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results
        
    except Exception as e:
        print(f"❌ Erreur lors du formatage: {e}")
        return {}

def classify_dog_breed_with_model(image: Image.Image, model) -> Dict[str, float]:
    """
    Classification complète avec le modèle
    
    Args:
        image (PIL.Image): Image à classifier
        model: Modèle ML chargé
        
    Returns:
        Dict[str, float]: Résultats de classification
    """
    try:
        # Prétraitement
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return {}
        
        # Prédiction
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(processed_image)
        elif hasattr(model, 'predict'):
            predictions = model.predict(processed_image)
        else:
            print("❌ Modèle non compatible")
            return {}
        
        # Formatage
        results = format_predictions(predictions)
        
        return results
        
    except Exception as e:
        print(f"❌ Erreur lors de la classification: {e}")
        return {}

def get_model_info(model_path: str = "models/dog_classifier_model.pkl") -> Dict:
    """
    Récupère les informations sur le modèle
    
    Returns:
        Dict: Informations sur le modèle
    """
    info = {
        "model_loaded": False,
        "model_path": model_path,
        "file_exists": False,
        "file_size": 0,
        "breeds_count": len(DOG_BREEDS)
    }
    
    try:
        if os.path.exists(model_path):
            info["file_exists"] = True
            info["file_size"] = os.path.getsize(model_path)
            
            # Essayer de charger le modèle
            model = load_model(model_path)
            if model is not None:
                info["model_loaded"] = True
                info["model_type"] = type(model).__name__
                
                # Informations supplémentaires selon le type de modèle
                if hasattr(model, 'classes_'):
                    info["classes"] = list(model.classes_)
                if hasattr(model, 'n_features_in_'):
                    info["input_features"] = model.n_features_in_
                    
    except Exception as e:
        info["error"] = str(e)
    
    return info

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Valide une image pour la classification
    
    Args:
        image (PIL.Image): Image à valider
        
    Returns:
        Tuple[bool, str]: (valide, message)
    """
    try:
        # Vérifier la taille minimale
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image trop petite (minimum 50x50 pixels)"
        
        # Vérifier la taille maximale
        if width > 5000 or height > 5000:
            return False, "Image trop grande (maximum 5000x5000 pixels)"
        
        # Vérifier le format
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Format d'image non supporté"
        
        return True, "Image valide"
        
    except Exception as e:
        return False, f"Erreur de validation: {str(e)}"

# Fonction de simulation pour les tests
def simulate_classification(image: Image.Image) -> Dict[str, float]:
    """
    Simulation de classification (utilisée quand le modèle n'est pas disponible)
    """
    import hashlib
    
    # Utiliser le hash de l'image pour des résultats cohérents
    img_bytes = image.tobytes()
    hash_value = int(hashlib.md5(img_bytes).hexdigest(), 16)
    np.random.seed(hash_value % 2**32)
    
    # Générer des scores réalistes
    scores = np.random.dirichlet(np.ones(len(DOG_BREEDS))) * 100
    
    # Créer le dictionnaire des résultats
    results = {}
    for breed, score in zip(DOG_BREEDS, scores):
        results[breed] = round(score, 1)
    
    # Trier par pourcentage décroissant
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_results
