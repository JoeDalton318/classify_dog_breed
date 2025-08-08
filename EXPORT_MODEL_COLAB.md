# 📤 Guide d'export du modèle depuis Google Colab

Ce guide vous explique comment exporter votre modèle de classification de chiens depuis Google Colab pour l'intégrer dans l'application Streamlit.

## 🎯 **Étapes d'export depuis Google Colab**

### 1. **Préparer le modèle dans Colab**

```python
# Dans votre notebook Google Colab
import joblib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # ou votre modèle
from sklearn.preprocessing import StandardScaler
import os

# Supposons que votre modèle est déjà entraîné et s'appelle 'model'
# Si ce n'est pas le cas, adaptez selon votre code

# Vérifier que le modèle existe
print(f"Type du modèle: {type(model)}")
print(f"Méthodes disponibles: {[method for method in dir(model) if not method.startswith('_')]}")
```

### 2. **Sauvegarder le modèle**

#### **Option A : Avec joblib (recommandé)**
```python
# Sauvegarder le modèle
model_filename = 'dog_classifier_model.pkl'
joblib.dump(model, model_filename)

# Vérifier la sauvegarde
print(f"Modèle sauvegardé: {model_filename}")
print(f"Taille du fichier: {os.path.getsize(model_filename) / 1024 / 1024:.2f} MB")

# Tester le chargement
loaded_model = joblib.load(model_filename)
print("✅ Modèle chargé avec succès")
```

#### **Option B : Avec pickle**
```python
# Sauvegarder avec pickle
model_filename = 'dog_classifier_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"Modèle sauvegardé: {model_filename}")
```

### 3. **Sauvegarder les métadonnées (optionnel mais recommandé)**

```python
# Sauvegarder les informations sur les classes/races
model_metadata = {
    'classes': model.classes_.tolist() if hasattr(model, 'classes_') else [],
    'model_type': type(model).__name__,
    'input_shape': getattr(model, 'n_features_in_', None),
    'training_date': '2024-01-15',  # Date d'entraînement
    'accuracy': 0.95,  # Précision du modèle
    'description': 'Modèle de classification de races de chiens'
}

# Sauvegarder les métadonnées
import json
with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("Métadonnées sauvegardées: model_metadata.json")
```

### 4. **Télécharger les fichiers**

```python
# Télécharger le modèle
from google.colab import files

# Télécharger le fichier du modèle
files.download('dog_classifier_model.pkl')

# Télécharger les métadonnées (si créées)
files.download('model_metadata.json')
```

## 🔧 **Intégration dans l'application**

### 1. **Placer le fichier dans le projet**

```bash
# Dans votre projet Front-end_Dogs
# Copiez le fichier téléchargé dans le dossier models/
cp ~/Downloads/dog_classifier_model.pkl models/
```

### 2. **Adapter le code si nécessaire**

Si votre modèle a des spécificités, modifiez `utils/model_utils.py` :

```python
# Exemple d'adaptation pour un modèle spécifique
def preprocess_image_for_your_model(image: Image.Image) -> np.ndarray:
    """
    Prétraitement spécifique à votre modèle
    """
    # Redimensionner selon les attentes de votre modèle
    target_size = (224, 224)  # Ajustez selon votre modèle
    
    # Convertir en RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir en array et normaliser
    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0
    
    # Reshape si nécessaire (ajustez selon votre modèle)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
```

### 3. **Tester l'intégration**

```python
# Test simple dans Python
from utils.model_utils import load_model, classify_dog_breed_with_model
from PIL import Image

# Charger le modèle
model = load_model('models/dog_classifier_model.pkl')

# Tester avec une image
test_image = Image.open('test_dog.jpg')
results = classify_dog_breed_with_model(test_image, model)
print(results)
```

## 🚨 **Problèmes courants et solutions**

### **Erreur : "Module not found"**
```bash
# Installer les dépendances manquantes
pip install scikit-learn joblib numpy pillow
```

### **Erreur : "Model incompatible"**
- Vérifiez que votre modèle a les méthodes `predict()` ou `predict_proba()`
- Adaptez la fonction `classify_dog_breed_with_model()` selon votre modèle

### **Erreur : "Input shape mismatch"**
- Vérifiez la taille d'entrée attendue par votre modèle
- Ajustez la fonction `preprocess_image()` en conséquence

### **Erreur : "Classes mismatch"**
- Vérifiez que la liste `DOG_BREEDS` dans `model_utils.py` correspond aux classes de votre modèle

## 📋 **Checklist d'export**

- [ ] Modèle entraîné et testé dans Colab
- [ ] Sauvegarde avec joblib ou pickle
- [ ] Téléchargement du fichier .pkl
- [ ] Placement dans le dossier `models/`
- [ ] Test de chargement du modèle
- [ ] Adaptation du prétraitement si nécessaire
- [ ] Test de classification avec une image réelle

## 🔍 **Vérification finale**

Après l'intégration, lancez l'application :

```bash
streamlit run app.py
```

Dans la sidebar, vous devriez voir :
- ✅ "Modèle chargé avec succès"
- 📊 Informations sur la taille du fichier
- 🤖 Type de modèle détecté

## 📞 **Support**

Si vous rencontrez des problèmes :

1. **Vérifiez les logs** dans la console Streamlit
2. **Testez le modèle** directement avec Python
3. **Adaptez le prétraitement** selon les spécificités de votre modèle
4. **Consultez la documentation** de votre framework ML

---

**💡 Conseil :** Testez toujours votre modèle avec quelques images de test avant de l'intégrer dans l'application !
