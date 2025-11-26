import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, precision_score
import os

# 1. Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'creditcard.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'fraud_model.pkl')

print("⚙️ Démarrage de l'entraînement...")

# 2. Chargement
try:
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    y = df['Class']
except FileNotFoundError:
    print(f"❌ Erreur : Impossible de trouver le fichier {DATA_PATH}")
    exit()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. SMOTE (Uniquement sur le Train)
print("🔄 Application du SMOTE (Patience...)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 5. Entraînement (Le Vainqueur : Logistic Regression)
print("🧠 Entraînement du modèle (Logistic Regression)...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_smote, y_train_smote)

# 6. Validation rapide
y_pred = model.predict(X_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("-" * 30)
print(f"🏆 RÉSULTATS FINAUX :")
print(f"✅ Recall (Fraudes détectées) : {recall:.2%}")
print(f"⚠️ Precision (Fausses alertes) : {precision:.2%}")
print("-" * 30)

# 7. Sauvegarde du modèle
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Modèle sauvegardé dans : {MODEL_PATH}")