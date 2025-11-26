import pandas as pd

# 1. Charger le dataset complet
df = pd.read_csv("data/creditcard_full.csv")

# 2. Séparer fraude / legit pour conserver le ratio
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# Taille cible finale
TARGET_SIZE = 60000   # tu peux descendre à 50k si besoin

# 3. Rééchantillonnage équilibré FRAUDE + LEGITIME
# Garder toutes les fraudes disponibles
fraud_count = len(fraud)

# Le reste sera rempli avec du non-fraude
legit_needed = TARGET_SIZE - fraud_count

legit_sampled = legit.sample(n=legit_needed, random_state=42)
df_small = pd.concat([fraud, legit_sampled]).sample(frac=1, random_state=42)

# 4. Sauvegarde
df_small.to_csv("data/creditcard.csv", index=False)

print("\n✅ SAMPLE CRÉÉ AVEC SUCCÈS")
print("📄 Nombre total de lignes :", df_small.shape[0])
print("⚠️ Fraudes conservées      :", fraud_count)
print("🔢 Dataset prêt pour GitHub")