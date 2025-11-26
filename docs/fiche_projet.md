# 📋 Fiche Projet : Détection de Fraude Transactionnelle

## 1. Contexte Métier
Les institutions financières subissent des pertes massives dues aux transactions frauduleuses. Au-delà de la perte sèche (le montant volé), la fraude érode la confiance client.
Le défi majeur est le **déséquilibre** : les fraudes représentent moins de 0.2% des transactions, ce qui rend les règles statiques inefficaces.

## 2. Objectifs Stratégiques
* **Minimiser le risque financier :** Détecter le maximum de fraudes réelles (Maximiser le **Recall**).
* **Préserver l'expérience client :** Éviter de bloquer à tort des transactions légitimes (Maximiser la **Précision**).
* **Industrialisation :** Fournir un outil d'aide à la décision pour les équipes d'audit.

## 3. Indicateurs de Performance (KPIs)
Dans ce contexte, l'Accuracy (taux de réussite global) est un indicateur trompeur (car dire "tout est légitime" donne 99.8% de réussite mais rate toutes les fraudes).

Nous piloterons la performance via :
1.  **Recall (Rappel) :** Priorité absolue. Combien de fraudes avons-nous attrapées sur le total existant ?
2.  **F1-Score :** La moyenne harmonique entre précision et rappel, pour assurer un équilibre.
3.  **Matrice de Coûts :** Estimation de l'impact financier (Coût d'un Faux Négatif vs Faux Positif).

## 4. Données
* **Source :** Dataset Kaggle "Credit Card Fraud Detection".
* **Volumétrie :** ~285 000 transactions.
* **Particularité :** Données anonymisées (PCA) pour confidentialité (V1, V2... V28) + Time + Amount.