# Projet MLflow - Évaluation et Tuning de Modèles (Gravité des Accidents)

Ce projet a pour objectif d'entraîner, tester et optimiser des modèles d'apprentissage automatique pour prédire la gravité d'accidents de la route. L'intégralité du suivi des expérimentations, des hyperparamètres et des métriques est gérée par **MLflow**.

## Structure des Notebooks

L'analyse est découpée en 3 grandes étapes séquentielles :

### 1. `01_Benchmark_Modeles.ipynb`
* **Rôle :** Évaluation de base (baseline).
* **Action :** Entraînement simultané de plusieurs algorithmes (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) avec leurs paramètres par défaut.
* **Sortie :** Identification de l'algorithme le plus performant (ex: XGBoost) via les métriques de base dans MLflow.

### 2. `02_Tuning_Manuel_Artefacts.ipynb` (En cours)
* **Rôle :** Compréhension approfondie du modèle gagnant.
* **Action :** Test manuel de quelques hyperparamètres sur le meilleur algorithme identifié à l'étape 1. Génération de graphiques d'évaluation (ex: Matrice de confusion).
* **Sortie :** Enregistrement des artefacts visuels sur MLflow et promotion du modèle testé dans le Model Registry.

### 3. `03_Tuning_Avance_Auto.ipynb` (En cours)
* **Rôle :** Optimisation algorithmique finale.
* **Action :** Utilisation de techniques d'optimisation automatisées (GridSearchCV, Hyperopt, Optuna) pour balayer un large spectre d'hyperparamètres de façon intelligente.
* **Sortie :** Enregistrement automatique des dizaines d'essais sur MLflow pour extraire la configuration ultime et obtenir le meilleur modèle prédictif possible.

---

## Démarrage rapide

1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer le serveur MLflow local : `mlflow server --host 127.0.0.1 --port 5000`
3. Ouvrir l'interface web MLflow à l'adresse : `http://127.0.0.1:5000`
4. Exécuter les notebooks dans l'ordre (01, puis 02, puis 03).