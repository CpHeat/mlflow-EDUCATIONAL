<div align="center">
  <h1>Projet MLflow - Prédiction de la Gravité des Accidents de la Route</h1>
  <p>
    <a href="https://github.com/CpHeat/mlflow-EDUCATIONAL/stargazers">
      <img src="https://img.shields.io/github/stars/CpHeat/mlflow-EDUCATIONAL?style=for-the-badge&color=ffb300&logo=github&label=Stars" alt="GitHub Stars">
    </a>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11">
    </a>
    <a href="https://jupyter.org/">
      <img src="https://img.shields.io/badge/Jupyter-Lab-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Lab">
    </a>
    <a href="https://scikit-learn.org/">
      <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
    </a>
    <a href="https://mlflow.org/">
      <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=MLflow&logoColor=white" alt="MLflow">
    </a>
    <a href="https://www.docker.com/">
      <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
    </a>
  </p>
</div>

<div align="center">
  <p style="margin: 10px 0 20px; font-weight: 600;">
    <a href="#contexte-et-objectif">Contexte</a> •
    <a href="#source-des-donnees">Source des Données</a> •
    <a href="#equipe">Équipe</a> •
    <a href="#architecture-et-ecosysteme">Architecture</a> •
    <a href="#installation-et-lancement">Installation & Docker</a> •
    <a href="#notebooks-devaluation">Notebooks</a> •
    <a href="#reproduction-des-experiences">Reproduction</a>
  </p>
</div>

## Contexte et Objectif

L'objectif de ce projet est de **prédire la gravité d'un accident de la route** à l'aide de modèles de Machine Learning. 
Afin de résoudre le problème de perte d'historique des modèles et de centraliser le suivi des expérimentations, **MLflow** a été intégré au projet pour :
- Tracker automatiquement chaque entraînement de modèle
- Logger les hyperparamètres, les **métriques de classification** (ROC-AUC, F1-score, Precision, Recall) et les modèles sérialisés
- Enregistrer les **artefacts visuels** (matrice de confusion, courbes ROC, feature importance)
- Comparer visuellement les modèles dans l'UI MLflow
- Enregistrer le meilleur modèle dans le Model Registry

## Source des Données

Les données utilisées proviennent de la base BAAC (Bulletins d'Analyse des Accidents Corporels) mise à disposition par le Ministère de l'Intérieur :

**[Bases de données annuelles des accidents corporels (2005-2024)](https://www.data.gouv.fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2024)**


## Equipe

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/CpHeat">
          <img src="https://github.com/CpHeat.png" alt="Charles" width="64" height="64" style="border-radius: 32px;">
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/GautierGavat">
          <img src="https://github.com/GautierGavat.png" alt="Gautier Gavat" width="64" height="64" style="border-radius: 32px;">
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/Rateur">
          <img src="https://github.com/Rateur.png" alt="Valentin PERIES" width="64" height="64" style="border-radius: 32px;">
        </a>
      </td>
    </tr>
    <tr>
      <td align="center"><sub><strong>Charles</strong></sub></td>
      <td align="center"><sub><strong>Gautier Gavat</strong></sub></td>
      <td align="center"><sub><strong>Valentin PERIES</strong></sub></td>
    </tr>
  </table>
</div>

## Architecture et Écosystème

Ce projet MLOps s'insère dans un **écosystème plus global** (composé d'un pipeline ETL, d'une API FastAPI/Flask et d'un Front-end). Le dépôt actuel se concentre sur la partie **Machine Learning et Expérimentation**.

Le serveur **MLflow** s'exécute via Docker et utilise **SQLite** (`mlflow.db`) en tant que backend database pour stocker les métriques et paramètres, ainsi qu'un stockage local pour les artefacts.

```text
.
├── README.md                # Documentation principale
├── docker-compose.yml       # Configuration Docker pour le serveur MLflow
├── requirements.txt         # Dépendances Python
├── notebooks/               # Dossier contenant les notebooks d'évaluation
│   ├── 01_Benchmark_Modeles.ipynb
│   ├── 02_Tuning_Manuel_Artefacts.ipynb
│   └── 03_Tuning_Advance_Auto.ipynb
├── .env.example             # Variables d'environnement
├── data/                    # Données (non versionnées) provenant du pipeline ETL
├── mlflow/
    ├── mlartifacts/   # Stockage local des artefacts MLflow (sauvegardes des modèles, graphiques)
        ├── models/                  # Modèles sérialisés
    ├── mlflow.db                # Base de données SQLite stockant l'historique MLflow
```

*(Note : L'interface MLflow est accessible en local sur `http://localhost:5000`)*

## Installation et Lancement

Pour installer les dépendances et exécuter le serveur MLflow, choisissez votre système d'exploitation ci-dessous.
> **Note technique MLflow :** Le fichier `docker-compose.yml` configure MLflow pour utiliser SQLite en backend et monte des **volumes persistants** pour associer les dossiers `mlruns/`, `mlartifacts/` et le fichier `mlflow.db` à votre répertoire local. Cela évite toute perte d'historique ou d'artefacts à l'arrêt du conteneur.

<details>
<summary><b>🐧 Linux / macOS</b></summary>
<br>

**1. Cloner le dépôt**
```bash
git clone https://github.com/CpHeat/mlflow-EDUCATIONAL.git
cd mlflow-EDUCATIONAL
```

**2. Lancer le serveur MLflow via Docker**
```bash
docker-compose up -d
```
*Le serveur MLflow sera accessible sur [http://localhost:5000](http://localhost:5000).*

**3. Créer l'environnement virtuel avec `uv` (recommandé) ou `venv`**
```bash
# Avec uv (plus rapide)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# OU avec venv standard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><b>🪟 Windows</b></summary>
<br>

**1. Cloner le dépôt**
```powershell
git clone https://github.com/CpHeat/mlflow-EDUCATIONAL.git
cd mlflow-EDUCATIONAL
```

**2. Lancer le serveur MLflow via Docker**
```powershell
docker-compose up -d
```
*Le serveur MLflow sera accessible sur [http://localhost:5000](http://localhost:5000).*

**3. Créer l'environnement virtuel avec `uv` (recommandé) ou `venv`**
```powershell
# Avec uv (plus rapide)
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# OU avec venv standard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
</details>

## Notebooks d'Evaluation

Le workflow MLOps de ce projet est documenté pas à pas dans trois notebooks principaux :

- **`01_Benchmark_Modeles.ipynb`** :
  - Évaluation de base (baseline).
  - Entraînement simultané de plusieurs algorithmes (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) avec leurs paramètres par défaut.
  - Identification de l'algorithme le plus performant via les métriques enregistrées dans MLflow.

- **`02_Tuning_Manuel_Artefacts.ipynb`** :
  - Compréhension approfondie du modèle gagnant (XGBoost ou autre algorithme identifié à l'étape 1).
  - Réglage manuel d'hyperparamètres spécifiques.
  - Génération de graphiques d'évaluation (Matrice de confusion, courbes ROC) enregistrés en tant qu'artefacts sur MLflow.
  - Promotion du modèle validé dans le **Model Registry** de MLflow.

- **`03_Tuning_Advance_Auto.ipynb`** :
  - Automatisation de l'optimisation des hyperparamètres pour le modèle XGBoost.
  - Comparaison expérimentale entre différentes approches : GridSearchCV, Hyperopt et Optuna.
  - Centralisation des résultats d'optimisation dans MLflow pour identifier le meilleur modèle.

## Reproduction des Experiences

Pour reproduire les résultats de ce dépôt :

1. Assurez-vous d'avoir lancé le serveur MLflow via Docker et activé votre environnement virtuel.
2. Démarrez Jupyter Lab à la racine du projet :
   ```bash
   jupyter lab
   ```
3. Exécutez les notebooks depuis le dossier `notebooks/`. Commencez par le `01_Benchmark_Modeles.ipynb` en entier. Vérifiez que les modèles expérimentés apparaissent dans l'interface MLflow (http://localhost:5000).
4. Poursuivez avec le `02_Tuning_Manuel_Artefacts.ipynb` pour observer le suivi des hyperparamètres, des métriques détaillées et l'enregistrement visuel des performances.
5. Terminez par le `03_Tuning_Advance_Auto.ipynb` pour analyser et comparer les diverses stratégies d'optimisation automatique des hyperparamètres via GridSearchCV, Hyperopt et Optuna.
