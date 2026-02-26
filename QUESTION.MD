# Nos questions

## Maîtrise de l'outil MLflow

1. Quelle est la différence exacte entre un Model, un Run et le Model Registry dans l'interface MLflow ?
2. Quelle est la différence entre l'utilisation de l'autolog et le log manuel des paramètres et des métriques ?
3. Avez-vous trouvé des visualisations intéressantes autres que des bar charts dans MLflow ?
4. Quels artefacts spécifiques avez-vous gardés pour chaque run et pourquoi ?

## Évaluation et Choix du Modèle

5. Comment avez-vous procédé pour choisir le meilleur modèle parmi toutes vos expérimentations ?
6. Quelles métriques avez-vous surveillées pour décider du meilleur modèle ?
7. Pourquoi avoir choisi une métrique *weighted* plutôt que non *weighted* ?
8. Comment départager deux modèles équivalents sur votre métrique principale ?
9. Comment gérez-vous le risque de fuite de données (Data Leakage) lors de vos étapes de preprocessing ?

## Optimisation et Reproductibilité

10. Pourquoi avoir choisi de tester ces hyperparamètres en particulier et à quoi correspondent-ils ?
11. Quelle est la différence entre une recherche exhaustive (Grid Search) et une recherche bayésienne ?
12. Avez-vous loggé le ou les datasets et comment en assurez-vous la reproductibilité stricte ?

## Architecture et Déploiement Avancé

13. Comment avez-vous déployé MLflow ?
14. Qu'est-ce que le format `mlflow.pyfunc` et à quoi sert-il pour le déploiement de modèles personnalisés ?
15. Si vous lancez 500 expérimentations en parallèle sur un cluster distribué, comment MLflow gère-t-il les accès concurrents ?
