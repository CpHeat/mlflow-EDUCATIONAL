import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
import numpy as np
import plotly.graph_objects as go


def plot_confusion_matrix(y_true, y_pred, model_name, labels=None):
    """Affiche une matrice de confusion interactive avec Plotly."""
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [str(i) for i in sorted(np.unique(y_true))]

    # Pourcentages par ligne (= par vraie classe)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    # Annotations : valeur absolue + pourcentage
    text_annotations = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm[0])):
            row.append(f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)")
        text_annotations.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm, x=labels, y=labels, text=text_annotations, texttemplate="%{text}", colorscale="Blues", showscale=True
        )
    )

    fig.update_layout(
        title=f"Matrice de Confusion - {model_name}",
        xaxis_title="Prédiction",
        yaxis_title="Réalité",
        width=500,
        height=450,
    )
    fig.show()
    return cm, fig

def plot_roc_curve(y_true, y_proba, model_name):
    """Affiche la courbe ROC avec l'AUC (classification binaire uniquement)."""
    if len(np.unique(y_true)) != 2:
        print(f"ROC non disponible pour {model_name}: pas binaire")
        return None

    if y_proba is None:
        print(f"ROC non disponible pour {model_name}: pas de probabilités")
        return None

    y_score = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC = {roc_auc:.3f})", line=dict(color="blue", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC = 0.5)", line=dict(color="gray", width=1, dash="dash")
        )
    )

    fig.update_layout(
        title=f"Courbe ROC - {model_name} (AUC = {roc_auc:.3f})",
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        width=550,
        height=450,
        showlegend=True,
    )
    fig.show()
    return roc_auc, fig