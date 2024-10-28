from ipywidgets import interact, FloatSlider
import numpy as np
import matplotlib.pyplot as plt
def plot_cost_function(compute_cost):

    """
    Affiche un graphique interactif du coût en fonction de la variable de poids `w` pour un modèle de régression linéaire.

    Cette fonction génère des données fictives pour illustrer comment le coût varie avec la valeur de `w`. Un widget interactif permet à l'utilisateur de sélectionner différentes valeurs de `w` et de visualiser comment cela affecte le coût.

    Args:
      compute_cost (function): Fonction pour calculer le coût de régression linéaire. Cette fonction doit avoir la signature suivante :
        compute_cost(x, y, w, b), où :
        - x (ndarray): Données d'entrée.
        - y (ndarray): Valeurs cibles.
        - w (float): Valeur du paramètre de poids.
        - b (float): Valeur du paramètre de biais.
        Retourne :
        - (float): Coût pour les paramètres fournis.

    Fonctionnement interne:
      - Génère des données d'exemple pour `x` et `y` en ajoutant du bruit à une relation linéaire.
      - Définie une gamme de valeurs pour `w` et calcule le coût pour chaque valeur avec un biais fixe.
      - Crée un graphique affichant le coût en fonction de `w` et ajoute un point de dispersion pour la valeur actuelle de `w`.
      - Utilise un widget interactif pour permettre à l'utilisateur de modifier la valeur de `w` et de voir en temps réel l'effet sur le coût.

    Exemple d'utilisation:
      plot_cost_function(compute_cost)
    """
    # Générer des données fictives pour l'exemple
    np.random.seed(0)
    x = 2 * np.random.rand(100)
    y = 100 + 200 * x + np.random.randn(100)  # y = 100 + 200x + bruit
    
    # Fonction pour afficher le graphique avec le coût en fonction de w
    def plot_cost(w):
        b_fixed = 100  # Fixer une valeur de b
        cost = compute_cost(x, y, w, b_fixed)
        
        plt.figure(figsize=(10, 6))
        plt.plot(w_range, cost_w, 'b-', linewidth=2, label='Coût en fonction de w')
        
        # Afficher un point de dispersion pour la valeur actuelle de w
        plt.scatter(w, compute_cost(x, y, w, b_fixed), color='red', zorder=5, label=f'W = {w:.2f}')
        
        plt.title(f'Coût en fonction de w (b = {b_fixed})\nValeur actuelle de w: {w:.2f} - Coût: {cost:.2f}')
        plt.xlabel('w')
        plt.ylabel('Coût')
        plt.ylim(0, max(cost_w) * 1.1)  # Ajuster l'échelle du y
        plt.legend()
        plt.show()
    
    # Définir une gamme de valeurs pour w
    w_range = np.linspace(0, 400, 100)
    
    # Calculer le coût pour chaque valeur de w avec un b fixe
    b_fixed = 100  # Fixer une valeur de b
    cost_w = np.zeros_like(w_range)
    for i, w in enumerate(w_range):
        cost_w[i] = compute_cost(x, y, w, b_fixed)
    
    # Utiliser interact pour créer le widget
    interact(plot_cost, w=FloatSlider(value=b_fixed, min=0, max=400, step=1, description='w'));


def plot_cost_with_gradient(x_train, y_train, compute_cost, compute_gradient):
    """
    Affiche le graphique du coût en fonction de w avec des lignes indiquant le gradient
    pour certaines valeurs de w.
    
    Args:
      x_train (ndarray): Données d'entrée
      y_train (ndarray): Valeurs cibles
      compute_cost (function): Fonction pour calculer le coût
      compute_gradient (function): Fonction pour calculer le gradient
    """

    # Calcul et affichage du coût en fonction de w
    w_array = np.linspace(0, 400, 50)
    cost = np.array([compute_cost(x_train, y_train, w, 100) for w in w_array])
    plt.plot(w_array, cost, linewidth=1)
    plt.title("Coût en fonction de w avec gradient; b fixé à 100")
    plt.ylabel('Coût')
    plt.xlabel('w')

    # Ajouter les lignes du gradient pour certaines valeurs de w
    for w_val in [100, 200, 300]:
        dj_dw, _ = compute_gradient(x_train, y_train, w_val, 100)
        cost_at_w = compute_cost(x_train, y_train, w_val, 100)
        x = np.linspace(w_val - 30, w_val + 30, 50)
        y = dj_dw * (x - w_val) + cost_at_w
        plt.scatter(w_val, cost_at_w, color='b', s=50)
        plt.plot(x, y, '--', c='r', zorder=10, linewidth=1)
        xoff = 30 if w_val == 200 else 10
        plt.annotate(
            r"$\frac{\partial J}{\partial w}$ = %.2f" % dj_dw, fontsize=14,
            xy=(w_val, cost_at_w), xycoords='data',
            xytext=(xoff, 10), textcoords='offset points',
            
            horizontalalignment='left', verticalalignment='top'
        )

    # Affichage du graphique
    plt.tight_layout()
    plt.show()
