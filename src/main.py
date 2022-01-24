import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
plt.rc('legend', fontsize=10)
plt.rc('figure', titlesize=15)
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15)
S = 5

def temps_d_attente(velos_par_station, velos_par_trajet, lambd, mu):
    """ Tirage du temps d'attente avant le prochain changement d'état

    Args:
        velos_par_station (array numpy de dimension 1): vecteur ligne représentant le nombre de vélos par station
        velos_par_trajet (array numpy de dimension 2): matrice représentant le nombre de vélos par station
        lambd (array numpy de dimension 1): intensité des lois exponentielles de départ des stations
        mu (array numpy de dimension 2): intensité des lois exponentielles de réalisation des trajet 

    Returns:
        tau (float): temps d'attente avant prochain changement d'état
    """

    Qnn = np.sum(np.where(velos_par_station > 0, 1, 0) * lambd.reshape(velos_par_station.shape)) + np.sum(velos_par_trajet * mu)
    return np.random.exponential(scale=1/Qnn)

def nouvel_etat(velos_par_station, velos_par_trajet, lambd, mu, routage, verbose):
    """ Tire le prochain état du système

    Args:
        velos_par_station (array numpy de dimension 1): vecteur ligne représentant le nombre de vélos par station
        velos_par_trajet (array numpy de dimension 2): matrice représentant le nombre de vélos par station
        lambd (array numpy de dimension 1): intensité des lois exponentielles de départ des stations
        mu (array numpy de dimension 2): intensité des lois exponentielles de réalisation des trajet 
        routage (array numpy de dimension 2): matrice de routage entre les stations
        verbose (bool): Indique si les changements d'états doivent être affichés. Defaults to True.

    Returns:
        ()

    Modification locale de velos_par_station et de velos_par_trajet
    """
    poids = np.concatenate([(np.where(velos_par_station > 0, 1, 0) * lambd.reshape(velos_par_station.shape)).flatten(), (velos_par_trajet * mu).flatten()])
    transfo = np.random.choice(poids.size, p=poids/np.sum(poids))

    if transfo < S:
        #Un velo quite une station : on realise un nouveau tirage pour définir sa destination
        arrivee = np.random.choice(S, p=routage[transfo, :])
        velos_par_station[transfo] -= 1
        velos_par_trajet[transfo, arrivee] += 1
        if verbose:
            print(f"Un velo part de {transfo} vers {arrivee}")

    else:
        depart = (transfo - S) // S
        arrivee = (transfo - S) % S
        velos_par_station[arrivee] += 1
        velos_par_trajet[depart, arrivee] -= 1
        if verbose:
            print(f"Un velo arrive en {arrivee} depuis {depart}")

def main(velos_par_station_0, velos_par_trajet_0, lambd, mu, routage, verbose=True, estim_remplissage=False, itermax=100):
    """ Simulation du système

    Args:
        velos_par_station_0 (array numpy de dimension 1): vecteur ligne représentant le nombre de vélos par station initial
        velos_par_trajet_0 (array numpy de dimension 2): matrice représentant le nombre de vélos par station initial
        lambd (array numpy de dimension 1): intensité des lois exponentielles de départ des stations
        mu (array numpy de dimension 2): intensité des lois exponentielles de réalisation des trajet 
        routage (array numpy de dimension 2): matrice de routage entre les stations
        verbose (bool, optional): Indique si les changements d'états doivent être affichés. Defaults to True.
        estim_remplissage (bool, optional): Indique si l'on calcule la matrice des remplissages cumulés. Defaults to False.
        itermax (int, optional): nombr d'itérations maximal. Defaults to 100.

    Returns:
        Si estim_remplissage :
            t (liste) : liste des dates de changements d'états
            remplissages (array numpy de dimension 2) : stocke la somme des remplissages de stations pondérées par le temps pour chaque station
    """    
    velos_par_station = np.copy(velos_par_station_0)
    velos_par_trajet = np.copy(velos_par_trajet_0)
    n_iter = 0
    if estim_remplissage:
        t = [0]
        remplissages = np.zeros((itermax+1, S))
    
    for n_iter in tqdm(range(itermax)):
        tau = temps_d_attente(velos_par_station, velos_par_trajet, lambd, mu)
        
        if estim_remplissage:
            t.append(t[-1] + tau)
            remplissages[n_iter+1, :] = remplissages[n_iter, :] + tau * velos_par_station.reshape(1, S)
    
        nouvel_etat(velos_par_station, velos_par_trajet, lambd, mu, routage, verbose)

    if estim_remplissage:
        return t, remplissages

    else:
        visualisation(velos_par_station, velos_par_trajet)

def visualisation(velos_par_station, velos_par_trajet):
    G = nx.complete_graph(5)
    options = {
    "node_color": velos_par_station,
    "edge_color": [velos_par_trajet[i,j] for (i,j) in G.edges],
    "width": 4,
    "node_cmap": plt.cm.Wistia,
    "edge_cmap": plt.cm.Wistia,
    "with_labels": True,
    }
    nx.draw(G, **options)
    plt.show()

def proba_stationnaire_1_velo(routage, lambd, mu):
    """ Calcule la probabilité stationnaire dans le cas à un seul vélo

    Args:
        routage (array numpy de dimension 2): matrice de routage entre les stations
        lambd (array numpy de dimension 2): intensités des processus de Poisson de départ des stations
        mu (array numpy de dimension 2): intensités des lois exponentielles de réalisation des temps de trajet  

    Returns:
        pi (array numpy de dimension 1): probabilite stationnaire du systeme à un seul velo
    """

    A = np.zeros((S**2 + S, S**2 + S))

    for i in range(S):
        A[i, i] = - lambd[i]
        for j in range(S):
            if  j != i:
                A[i, S + i * S + j] = routage[i, j] * lambd[i]

    for traj in range(S, S**2 + S):
        depart = (traj - S) // S
        arrivee = (traj - S) % S
        if depart != arrivee:
            A[traj, traj] = - mu[depart, arrivee]
            A[traj, arrivee] = mu[depart, arrivee]

    # Jusqu'alors, la matrice contient encore les trajtes impossibles P_ii
    # On force leur probabilité à etre nulle
    B = np.zeros((S ** 2 + S, S))
    for i in range(S):
        B[S * (i+1) + i, i] = 1 # S + i * S + i

    # On ajoute une contrainte de normalisation
    C = np.ones((S ** 2 + S, 1))
    M = np.concatenate([A, B, C], axis=1)

    second_membre = np.zeros(M.shape[1])
    second_membre[-1] = 1

    pi = np.linalg.lstsq(M.T, second_membre)
    return pi[0]


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--initial", help="conditions initiales (un seul velo ou donnees de l'énoncé",
                        choices=["1_velo", "donnees"], default="donnees")
    args = parser.parse_args()
    conditions_initiales = args.initial

    temps_moyen_par_trajet = np.load("data/temps_moyen_par_trajet.npy", allow_pickle=True)
    tx_depart_a_lheure = np.load("data/taux_depart_a_l_heure.npy", allow_pickle=True)
    routage = np.load("data/routage.npy", allow_pickle=True)
    # On choisit de prendre la minute comme unite de temps
    lambd = tx_depart_a_lheure / 60
    mu = np.nan_to_num(1 / temps_moyen_par_trajet, nan=0.0)

    if conditions_initiales == "donnees":
        velos_par_station_0 = np.load("data/velos_par_station_initial.npy", allow_pickle=True)
        velos_par_trajet_0 = np.load("data/velos_par_trajet_initial.npy", allow_pickle=True)
        verbose = False
        estim_remplissage = True
        itermax=100000

        t, remplissages = main(velos_par_station_0, velos_par_trajet_0, lambd, mu, routage, verbose=verbose, estim_remplissage=estim_remplissage, itermax=itermax)

        # remplissage moyen des stations
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        height = remplissages[-1, :] / t[-1]     
        bars = ["Station "+ str(i) for i in range(S)]
        x_pos = np.arange(len(bars))
        ax1.bar(x_pos, height)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bars)
        ax1.set_ylabel("Remplissage moyen de chaque station")
        ax1.set_title(r"Remplissage moyen des stations après " + str(round(t[-1] / (24 * 60),1)) + " jours")
        ax1.grid(True)
   
        # proba de vacuite des stations
        temps_vacuite = np.array([t[i] - t[i-1] for i in range(1, len(t))]).reshape(1, -1) @ np.where(remplissages[1:] - remplissages[:-1]  == 0, 1, 0)
        proba_vacuite = temps_vacuite / t[-1]
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
        height = proba_vacuite.flatten()
        bars = ["Station "+ str(i) for i in range(S)]
        x_pos = np.arange(len(bars))
        ax3.bar(x_pos, height)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bars)
        ax3.set_ylabel("Probabilité de vacuité de l'état")
        ax3.set_title(r"Probablilités de vacuité des stations obtenues par simulation après " + str(round(t[-1] / (24 * 60), 1)) + " jours")
        ax3.grid(True)
        plt.show()

    elif conditions_initiales == "1_velo":
        # Conditions initiales :
        velos_par_station_0 = np.zeros(S)
        velos_par_trajet_0 = np.zeros((S,S))
        velos_par_station_0[0] = 1
        estim_remplissage = True
        pi = proba_stationnaire_1_velo(routage, lambd, mu)
        verbose = False
        itermax=10000#00


        #Probabilite stationnaire
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        height = np.zeros(S+1)
        height[:S] = pi[:S]
        height[-1] = np.sum(pi[S:])
        bars = ["Station "+ str(i) for i in range(S)] + ["En trajet"]
        x_pos = np.arange(len(bars))
        ax1.bar(x_pos, height)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bars)
        ax1.set_ylabel("Probabilité de l'état")
        ax1.set_title(r"Probablilité stationnaire $\pi$")
        ax1.grid(True)

        # Probabilite de vacuite
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
        height = np.zeros(S)
        height[:S] = 1 - pi[:S]
        bars = ["Station "+ str(i) for i in range(S)]
        x_pos = np.arange(len(bars))
        ax3.bar(x_pos, height)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bars)
        ax3.set_ylabel("Probabilité de vacuité de l'état")
        ax3.set_title(r"Probablilités de vacuité des stations $1 - \pi$")
        ax3.grid(True)

        # Etude de la vitesse de convergence
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))
        for i in range(10):
            t, remplissages = main(velos_par_station_0, velos_par_trajet_0, lambd, mu, routage, verbose=verbose, estim_remplissage=estim_remplissage, itermax=itermax)
            x = np.log10(np.array(t[1:]))
            y = np.log10(np.array([np.linalg.norm(pi[:S] - remplissages[i] / t[i]) for i in range(1, len(t))]))
            ax2.plot(x,y, label=f"simulation {i}")
        reg = LinearRegression(fit_intercept=True)
        reg.fit(x.reshape(-1, 1), y)
        ax2.plot(x[[0, -1]], reg.predict(x[[0, -1]].reshape(-1, 1)), linewidth=2, color = "black", label=r"approximation linéaire : $||\pi -\hat{\pi}||_2 = T^{" + str(round(reg.coef_[0], 2)) + "} + " + str(round(reg.intercept_, 2)) + "$")
        ax2.set_ylabel(r"$\log_{10}(||\pi -\hat{\pi}||_2)$")
        ax2.set_xlabel(r"$\log_{10}(T)$")
        ax2.grid(True)
        ax2.set_title(r"Evolution de l'écart entre $\pi$ et $\hat{\pi}$ en échelle logarithmique")
        ax2.legend()

        # Deuxieme calcul dec la proba de vacuite (pour corroborer la méthode de calcul) 
        t, remplissages = main(velos_par_station_0, velos_par_trajet_0, lambd, mu, routage, verbose=verbose, estim_remplissage=estim_remplissage, itermax=itermax)        
        temps_vacuite = np.array([t[i] - t[i-1] for i in range(1, len(t))]).reshape(1, -1) @ np.where(remplissages[1:] - remplissages[:-1]  == 0, 1, 0)
        proba_vacuite = temps_vacuite / t[-1]
        fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5))
        height = proba_vacuite.flatten()
        bars = ["Station "+ str(i) for i in range(S)]
        x_pos = np.arange(len(bars))
        ax4.bar(x_pos, height)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(bars)
        ax4.set_ylabel("Probabilité de vacuité de l'état")
        ax4.set_title(r"Probablilités de vacuité des stations obtenues par simulation après " + str(round(t[-1] / (24 * 60),1)) + " jours")
        ax4.grid(True)



        plt.show()