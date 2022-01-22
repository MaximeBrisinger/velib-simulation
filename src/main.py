import numpy as np

S = 5

def temps_d_attente(velos_par_station, velos_par_trajet, tx_depart_a_lheure, temps_moyen_par_trajet):
    """renvoie le temps d'attente avant le prochain changement d'état"""

    Qnn = np.sum(velos_par_station * tx_depart_a_lheure.reshape(velos_par_station.shape)) + np.sum(velos_par_trajet * temps_moyen_par_trajet)
    return np.random.exponential(scale=1/Qnn)

def nouvel_etat(velos_par_station, velos_par_trajet, tx_depart_a_lheure, temps_moyen_par_trajet, routage):
    """transforme localement les array velos par station, velo_par_trajet"""
    weights = np.concatenate([(velos_par_station * tx_depart_a_lheure).flatten(), (velos_par_trajet * temps_moyen_par_trajet).flatten()])
    transfo = np.random.choice(weights.size, p=weights / np.sum(weights))

    if transfo < S:
        #Un velo quite une station : on realise un nouveau tirage pour définir sa destination

        arrivee = np.random.choice(S, p=routage[transfo, :])
        velos_par_station[transfo] -= 1
        velos_par_trajet[transfo, arrivee] += 1
        print(f"Un velo part de {transfo} vers {arrivee}")
    else:
        depart = (transfo - S) // S
        arrivee = (transfo - S) % S
        velos_par_station[arrivee] += 1
        velos_par_trajet[depart, arrivee] -= 1
        print(f"Un velo arrive en {arrivee} depuis {depart}")


def main(velos_par_station_0, velos_par_trajet_0, temps_moyen_par_trajet, tx_depart_a_lheure, routage):

    velos_par_station = np.copy(velos_par_station_0)
    velos_par_trajet = np.copy(velos_par_trajet_0)
    n_iter = 0
    t = [0]
    
    while n_iter < 100:
        n_iter += 1
        tau = temps_d_attente(velos_par_station, velos_par_trajet, tx_depart_a_lheure, temps_moyen_par_trajet)
        t.append(t[-1] + tau)
        nouvel_etat(velos_par_station, velos_par_trajet, tx_depart_a_lheure, temps_moyen_par_trajet, routage)



if __name__ =="__main__":
    temps_moyen_par_trajet = np.load("data/temps_moyen_par_trajet.npy", allow_pickle=True)
    tx_depart_a_lheure = np.load("data/taux_depart_a_l_heure.npy", allow_pickle=True)
    routage = np.load("data/routage.npy", allow_pickle=True)
    velos_par_station_0 = np.load("data/velos_par_station_initial.npy", allow_pickle=True)
    velos_par_trajet_0 = np.load("data/velos_par_trajet_initial.npy", allow_pickle=True)
    main(velos_par_station_0, velos_par_trajet_0, temps_moyen_par_trajet, tx_depart_a_lheure, routage)
