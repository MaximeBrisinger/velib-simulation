import numpy as np

if __name__ == "__main__":

    temps_moyen_par_traj = np.array([[0, 3, 5, 7, 7], 
                                     [2, 0, 2, 5, 5],
                                     [4, 2, 0, 3, 3], 
                                     [8, 6, 4, 0, 2],
                                     [7, 7, 5, 2, 0]])

    tx_depart_a_lheure =  np.array([2.8, 3.7, 5.5, 3.5, 4.6])

    routage = np.array([[0, .22, .32, .2, .26],
                        [.17, 0, .34, .21, .28],
                        [.19, .26, 0, .24, .31],
                        [.17, .22, .33, 0, .28],
                        [.18, .24, .35, .23, 0]])   
    
    velos_par_station_initial = np.array([20, 15, 17, 13, 18])

    velos_par_trajet_initial = np.array([[0, 1, 0, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0],
                                [0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 0]])

    np.save("data/temps_moyen_par_trajet.npy", temps_moyen_par_traj)
    np.save("data/taux_depart_a_l_heure.npy", tx_depart_a_lheure)
    np.save("data/routage.npy", routage)
    np.save("data/velos_par_station_initial.npy", velos_par_station_initial)
    np.save("data/velos_par_trajet_initial.npy", velos_par_trajet_initial)
