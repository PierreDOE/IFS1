# -*- coding: utf-8 -*-
"""

Pierre DOERFLER, January 2025

"""

import matplotlib.pyplot as plt
from fsc.falknerskan import FalknerSkan
from fsc.fsk_appli import FalknerSkanAppli
# from fsc.fsk_appli import FalknerSkanAppli
from toolbox.colored_messages import *

par_fsk = dict(eta_max=5.9,
               npt=201,
               beta=-0.0,
               method=1,
               grid="geometric",
               verbose=False,
               plot=True
               )

s = FalknerSkan(par_fsk)
s.solve()
print("outputs: ", s.output)

s = FalknerSkanAppli(dict(beta=(0, -0.50), npt_beta=3), par_fsk)
s.beta_sweep()  # lancement du balayage
s.save()       # sauvegarde dans un fichier outputs/output.out
print(s)       # affichage d'un tableau

# FalknerSkan.display_profiles()
# set_info("normal end of execution")

if __name__ == "__main__":
    flutter = Flutter()
    print("Racines sans forçage :", flutter.racines_sans_forcage())
    print("Vitesse critique :", flutter.vitesse_critique())
    flutter.tracer_frequences()
