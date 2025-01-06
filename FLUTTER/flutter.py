import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

class Flutter:
    def __init__(self, params):
        self.rho = params["rho"]  # kg/m³
        self.rho_s = params["rho_s"]  # kg/m³
        self.CL_alpha = params["CL_alpha"]
        self.CmF = params["CmF"]
        self.alpha0 = params["alpha0"]
        self.alphaL0 = params["alphaL0"]
        self.k_alpha = params["k_alpha"]  # N.m/rad
        self.k_z = params["k_z"]  # N/m
        self.U = params["U"]  # m/s
        self.J0 = params["J0"]  # kg.m²
        self.a = params["a"]  # m
        self.m = params["m"]  # kg
        self.d = params["d"]  # m
        self.c = params["c"]  # m
        self.delta_b = params["delta_b"]  # m
        self.U_test = 75 # m.s

        # Fréquences propres en torsion et flexion
        self.lambda_z = self.k_z / self.m
        self.lambda_alpha = self.k_alpha / self.J0

        # Aire de référence
        self.S = self.c * self.delta_b

    def determinant_sans_forcage(self, lambd):
        """Calcul du déterminant sans forçage"""
        A = 1 - (self.m * self.d ** 2) / self.J0
        return A * lambd ** 2 - (self.lambda_alpha + self.lambda_z) * lambd + self.lambda_z * self.lambda_alpha

    def racines_sans_forcage(self):
        """Calcul des racines du determinant sans forçage"""
        coeffs = [
            1 - (self.m * self.d ** 2) / self.J0,
            -(self.lambda_alpha + self.lambda_z),
            self.lambda_z * self.lambda_alpha
        ]
        racines = np.roots(coeffs)
        return racines

    def pulsation_sans_forçage(self):
        roots = self.racines_sans_forcage()
        print(f"pulsation pour \u03BB_1\n\u03C9_11={roots[0]}\t\u03C9_12={-roots[0]}")
        print(f"pulsation pour \u03BB_2\n\u03C9_21={roots[1]}\t\u03C9_22={-roots[1]}")

    def sans_forage(self):
        _lambda = self.racines_sans_forcage()
        line()
        print(f"Racines sans forçage :\n\u03BB_1={_lambda[0]} et \u03BB_2={_lambda[1]}")
        line()
        self.pulsation_sans_forçage()
        line()

    def determinant_avec_forcage(self, U):
        """ Calcul du déterminant avec forçage"""
        q = 0.5 * self.rho * U ** 2
        r = (q * self.S * self.CL_alpha) / self.J0
        s = (q * self.S * self.CL_alpha) / self.m

        A = 1 - (self.m * self.d ** 2) / self.J0
        B = -(self.lambda_alpha + self.lambda_z) + r * (self.a - self.d)
        C = self.lambda_z * (self.lambda_alpha - self.a * r)

        delta = B ** 2 - 4 * A * C
        return delta

    def vitesse_critique(self):
        """Determination de la vitesse critique en utilisant
          le determinant avec forçage"""
        Uc = fsolve(self.determinant_avec_forcage, self.U / 2) # initialisation
        # de la vitesse au hasard (U/2)
        return Uc[0] # fsolve retourne les résultats sous forme de tableau numpy.

    def tracer_frequences(self):
        """Fonction de calcul des fréquences """
        vitesses = np.linspace(0, self.U, 100)
        frequences = []

        for U in vitesses:
            delta = self.determinant_avec_forcage(U)
            if delta >= 0:
                freqs = np.sqrt(np.roots([
                    1 - (self.m * self.d ** 2) / self.J0,
                    -(self.lambda_alpha + self.lambda_z) +
                    (0.5 * self.rho * U ** 2 * self.S * self.CL_alpha) / self.J0,
                    self.lambda_z * (self.lambda_alpha - self.a *
                                     (0.5 * self.rho * U ** 2 * self.S * self.CL_alpha) / self.J0)
                ]))
                frequences.append(freqs)
            else:
                frequences.append([0, 0])

        frequences = np.array(frequences)
        print(frequences)
        plt.plot(vitesses, frequences[:, 0],linewidth = 3, label="f z")
        plt.plot(vitesses, frequences[:, 1],linewidth = 3, label="f a")
        plt.axvline(self.vitesse_critique(), color='r',linewidth = 2,
                     linestyle='--', label='Vitesse critique')
        plt.xlabel('Vitesse (m/s)')
        plt.ylabel('Fréquence (Hz)')
        plt.title('Fréquences en fonction de la vitesse')
        plt.grid()
        plt.legend()
        plt.show()

    def pulsations_avec_forcage(self, U):
        """Fonction qui fournie les 4 pulsations du système forcé
        Et les affiche sous forme de tableau"""
        q = 0.5 * self.rho * U ** 2
        r = (q * self.S * self.CL_alpha) / self.J0
        s = (q * self.S * self.CL_alpha) / self.m

        # Coefficients du polynôme caractéristique
        A = 1 - (self.m * self.d ** 2) / self.J0
        B = -(self.lambda_alpha + self.lambda_z) + r * (self.a - self.d)
        C = self.lambda_z * (self.lambda_alpha - self.a * r)

        # Résolution du polynôme caractéristique
        coeffs = [A, B, C, -s, 0]  # Polynôme de degré 4
        lambdas = np.roots(coeffs)  # Trouve les racines du polynôme

        # Conversion des racines en pulsations complexes (ω = sqrt(λ))
        pulsations = np.sqrt(lambdas)
        print("test =",lambdas) # Trouve les racines du polynôme
        data = {
                "Pulsation (ω)": [f"ω{i+1}" for i in range(4)],
                "Partie Réelle": np.real(pulsations),
                "Partie Imaginaire": np.imag(pulsations)
                }
        df = pd.DataFrame(data)

        print(f" Pulsations complexes pour U = {U} m/s :")
        print(df)

    def solve(self):
        """Showing the results"""
        # flutter = Flutter()
        self.racines_sans_forcage()
        print("Vitesse critique :", self.vitesse_critique())
        self.tracer_frequences()
