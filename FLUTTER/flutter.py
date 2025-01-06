import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Flutter:
    def __init__(self):
        # Paramètres donnés
        self.rho = 1        # kg/m³
        self.rho_s = 600    # kg/m³
        self.CL_alpha = 2 * np.pi
        self.CmF = 0.01
        self.alpha0 = np.radians(5)
        self.alphaL0 = np.radians(-3)
        self.k_alpha = 9e4  # N.m/rad
        self.k_z = 3e4      # N/m
        self.U = 120        # m/s
        self.J0 = 205       # kg.m²
        self.a = 0.5        # m
        self.m = 600        # kg
        self.d = -0.2       # m
        self.c = 2          # m
        self.delta_b = 1    # m

        # Fréquences propres en torsion et flexion
        self.lambda_z = self.k_z / self.m
        self.lambda_alpha = self.k_alpha / self.J0

        # Aire de référence
        self.S = self.c * self.delta_b

    def determinant_sans_forcage(self, lambd):
        A = 1 - (self.m * self.d ** 2) / self.J0
        return A * lambd ** 2 - (self.lambda_alpha + self.lambda_z) * lambd + self.lambda_z * self.lambda_alpha

    def racines_sans_forcage(self):
        coeffs = [
            1 - (self.m * self.d ** 2) / self.J0,
            -(self.lambda_alpha + self.lambda_z),
            self.lambda_z * self.lambda_alpha
        ]
        racines = np.roots(coeffs)
        return racines

    def determinant_avec_forcage(self, U):
        q = 0.5 * self.rho * U ** 2
        r = (q * self.S * self.CL_alpha) / self.J0
        s = (q * self.S * self.CL_alpha) / self.m

        A = 1 - (self.m * self.d ** 2) / self.J0
        B = -(self.lambda_alpha + self.lambda_z) + r * (self.a - self.d)
        C = self.lambda_z * (self.lambda_alpha - self.a * r)

        delta = B ** 2 - 4 * A * C
        return delta

    def vitesse_critique(self):
        def delta_nul(U):
            return self.determinant_avec_forcage(U)

        Uc = fsolve(delta_nul, self.U / 2)
        return Uc[0]

    def tracer_frequences(self):
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
        plt.plot(vitesses, frequences[:, 0],linewidth = 3, label="")
        plt.axvline(self.vitesse_critique(), color='r',linewidth = 2,
                     linestyle='--', label='Vitesse critique')
        plt.xlabel('Vitesse (m/s)')
        plt.ylabel('Fréquence (Hz)')
        plt.title('Fréquences en fonction de la vitesse')
        plt.grid()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    flutter = Flutter()
    print("Racines sans forçage :", flutter.racines_sans_forcage())
    print("Vitesse critique :", flutter.vitesse_critique())
    flutter.tracer_frequences()
