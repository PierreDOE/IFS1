import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from FLUTTER.tools import line
from tabulate import tabulate

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

        # test
        self.U_test = 116 # m/s
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

    def pulsation_sans_forcage(self):
        """Fonction qui affiche les pulsation du mode sans forcage sous forme de tableau"""
        roots = self.racines_sans_forcage()
        print(f"pulsation pour \u03BB_1\n\u03C9_11={np.sqrt(roots[0])}\t\u03C9_12={-np.sqrt(roots[0])}")
        line()
        print(f"pulsation pour \u03BB_2\n\u03C9_21={np.sqrt(roots[1])}\t\u03C9_22={-np.sqrt(roots[1])}")
        line()

    def sans_forcage(self):
        _lambda = self.racines_sans_forcage()
        roots = self.racines_sans_forcage()
        line()
        print(f"Racines sans forçage :\n\u03BB_1={_lambda[0]} et \u03BB_2={_lambda[1]}")
        line()
        print(f"Decomposition de \u03BB_1 :\n Re(\u03BB_1)={np.real(_lambda[0])} et Im(\u03BB_1)={np.imag(_lambda[0])}")
        line()
        print(f"Decomposition de \u03BB_2 :\n Re(\u03BB_2)={np.real(_lambda[1])} et Im(\u03BB_2)={np.imag(_lambda[1])}")
        line()
        self.pulsation_sans_forcage()
        line()

    def determinant_avec_forcage(self, U):
        """ Calcul du déterminant avec forçage"""
        self.q = 0.5 * self.rho * U ** 2
        self.r = (self.q * self.S * self.CL_alpha) / self.J0
        self.s = (self.q * self.S * self.CL_alpha) / self.m

        A = 1 - (self.m * self.d ** 2) / self.J0
        B = -(self.lambda_alpha + self.lambda_z) + self.r * (self.a - self.d)
        C = self.lambda_z * (self.lambda_alpha - self.a * self.r)

        delta = B ** 2 - 4 * A * C
        return delta

    def racines_avec_forcage(self,U):
        """Calcul des racines du determinant avec forçage"""
        delta = self.determinant_avec_forcage(U)
        coeffs = [
                1 - (self.m * self.d ** 2) / self.J0,  # A
                -(self.lambda_alpha + self.lambda_z) + self.r * (self.a - self.d),  # B
                self.lambda_z * (self.lambda_alpha - self.a * self.r) # C    
        ]
        racines = np.roots(coeffs)
        return racines

    def pulsation_avec_forcage(self,U):
        """Fonction qui retourne les pulsations du mode avec forcage"""
        roots = self.racines_avec_forcage(U)
        print(f"Pulsation pour \u03BB_1\n\u03C9_11={np.sqrt(roots[0])}\t\u03C9_12={-np.sqrt(roots[0])}")
        line()
        print(f"Pulsation pour \u03BB_2\n\u03C9_21={np.sqrt(roots[1])}\t\u03C9_22={-np.sqrt(roots[1])}")
        line()
        print(f"Décomposition pour \u03C9_11 et \u03C9_12\n Re(\u03C9_11)={np.real(np.sqrt(roots[0]))}\tIm(\u03C9_11)={np.imag(np.sqrt(roots[0]))} \nRe(\u03C9_12)={np.real(-np.sqrt(roots[0]))}\tIm(\u03C9_12)={np.imag(-np.sqrt(roots[0]))}")
        line()
        print(f"Décomposition pour \u03C9_21 et \u03C9_22\n Re(\u03C9_21)={np.real(np.sqrt(roots[1]))}\tIm(\u03C9_21)={np.imag(np.sqrt(roots[1]))} \nRe(\u03C9_22)={np.real(-np.sqrt(roots[1]))}\tIm(\u03C9_22)={np.imag(-np.sqrt(roots[1]))}")

    def avec_forcage(self,U):
        """Fonction qui affiche les racines et pulsation du système"""
        _lambda = self.racines_avec_forcage(U)
        line()
        print(f"Racines pour U = {U}")
        print(f"Racines avec forçage :\n\u03BB_1={_lambda[0]} et \u03BB_2={_lambda[1]}")
        line()
        print(f"Decomposition de \u03BB_1 :\n Re(\u03BB_1)={np.real(_lambda[0])} et Im(\u03BB_1)={np.imag(_lambda[0])}")
        line()
        print(f"Decomposition de \u03BB_2 :\n Re(\u03BB_2)={np.real(_lambda[1])} et Im(\u03BB_2)={np.imag(_lambda[1])}")
        line()

        self.pulsation_avec_forcage(U)
        line()

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
            freqs = np.sqrt(np.roots([
                1 - (self.m * self.d ** 2) / self.J0,  # A
                -(self.lambda_alpha + self.lambda_z) + self.r * (self.a - self.d),  # B
                self.lambda_z * (self.lambda_alpha - self.a * self.r)]))  # C
            frequences.append(freqs)

        frequences = np.array(frequences)
        plt.plot(vitesses, frequences[:, 0], linewidth=3, label="f 1")
        plt.plot(vitesses, frequences[:, 1], linewidth=3, label="f 2")
        plt.axvline(self.vitesse_critique(), color='r',linewidth = 2,
                     linestyle='--', label='Vitesse critique')
        plt.xlabel('Vitesse (m/s)')
        plt.ylabel('Fréquence (Hz)')
        plt.title('Fréquences en fonction de la vitesse')
        plt.grid()
        plt.legend()
        plt.show()

    def tracer_fonctions_transfert(self):
        """Trace les fonctions de transferts en semi-log"""
        f = np.linspace(0, 5, 500)
        omega = 2 * np.pi * f
        lambd = omega ** 2

        TzL = (self.lambda_alpha - lambd) / (self.m * self.determinant_sans_forcage(lambd))
        TalphaL = lambd / self.determinant_sans_forcage(lambd)
        TzM = TalphaL
        TalphaM = (self.lambda_z - lambd) / (self.J0 * self.determinant_sans_forcage(lambd))

        plt.figure(1)
        ax1 = plt.subplot(311)
        plt.title('Diagrammes des fonctions de transfert')
        plt.semilogy(f, np.abs(TzL),color='b',linewidth = 2, label=r'$T_{zl}$')
        plt.tick_params('x', labelbottom=False)
        plt.legend()
        plt.grid()
        plt.subplot(312, sharex=ax1)
        plt.semilogy(f, np.abs(TalphaL),color='r',linewidth = 2, label=r'$T_{\alpha L}$')
        plt.tick_params('x', labelbottom=False)
        plt.ylabel('Module des fonctions de transfert')
        plt.legend()
        plt.grid()
        plt.subplot(313)
        plt.semilogy(f, np.abs(TalphaM),color='g',linewidth = 2, label=r'$T_{\alpha M}$')
        plt.xlabel('Fréquence (Hz)')
        plt.grid()
        plt.legend()
        plt.show()

    def solve(self):
        """Showing the results"""
        # flutter = Flutter()
        self.sans_forcage()
        print("Vitesse critique :", self.vitesse_critique())
        self.avec_forcage(self.U_test)
        self.tracer_frequences()
        self.tracer_fonctions_transfert()


