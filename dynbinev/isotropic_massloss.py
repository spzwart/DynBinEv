#!/usr/bin/env python3
import numpy as np

class IsotropicMassLoss():
    def orbital_elements_evolution(m1, m2, a, e, nu, m1dot, m2dot):
        mdot = m1dot + m2dot
        m = m1+m2
        mdot_m = mdot / m
        e2 = e * e
        one_m_e2 = 1 - e2
        cosnu = np.cos(nu)
        sinnu = np.sin(nu)

        dadt = - mdot_m * a / one_m_e2 * (e2 + 1 + 2*e*cosnu)

        dedt = - mdot_m * (e+cosnu)

        domedt = - mdot_m * sinnu
        return dadt, dedt, domedt