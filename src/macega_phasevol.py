#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from scipy.integrate import solve_ivp
import numpy as np


def mod2pi(f):
    while f < 0:
        f += 2 * np.pi
    while f > 2 * np.pi:
        f -= 2 * np.pi
    return f


class MacegaPhaseEvolve():
    def __init__(self, l=2, k=0):
        self.l = l
        self.k = k

        self.dnu_dt = self.dnu_dt_unperturbed
        self.select_model(l, k)

        self.length_unit = 1 * units.RSun
        self.time_unit = 1 * units.yr
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)
        self.mass_unit = self.conv.to_si(1 | nbody_system.mass)
        print("Mass units:", self.mass_unit.as_string_in(units.MSun))

        Cunits = nbody_system.length ** (1 - l + k) * nbody_system.time ** (l - 2)
        self.Cunits = self.conv.to_si(Cunits).as_unit()
        print("C units:", Cunits)

        self.method = "DOP853"

    def select_model(self, l, k):
        if k == 0 and l == 2:
            self.da_dt = self.da_dt_l2k0
            self.de_dt = self.de_dt_l2k0
            self.dome_dt = self.dome_dt_l2k0
        elif k == 0 and l == 1:
            self.da_dt = self.da_dt_l1k0
            self.de_dt = self.de_dt_l1k0
            self.dome_dt = self.dome_dt_l1k0
        else:
            raise ValueError("forceform option l={:g}, k={:g} not found".format(l, k))

    def initialize_integration(self, C, mu):
        self.C = C
        self.mu = mu

    def calc_derivatives(self, t, y):
        a = y[0]
        e = y[1]
        ome = y[2]
        nu = y[3]

        self.e2 = e * e
        self.ome2 = 1 - self.e2
        self.cosnu = np.cos(nu)
        self.elfac = 1 + 2 * e * self.cosnu + self.e2

        adot = self.da_dt(a)
        edot = self.de_dt(a, e)
        omedot = self.dome_dt(a, e)
        nudot = self.dnu_dt(a, e) - omedot

        return [adot, edot, omedot, nudot]

    def dnu_dt_unperturbed(self, a, e):
        n = (self.mu / (a * a * a)) ** 0.5
        opecosnu = 1 + e * self.cosnu

        nudot = n * opecosnu * opecosnu / self.ome2 ** 1.5
        return nudot

    def da_dt_l2k0(self, a):
        adot = -2 * self.C * self.mu ** 0.5
        adot *= a ** 0.5 * self.ome2 ** -1.5 * self.elfac ** 1.5
        return adot

    def de_dt_l2k0(self, a, e):
        edot = - self.C * self.mu ** 0.5
        edot *= 1 / e * a ** -0.5 * self.ome2 ** 0.5 * self.elfac ** 0.5 * (self.e2 + 2 * e * self.cosnu)
        return edot

    def dome_dt_l2k0(self, a, e):
        omedot = - 2 * self.C * self.mu ** 0.5
        omedot *= 1 / e * a ** -0.5 * self.ome2 ** -0.5 * self.elfac ** 0.5
        return omedot

    def da_dt_l1k0(self, a):
        adot = -2 * self.C * a
        adot *= self.elfac/self.ome2
        return adot

    def de_dt_l1k0(self, a, e):
        edot = - self.C
        edot *= self.ome2 / e * (self.e2 + 2 * e * self.cosnu)
        return edot

    def dome_dt_l1k0(self, a, e):
        omedot = - 2 * self.C / e
        return omedot

    def evolve(self, y0, tfin, t_eval):
        sol = solve_ivp(self.calc_derivatives, [0, tfin], y0, t_eval=t_eval, method=self.method)
        return sol.t, sol.y

    def evolve_until(self, y0, tfin, afin, dt_out, tstart=0):
        time = tstart
        tsols = [tstart]
        ysols = [y0]
        y = y0
        while time < tfin:
            nexttime = time + dt_out
            sol = solve_ivp(self.calc_derivatives, [time, nexttime], y, t_eval=[nexttime], method=self.method)

            if sol.y[0] < afin:
                print("Reached final semimajor axis, halting integration")
                break

            y = sol.y.flatten()
            y[2] = mod2pi(y[2])
            y[3] = mod2pi(y[3])

            tsols.append(sol.t.item())
            ysols.append(y)
            time = nexttime

        return tsols, np.vstack(ysols).T

    def initialize_system(self, m1, m2, a0, e0, ome0, nu0, C_nb, a1):
        mu = (m1 + m2) * constants.G
        E0 = m1 * m2 * constants.G / (2 * a0)
        Eps0 = mu / (2 * a0)
        Eps1 = mu / (2 * a1)
        C = C_nb | self.Cunits

        self.Period0 = 2 * np.pi * (a0 * a0 * a0 / mu) ** 0.5
        print("Period0 =", self.Period0.as_string_in(units.yr))

        Eps_ce = Eps1 - Eps0
        print("Eps_ce/Eps0", Eps_ce / Eps0)

        a0_nb = a0.value_in(self.length_unit)
        a1_nb = a1.value_in(self.length_unit)
        mu_nb = mu.value_in(self.length_unit ** 3 / self.time_unit ** 2)
        Period0_nb = self.Period0.value_in(self.time_unit)

        self.a1 = a1_nb
        self.C = C_nb
        self.mu = mu_nb

        self.y0 = [a0_nb, e0, ome0, nu0]

        firstDerivs = self.calc_derivatives(0.0, self.y0)
        adot = firstDerivs[0]
        tdecay = a0_nb / adot
        secularfactor = Period0_nb / tdecay
        print("a/adot = {:g} Periods".format(secularfactor))

    def run_system(self, tfin, dt_out):
        t, y = self.evolve_until(self.y0, tfin.value_in(self.time_unit), afin=self.a1, dt_out=dt_out.value_in(self.time_unit))
        a, e, ome, nu = y
        return t, a, e, ome, nu


def test_evol():
    m1, m2 = 80 | units.MSun, 55 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.1
    ome0 = 2 * np.pi
    nu0 = np.pi

    mu = (m1 + m2) * constants.G
    E0 = m1 * m2 * constants.G / (2 * a0)
    Eps0 = mu / (2 * a0)

    a1 = 40 | units.RSun
    Tce = 1000 | units.yr
    Eps1 = mu / (2 * a1)
    C = 1e-5 | units.RSun

    Period0 = 2 * np.pi * (a0 * a0 * a0 / mu) ** 0.5
    print("Period0 =", Period0.value_in(units.yr))

    Eps_ce = Eps1 - Eps0
    print("Eps_ce/Eps0", Eps_ce / Eps0)

    conv = nbody_system.nbody_to_si(1 | units.RSun, 1 | units.yr)

    TCE_nb = conv.to_nbody(Tce).number
    a0_nb = conv.to_nbody(a0).number
    a1_nb = conv.to_nbody(a1).number
    mu_nb = conv.to_nbody(mu).number
    C_nb = conv.to_nbody(C).number
    Period0_nb = conv.to_nbody(Period0).number

    CEPhase = PhaseDependentEvol(l=2, k=0)
    CEPhase.initialize_integration(C=C_nb, mu=mu_nb)

    y0 = [a0_nb, e0, ome0, nu0]

    derivs = CEPhase.calc_derivatives(0.0, y0)
    print(derivs)

    tfin = Period0_nb * 200
    dt_out = Period0_nb / 500
    t, y = CEPhase.evolve_until(y0, tfin, afin=a1_nb, dt_out=dt_out)
    a, e, ome, nu = y

    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    f, ax = plt.subplots(4, sharex=True, figsize=(15, 10))

    ax[0].plot(t, a, lw=4, label="phase-dependent, integrated")

    ax[0].set_ylabel("semimajor axis [RSun]")
    ax[0].axhline(a0.value_in(units.RSun), c="black", ls="--")
    ax[0].axhline(a1.value_in(units.RSun), c="red", ls="--", label="zero eccentricity final orbit")
    ax[0].legend()
    ax[0].set_title("e0 = {:g}".format(e0))

    ax[0].set_yscale("log")
    ax[1].plot(t, e, lw=2)
    ax[1].set_ylabel("eccentricity")
    ax[1].set_xlabel("time [yr]")
    ax[1].set_ylim(bottom=0)

    ax[2].plot(t, ome, lw=2)
    ax[2].set_ylabel("omega")
    ax[2].set_xlabel("time [yr]")
    ax[2].set_ylim(bottom=0, top=2 * np.pi)

    ax[3].plot(t, nu, lw=2)
    ax[3].set_ylabel("nu")
    ax[3].set_xlabel("time [yr]")
    ax[3].set_ylim(bottom=0, top=2 * np.pi)

    f.subplots_adjust(hspace=0)

    plt.show()
    # plt.savefig("averaged_eccentric_integrated_vs_analytic.pdf")


if __name__ == "__main__":
    test_evol()
