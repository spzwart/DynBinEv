#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from scipy.integrate import solve_ivp
import numpy as np
import functools

def mod2pi(f):
    while f < 0:
        f += 2 * np.pi
    while f > 2 * np.pi:
        f -= 2 * np.pi
    return f


class MacegaPhaseEvolve():
    """
    Evolves the set of ODEs in a, e, ome, and nu,
    and optionally g, the envelope expansion factor
    """

    def C_from_X(X, mu, a, l, k):
        """
        Obtains C coeffiecient from dimensionless inspiral parameter X (\chi)
        :param mu: standard gravitational parameter
        :param a: initial semimajor axis
        :param l: l exponent
        :param k: k exponent
        :return:
        """
        C = X / np.pi * mu**(1-l/2) * a**(l/2+k-2)
        Cunit_L = 1 - l + k
        Cunit_T = l - 2
        return C, Cunit_L, Cunit_T

    def __init__(self, l=2, k=0, force_generic=False, evolve_g=False, limit_radius=None):
        """
        Initializes the model
        :param l: l exponent
        :param k: k exponent
        :param force_generic: use the equations for generic k and l
        :param evolve_g: have the coefficient g evolve with time, halting the inspiral
        :param limit_radius: limit the force to be non-zero only within this radius (simulates the finite size of envelope)
        """
        self.l = l
        self.k = k

        self.dnu_dt = self.dnu_dt_unperturbed
        self.force_generic = force_generic
        self.evolve_g = evolve_g
        self.limit_radius = limit_radius

        self.select_model(l, k)

        self.length_unit = 1 * units.RSun
        self.time_unit = 1 * units.yr
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)
        self.mass_unit = self.conv.to_si(1 | nbody_system.mass).as_unit()
        print("Mass units:", (1|self.mass_unit).as_string_in(units.MSun))

        Cunits = nbody_system.length ** (1 - l + k) * nbody_system.time ** (l - 2)
        try:
            self.Cunits = self.conv.to_si(Cunits).as_unit()
        except AttributeError:
            self.Cunits = units.none
        print("C units:", Cunits)

        self.method = "DOP853"

    def select_model(self, l, k):
        """
        Select the inspiral model, only power-law for now
        :param l: l exponent
        :param k: k exponent
        :return:
        """
        if not self.force_generic and k == 0 and l == 2:
            self.da_dt = self.da_dt_l2k0
            self.de_dt = self.de_dt_l2k0
            self.dome_dt = self.dome_dt_l2k0
        elif not self.force_generic and k == 0 and l == 1:
            self.da_dt = self.da_dt_l1k0
            self.de_dt = self.de_dt_l1k0
            self.dome_dt = self.dome_dt_l1k0
        else:
            print("Using generic functions")
            self.da_dt = functools.partial(self.da_dt_generic, l=l, k=k)
            self.de_dt = functools.partial(self.de_dt_generic, l=l, k=k)
            self.dome_dt = functools.partial(self.dome_dt_generic, l=l, k=k)

    def initialize_integration(self, C, mu):
        """
        Initializes C and mu
        :param C: dimensional coefficient C
        :param mu: standard gravitational parameter
        :return:
        """
        self.C0 = self.C = C
        self.mu = mu

    def calc_derivatives(self, t, y):
        """
        Calcualates derivatives
        :param t: time, not used
        :param y: state vector (a, e, ome, nu, g) g is the self-limiting function
        :return:
        """
        a = y[0]
        e = y[1]
        ome = y[2]
        nu = y[3]
        g = y[4]

        if self.limit_radius is not None:
            r = a * (1 - e*e) / (1 + e*np.cos(nu))
            if r > self.limit_radius:
                return [0, 0, 0, 0, 0]

        if a < 0.0 or a*(1-e) < 1e-8: return [0, 0, 0, 0, 0]

        self.C = self.C0 / g**(3-self.k)

        self.e2 = e * e
        self.ome2 = 1 - self.e2
        self.cosnu = np.cos(nu)
        self.sinnu = np.sin(nu)
        self.elfac = 1 + 2 * e * self.cosnu + self.e2

        adot = self.da_dt(a, e)
        edot = self.de_dt(a, e)
        omedot = self.dome_dt(a, e)
        nudot = self.dnu_dt(a, e) - omedot
        gdot = 0.0

        if self.evolve_g:
            Edot = self.mred * self.mu * adot / (2*a*a)
            gdot = - Edot / self.B0 * g*g
            pass

        return [adot, edot, omedot, nudot, gdot]

    def dnu_dt_unperturbed(self, a, e):
        n = (self.mu / (a * a * a)) ** 0.5
        opecosnu = 1 + e * self.cosnu

        try:
            nudot = n * opecosnu * opecosnu / self.ome2 ** 1.5
        except RuntimeWarning:
            print(self.ome2)
        return nudot

    def da_dt_l2k0(self, a, e):
        adot = -2 * self.C * (self.mu * a) ** 0.5
        adot *= (self.elfac/self.ome2) ** 1.5
        return adot

    def de_dt_l2k0(self, a, e):
        edot = - 2*self.C * (self.mu/a) ** 0.5
        edot *= (self.ome2 * self.elfac) ** 0.5 * (e + self.cosnu)
        return edot

    def dome_dt_l2k0(self, a, e):
        omedot = - 2 * self.C * (self.mu/a) ** 0.5 * self.sinnu
        omedot *= 1 / e * (self.elfac/self.ome2) ** 0.5
        return omedot

    def da_dt_l1k0(self, a, e):
        adot = -2 * self.C * a
        adot *= self.elfac/self.ome2
        return adot

    def de_dt_l1k0(self, a, e):
        edot = - 2*self.C
        edot *= self.ome2 * (e + self.cosnu)
        return edot

    def dome_dt_l1k0(self, a, e):
        omedot = - 2 * self.C / e * self.sinnu
        return omedot

    def da_dt_generic(self, a, e, l, k):
        adot = -2 * self.C * self.mu**((l-1)/2) * a**((3-l-2*k)/2)
        adot *= self.ome2**-((l+1+2*k)/2) * (1 + e*self.cosnu)**k * self.elfac**((l+1)/2)
        return adot

    def de_dt_generic(self, a, e, l, k):
        edot = -2 * self.C * self.mu**((l-1)/2) * a**((1-l-2*k)/2)
        edot *= self.ome2**-((l-1+2*k)/2) * (1 + e*self.cosnu)**k * self.elfac**((l-1)/2) * (e + self.cosnu)
        return edot

    def dome_dt_generic(self, a, e, l, k):
        omedot = -2 * self.C * self.mu**((l-1)/2) * a**((1-l-2*k)/2)
        omedot *= self.ome2**-((l-1+2*k)/2) * (1 + e*self.cosnu)**k * self.elfac**((l-1)/2) * self.sinnu / e
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

    def initialize_system(self, m1, m2, a0, e0, ome0, nu0, C, a1, g0=1, B0=None, lambd=2):
        mu = (m1 + m2) * constants.G
        E0 = m1 * m2 * constants.G / (2 * a0)
        Eps0 = mu / (2 * a0)
        Eps1 = mu / (2 * a1)

        if not np.isscalar(C) and C.is_scalar():
            self.C0 = self.C = C.value_in(self.Cunits)
            print("C = {:s}".format(C.as_string_in(self.Cunits)))
        else:
            self.C0 = self.C = C #| self.Cunits
            print("C =", C, "in", self.Cunits)

        self.Period0 = 2 * np.pi * (a0 * a0 * a0 / mu) ** 0.5
        print("Period0 =", self.Period0.as_string_in(units.yr))

        Eps_ce = Eps1 - Eps0
        print("Eps_ce/Eps0", Eps_ce / Eps0)

        a0_nb = a0.value_in(self.length_unit)
        a1_nb = a1.value_in(self.length_unit)
        mu_nb = mu.value_in(self.length_unit ** 3 / self.time_unit ** 2)
        Period0_nb = self.Period0.value_in(self.time_unit)

        self.a1 = a1_nb
        self.mu = mu_nb

        self.m1 = m1.value_in(self.mass_unit)
        self.m2 = m2.value_in(self.mass_unit)
        self.mred = self.m1*self.m2 / (self.m1 + self.m2)

        if B0 is None:
            self.B0 = self.m1*self.m1/self.a1/lambd
            print("B0:", self.B0)
        else: self.B0 = B0

        self.y0 = [a0_nb, e0, ome0, nu0, g0]

        firstDerivs = self.calc_derivatives(0.0, self.y0)
        adot = firstDerivs[0]
        tdecay = a0_nb / adot
        secularfactor = Period0_nb / tdecay
        print("a/adot = {:g} Periods".format(secularfactor))

    def run_system(self, tfin, dt_out):
        t, y = self.evolve_until(self.y0, tfin.value_in(self.time_unit), afin=self.a1, dt_out=dt_out.value_in(self.time_unit))
        a, e, ome, nu, g = y
        return t, a, e, ome, nu, g


def test_evol():
    m1, m2 = 80 | units.MSun, 55 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.1
    ome0 = 2 * np.pi
    nu0 = np.pi

    # Expansion factor
    g0 = 1

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

    CEPhase = MacegaPhaseEvolve(l=2, k=0)
    CEPhase.initialize_integration(C=C_nb, mu=mu_nb)

    y0 = [a0_nb, e0, ome0, nu0, g0]

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
