#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.community.hermite.interface import Hermite
from amuse.ext.orbital_elements import orbital_elements_from_binary

from isotropic_massloss import IsotropicMassLoss
from dynbindev_utilities import make_binary_star, check_collisions, get_period, mod2pi
from dynbindev_utilities import stateid as st

from scipy.integrate import solve_ivp
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

class PerturbativeEvolution():
    """
    Evolves the set of ODEs in a, e, ome, and nu,
    """
    def dnudt(a, e, m1, m2, nu):
        one_m_e2 = 1 - e*e
        mu = m1 + m2
        n = (mu / (a * a * a)) ** 0.5
        opecosnu = 1 + e * np.cos(nu)
        nudot = n * opecosnu * opecosnu / one_m_e2 ** 1.5
        return nudot

    def __init__(self, method="DOP853"):
        self.length_unit = 1 * units.RSun
        self.time_unit = 1 * units.yr
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)
        self.mass_unit = self.conv.to_si(1 | nbody_system.mass).as_unit()
        self.method = method
        self.m1dot = self.m2dot = 0.0

    def select_model(self, isotropic_mass_loss=True):
        self.isotropic_mass_loss = isotropic_mass_loss

    def set_mass_loss(self, m1dot, m2dot):
        """
        Mass loss in N-body units
        """
        self.m1dot = m1dot
        self.m2dot = m2dot

    def compute_derivatives(self, t, y):
        """
        Calcualates derivatives
        :param t: time, not used
        :param y: state vector (m1, m2, a, e, i, ome, Ome, nu)
        :return:
        """
        ### Input values need to be plain scalars in N-body units (G=1)
        ### TODO: Make it accept AMUSE quantity
        m1, m2, a, e, i, ome, Ome, nu = y

        dydt = np.zeros(8)

        dnudt = PerturbativeEvolution.dnudt(a, e, m1, m2, nu)
        dm1dt = self.m1dot
        dm2dt = self.m2dot

        if self.isotropic_mass_loss:
            dadt, dedt, domedt = IsotropicMassLoss.orbital_elements_evolution(m1, m2, a, e, nu, dm1dt, dm2dt)
            dydt[st.a] += dadt
            dydt[st.e] += dedt
            dydt[st.ome] += domedt
            dydt[st.m1] += dm1dt
            dydt[st.m2] += dm2dt
        dydt[st.nu] += dnudt - dydt[st.ome]

        return dydt

    def evolve_until(self, y0, tfin, dt_out, afin=None, tstart=0):
        time = tstart
        tsols = [tstart]
        ysols = [y0]
        y = y0
        while time < tfin:
            nexttime = time + dt_out
            sol = solve_ivp(self.compute_derivatives, [time, nexttime], y, t_eval=[nexttime], method=self.method)

            if afin is not None:
                if sol.y[st.a] < afin:
                    print("Reached final semimajor axis, halting integration")
                    break

            y = sol.y.flatten()
            # Keep cyclic angles in 0-2pi
            y[st.nu] = mod2pi(y[st.nu])
            y[st.ome] = mod2pi(y[st.ome])
            y[st.Ome] = mod2pi(y[st.Ome])

            tsols.append(sol.t.item())
            ysols.append(y)
            time = nexttime

        return tsols, np.vstack(ysols).T


def test_perturbative_evol():
    m1_0, m2_0 = 80 | units.MSun, 55 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.1
    i0 = 0.0
    ome0 = 2 * np.pi
    Ome0 = 0.0
    nu0 = np.pi

    mu0 = (m1_0 + m2_0) * constants.G

    afin = 40 | units.RSun

    Period0 = 2 * np.pi * (a0 * a0 * a0 / mu0) ** 0.5
    print("Period0 =", Period0.value_in(units.yr))

    conv = nbody_system.nbody_to_si(1 | units.RSun, 1 | units.yr)
    to_MSun = conv.to_si(1 | nbody_system.mass).value_in(units.MSun)

    a0_nb = conv.to_nbody(a0).number
    afin_nb = conv.to_nbody(afin).number
    m1_0_nb = conv.to_nbody(m1_0).number
    m2_0_nb = conv.to_nbody(m2_0).number
    Period0_nb = conv.to_nbody(Period0).number
    dm1dt_nb = conv.to_nbody(-1e-1 | units.MSun / units.yr).number
    dm2dt_nb = conv.to_nbody(-1e-1 | units.MSun / units.yr).number

    PertEvolve = PerturbativeEvolution()
    PertEvolve.select_model(isotropic_mass_loss=True)
    PertEvolve.set_mass_loss(dm1dt_nb, dm2dt_nb)

    y0 = [m1_0_nb, m2_0_nb, a0_nb, e0, i0, ome0, Ome0, nu0]

    tfin = Period0_nb * 20
    dt_out = Period0_nb / 100
    t, y = PertEvolve.evolve_until(y0, tfin, afin=afin_nb, dt_out=dt_out)
    m1_nb, m2_nb, a, e, i, ome, Ome, nu = y
    m1, m2 = m1_nb * to_MSun, m2_nb * to_MSun

    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    f, ax = plt.subplots(5, sharex=True, figsize=(15, 10))

    ax[0].plot(t, a, lw=4)
    ax[0].set_ylabel("semimajor axis [RSun]")
    ax[0].axhline(a0.value_in(units.RSun), c="black", ls="--")
    ax[0].axhline(afin.value_in(units.RSun), c="red", ls="--")

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
    ax[3].set_ylim(bottom=0, top=2 * np.pi)

    ax[4].plot(t, m1, lw=2)
    ax[4].plot(t, m2, lw=2)
    ax[4].set_ylabel("mass [MSun]")
    ax[4].set_xlabel("time [yr]")

    for axx in ax: axx.grid()

    f.subplots_adjust(hspace=0)

    plt.show()
    # plt.savefig("averaged_eccentric_integrated_vs_analytic.pdf")


class NbodyEvolution():
    def __init__(self, dtkick=0.01, dtkick_update=True):
        """
        Initialization
        :param dtkick: timestep between kicks, as fraction of binary periods
        :param dtkick_update: dtkick as the binary shrinks, or keep the timestep constant
        """
        self.length_unit = 1 * units.RSun
        self.time_unit = 1 * units.yr
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)
        self.mass_unit = self.conv.to_si(1 | nbody_system.mass)
        self.dtkick = dtkick
        self.dtkick_update = dtkick_update

    def set_mass_loss(self, m1dot, m2dot):
        """
        Mass loss rates
        """
        self.m1dot = m1dot
        self.m2dot = m2dot

    def initialize_model(self, m1, m2, a0, e0, ome0, nu0, afin):
        """
        Initializes the kick module
        :param m1: mass star 1
        :param m2: mass star 2
        :param a0: semimajor axis
        :param e0: eccentricity
        :param ome0: argument of pericenter
        :param nu0: phase
        :param afin: final semimajor axis (to stop the simulation)
        :return:
        """

        double_star, stars = make_binary_star(m1, m2,
                                              a0, e0,
                                              argument_of_periapsis=ome0|units.rad,
                                              true_anomaly=nu0|units.rad)

        self.stars = stars
        self.gravity = Hermite(self.conv)
        self.gravity.particles.add_particle(self.stars)
        self.to_stars = self.gravity.particles.new_channel_to(self.stars)
        self.from_stars = self.stars.new_channel_to(self.gravity.particles)

        self.Period0 = get_period(double_star)
        print("Period =", self.Period0.as_string_in(units.yr))
        print("Steps per period: = {:1.2f}".format(self.dtkick))

        mu = double_star.mass * constants.G
        self.afin = afin
        self.double_star = double_star

    def update_stars(self, stars, dt):
        stars.mass[0] += self.m1dot*dt
        stars.mass[1] += self.m2dot*dt

        # Kick stars here

    def run_model(self, tfin, dt_out, tstart=0|units.yr):
        time = tstart
        dt = self.Period0 * self.dtkick
        dtout_next = time + dt_out

        a = [self.double_star.semimajor_axis.value_in(self.length_unit)]
        e = [self.double_star.eccentricity]
        ome = [self.double_star.argument_of_periapsis.value_in(units.rad)]
        nu = [self.double_star.true_anomaly.value_in(units.rad)]
        t = [tstart.value_in(self.time_unit)]
        m1, m2 = [self.stars.mass[0].value_in(units.MSun)], [self.stars.mass[1].value_in(units.MSun)]
        while time < tfin:
            time += dt
            self.gravity.evolve_model(time)
            self.to_stars.copy()
            self.update_stars(self.stars, dt)
            self.from_stars.copy()

            orbital_elements = orbital_elements_from_binary(self.stars,
                                                            G=constants.G)

            collision = check_collisions(self.stars)
            if collision: break
            if orbital_elements[2] < self.afin: break

            if self.dtkick_update == True:
                Period = 2 * np.pi * (orbital_elements[2] * orbital_elements[2] * orbital_elements[2] /
                                      (constants.G * self.stars.mass.sum())).sqrt()
                dt = Period*self.dtkick

            a.append(orbital_elements[2].value_in(self.length_unit))
            e.append(orbital_elements[3])
            ome.append(mod2pi(np.radians(orbital_elements[7])))
            nu.append(mod2pi(np.radians(orbital_elements[4])))
            t.append(time.value_in(self.time_unit))
            m1.append(self.stars.mass[0].value_in(units.MSun))
            m2.append(self.stars.mass[1].value_in(units.MSun))
            print("time=", time.value_in(self.time_unit),
                  "a=", a[-1],
                  "e=", e[-1],
                  "m=", self.stars.mass.in_(units.MSun), end="\r")

        self.gravity.stop()

        return t, a, e, ome, nu, m1, m2

def test_nbody_evol():
    m1_0, m2_0 = 80 | units.MSun, 55 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.1
    i0 = 0.0
    ome0 = 2 * np.pi
    Ome0 = 0.0
    nu0 = np.pi

    afin = 40 | units.RSun
    dm1dt = -1e-1 | units.MSun / units.yr
    dm2dt = -1e-1 | units.MSun / units.yr

    NbodyEvolve = NbodyEvolution(dtkick=0.05)
    NbodyEvolve.initialize_model(m1_0, m2_0, a0, e0, ome0, nu0, afin=afin)
    NbodyEvolve.set_mass_loss(dm1dt, dm2dt)

    tfin = NbodyEvolve.Period0*20
    dtout = NbodyEvolve.Period0*0.01
    t, a, e, ome, nu, m1, m2 = NbodyEvolve.run_model(tfin, dtout)

    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    f, ax = plt.subplots(nrows=5, sharex=True, figsize=(15, 10))

    ax[0].plot(t, a, lw=4)
    ax[0].set_ylabel("semimajor axis [RSun]")
    ax[0].axhline(a0.value_in(units.RSun), c="black", ls="--")
    ax[0].axhline(afin.value_in(units.RSun), c="red", ls="--")

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
    ax[3].set_ylim(bottom=0, top=2 * np.pi)

    ax[4].plot(t, m1, lw=2)
    ax[4].plot(t, m2, lw=2)
    ax[4].set_ylabel("mass [MSun]")
    ax[4].set_xlabel("time [yr]")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.show()

def test_compare_nbody_perturbative():
    m1_0, m2_0 = 80 | units.MSun, 55 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.1
    i0 = 0.0
    ome0 = np.pi
    Ome0 = 0.0
    nu0 = np.pi

    afin = 40 | units.RSun
    dm1dt = -0.5e-1 | units.MSun / units.yr
    dm2dt = -0.5e-1 | units.MSun / units.yr

    mu0 = (m1_0 + m2_0) * constants.G

    Period0 = 2 * np.pi * (a0 * a0 * a0 / mu0) ** 0.5
    print("Period0 =", Period0.value_in(units.yr))

    conv = nbody_system.nbody_to_si(1 | units.RSun, 1 | units.yr)
    to_MSun = conv.to_si(1 | nbody_system.mass).value_in(units.MSun)

    a0_nb = conv.to_nbody(a0).number
    afin_nb = conv.to_nbody(afin).number
    m1_0_nb = conv.to_nbody(m1_0).number
    m2_0_nb = conv.to_nbody(m2_0).number
    Period0_nb = conv.to_nbody(Period0).number
    dm1dt_nb = conv.to_nbody(dm1dt).number
    dm2dt_nb = conv.to_nbody(dm2dt).number

    tfin_in_periods = 50
    dtout_in_periods = 0.01

    PertEvolve = PerturbativeEvolution()
    PertEvolve.select_model(isotropic_mass_loss=True)
    PertEvolve.set_mass_loss(dm1dt_nb, dm2dt_nb)

    y0 = [m1_0_nb, m2_0_nb, a0_nb, e0, i0, ome0, Ome0, nu0]

    tfin = Period0_nb * tfin_in_periods
    dt_out = Period0_nb * dtout_in_periods
    t_p, y = PertEvolve.evolve_until(y0, tfin, afin=afin_nb, dt_out=dt_out)
    m1_nb, m2_nb, a_p, e_p, i_p, ome_p, Ome_p, nu_p = y
    m1_p, m2_p = m1_nb * to_MSun, m2_nb * to_MSun

    NbodyEvolve = NbodyEvolution(dtkick=0.05)
    NbodyEvolve.initialize_model(m1_0, m2_0, a0, e0, ome0, nu0, afin=afin)
    NbodyEvolve.set_mass_loss(dm1dt, dm2dt)

    tfin = NbodyEvolve.Period0 * tfin_in_periods
    dtout = NbodyEvolve.Period0 * dtout_in_periods
    t_n, a_n, e_n, ome_n, nu_n, m1_n, m2_n = NbodyEvolve.run_model(tfin, dtout)

    ####
    # PLOTTING
    ####
    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    f, ax = plt.subplots(nrows=5, sharex=True, figsize=(15, 10))

    ax[0].plot(t_p, a_p, lw=4, label="Perturbative", alpha=0.66)
    ax[0].plot(t_n, a_n, lw=4, label="N-body", alpha=0.66)
    ax[0].set_ylabel("semimajor axis [RSun]")
    ax[0].axhline(a0.value_in(units.RSun), c="black", ls="--")
    ax[0].axhline(afin.value_in(units.RSun), c="red", ls="--")

    ax[0].set_yscale("log")
    ax[1].plot(t_p, e_p, lw=2, alpha=0.66, label="Perturbative")
    ax[1].plot(t_n, e_n, lw=2, alpha=0.66, label="N-body")
    ax[1].set_ylabel("eccentricity")
    ax[1].set_xlabel("time [yr]")
    ax[1].set_ylim(bottom=0)

    ax[2].plot(t_p, ome_p, lw=2, alpha=0.66, label="Perturbative")
    ax[2].plot(t_n, ome_n, lw=2, alpha=0.66, label="N-body")
    ax[2].set_ylabel("omega")
    ax[2].set_xlabel("time [yr]")
    #ax[2].set_ylim(bottom=0, top=2 * np.pi)

    ax[3].plot(t_p, nu_p, lw=2, alpha=0.66, label="Perturbative")
    ax[3].plot(t_n, nu_n, lw=2, alpha=0.66, label="N-body")
    ax[3].set_ylabel("nu")
    ax[3].set_ylim(bottom=0, top=2 * np.pi)

    ax[4].plot(t_p, m1_p, lw=2, c="tab:blue", alpha=0.66, label="m1, Perturbative")
    ax[4].plot(t_p, m2_p, lw=2, c="tab:orange", alpha=0.66, label="m2, Perturbative")
    ax[4].plot(t_n, m1_n, lw=2, c="tab:cyan", alpha=0.66, label="m1, N-body")
    ax[4].plot(t_n, m2_n, lw=2, c="tab:red", alpha=0.66, label="m2, N-body")
    ax[4].set_ylabel("mass [MSun]")
    ax[4].set_xlabel("time [yr]")

    for axx in ax:
        axx.grid()
        axx.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.show()

if __name__ == "__main__":
    #test_perturbative_evol()
    #test_nbody_evol()
    test_compare_nbody_perturbative()