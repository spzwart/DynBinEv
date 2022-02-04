#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
import numpy as np

from dynbin_common import make_binary_star, new_option_parser, get_period


def mod2pi(f):
    while f < 0:
        f += 2 * np.pi
    while f > 2 * np.pi:
        f -= 2 * np.pi
    return f


def check_collisions(stars):
    pos = stars[1].position - stars[0].position
    r = pos.length()
    sumrad = stars.radius.sum()
    if sumrad > r:
        print("Collided!")
        return True
    else:
        return False


# KickStars function
def kick_star(
    dt,
    list_of_stellar_pairs=[],
    l=2,
    k=0,
):
    c = some_constant_with_unit
    Cunits = nbody_system.length**(1 - l + k) * nbody_system.time**(l - 2)
    return
    
def find_stellar_pairs():
    # use AMUSE's find binaries function

    return

class MacegaKick():
    def __init__(
            self,
            l=2,
            k=0,
            dtkick=0.01,
            dtkick_update=True,
    ):
        """
        Initi
        :param l: exponent of velocity in the drag force
        :param k: exponent of radius in drag force
        :param dtkick: timestep between kicks, as fraction of binary periods
        :param dtkick_update: dtkick as the binary shrinks, or keep the
        timestep constant
        """
        self.l = l
        self.k = k

        self.select_model(l, k)

        self.length_unit = 1 * units.RSun
        self.time_unit = 1 * units.yr
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)
        self.mass_unit = self.conv.to_si(1 | nbody_system.mass)
        print("Mass units:", self.mass_unit.as_string_in(units.MSun))
        self.dtkick = dtkick
        self.dtkick_update = dtkick_update

        Cunits = nbody_system.length**(1 - l + k) * nbody_system.time**(l - 2)
        self.Cunits = self.conv.to_si(Cunits).as_unit()
        print("C units:", Cunits)

    def select_model(self, l, k):
        """
        Select model, so far only l=1,2 k=0 are added
        :param l: exponent of velocity in the drag force
        :param k: exponent of radius in drag force
        :return:
        """
        if k == 0 and l == 2:
            self.kick_stars = self.kick_stars_l2k0
        elif k == 0 and l == 1:
            self.kick_stars = self.kick_stars_l1k0
        else:
            raise ValueError(f"forceform option l={l}, k={k} not found")

    def kick_stars_l2k0(self, stars, dt, C):
        # stars.move_to_center()
        # vcom = stars.center_of_mass_velocity()
        # pos = stars[1].position - stars[0].position
        vel = stars[1].velocity - stars[0].velocity

        # r = pos.length()
        v = vel.length()
        vvec = vel / v

        acc = (-C | self.Cunits) * v * v * vvec
        mtot = stars[0].mass + stars[1].mass

        stars[0].velocity += -acc * dt * stars[1].mass / mtot
        stars[1].velocity += acc * dt * stars[0].mass / mtot

    def kick_stars_l1k0(self, stars, dt, C):
        # stars.move_to_center()
        # vcom = stars.center_of_mass_velocity()
        # pos = stars[1].position - stars[0].position
        vel = stars[1].velocity - stars[0].velocity

        # r = pos.length()
        v = vel.length()
        vvec = vel / v

        acc = -C * v * vvec
        mtot = stars[0].mass + stars[1].mass

        stars[0].velocity += -acc * dt * stars[1].mass / mtot
        stars[1].velocity += acc * dt * stars[0].mass / mtot

    def initialize_model(self, m1, m2, a0, e0, ome0, nu0, afin, C):
        """
        Initializes the kick module
        :param m1: mass star 1
        :param m2: mass star 2
        :param a0: semimajor axis
        :param e0: eccentricity
        :param ome0: argument of pericenter
        :param nu0: phase
        :param afin: final semimajor axis (to stop the simulation)
        :param C: coefficient of the drag force, should have correct
        dimensionality depending on k,l
        :return:
        """

        double_star, stars = make_binary_star(
            m1, m2, a0, e0,
            argument_of_periapsis=ome0 | units.rad,
            true_anomaly=nu0 | units.rad
        )

        self.stars = stars
        self.gravity = Hermite(self.conv)
        self.gravity.particles.add_particle(self.stars)
        self.to_stars = self.gravity.particles.new_channel_to(self.stars)
        self.from_stars = self.stars.new_channel_to(self.gravity.particles)

        self.Period0 = get_period(double_star)
        print("Period =", self.Period0.as_string_in(units.yr))
        print("Steps per period: = {:1.2f}".format(self.dtkick))

        mu = double_star.mass * constants.G
        Eps0 = mu / (2 * double_star.semimajor_axis)
        Eps1 = mu / (2 * afin)

        # Eps_ce should come from alpha lambda model, but we just fix the final
        # semimajor axis here for simplicity
        Eps_ce = Eps1 - Eps0
        print("Eps_ce/Eps0", Eps_ce / Eps0)

        # Tce = 1000 | units.yr
        # vorb = (mu / double_star.semimajor_axis).sqrt()

        self.afin = afin
        self.C = C
        self.double_star = double_star

        print("C:", self.C)

    def run_model(
        self, tfin, dt_out, tstart=0 | units.yr,
        check_collisions=False  # FIXME
    ):
        time = tstart
        dt = self.Period0 * self.dtkick
        # dtout_next = time + dt_out

        a = [self.double_star.semimajor_axis.value_in(self.length_unit)]
        e = [self.double_star.eccentricity]
        ome = [self.double_star.argument_of_periapsis.value_in(units.rad)]
        nu = [self.double_star.true_anomaly.value_in(units.rad)]
        t = [tstart.value_in(self.time_unit)]
        while time < tfin:
            time += dt
            self.gravity.evolve_model(time)
            self.to_stars.copy()
            self.kick_stars(self.stars, dt, C=self.C)
            self.from_stars.copy()

            orbital_elements = orbital_elements_from_binary(
                self.stars,
                G=constants.G
            )

            if check_collisions:
                collision = check_collisions(self.stars)
                if collision:
                    break
            if orbital_elements[2] < self.afin:
                break

            if self.dtkick_update is True:
                Period = 2 * np.pi * (
                    orbital_elements[2] * orbital_elements[2]
                    * orbital_elements[2]
                    / (constants.G * self.stars.mass.sum())
                ).sqrt()
                dt = Period*self.dtkick

            a.append(orbital_elements[2].value_in(self.length_unit))
            e.append(orbital_elements[3])
            ome.append(mod2pi(np.radians(orbital_elements[7])))
            nu.append(mod2pi(np.radians(orbital_elements[4])))
            t.append(time.value_in(self.time_unit))
            print("time=", time.value_in(self.time_unit),
                  "a=", a[-1],
                  "e=", e[-1],
                  "m=", self.stars.mass.in_(units.MSun), end="\r")

        self.gravity.stop()

        return t, a, e, ome, nu


def main():
    """
    Tests the module and plots the evolution of the orbital parameters
    :return:
    """
    m1, m2 = 15 | units.MSun, 15 | units.MSun
    a0 = 4000 | units.RSun
    a1 = 40 | units.RSun
    e0 = 0.4
    ome0 = 0.0
    nu0 = np.pi
    C = 1e-5

    o, arguments = new_option_parser(M_default=m1, m_default=m2,
                                     a_default=a0, e_default=e0).parse_args()

    CEKickEvolve = MacegaKick(l=2, k=0, dtkick=0.05)
    CEKickEvolve.initialize_model(
        o.mprim, o.msec, o.semimajor_axis, o.eccentricity, ome0, nu0, afin=a1,
        C=C
    )

    tfin = CEKickEvolve.Period0*200
    dtout = CEKickEvolve.Period0*0.01
    t, a, e, ome, nu = CEKickEvolve.run_model(tfin, dtout)

    from matplotlib import pyplot
    import seaborn as sns
    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    fig, axis = pyplot.subplots(nrows=3, sharex=True)
    axis[0].plot(t, a, label="nbody k=0")
    axis[0].set_ylabel("semimajor axis [$R_\odot$]")
    axis[0].set_yscale("log")
    axis[0].legend()

    axis[1].plot(t, e)
    axis[1].set_ylabel("eccentricity")

    axis[1].set_xlabel("time [yr]")
    axis[0].set_xlabel("time [yr]")

    axis[2].plot(t, ome)
    axis[2].set_ylabel("omega")

    axis[2].set_xlabel("time [yr]")

    pyplot.tight_layout()
    pyplot.subplots_adjust(hspace=0.0)
    pyplot.show()


if __name__ == "__main__":
    main()
