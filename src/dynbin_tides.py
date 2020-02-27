#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
from tides_analytic import QuickTides
import numpy

from dynbin_common import (
    make_binary_star, new_option_parser,
    mass_loss_rate, dadt_masschange, dedt_masschange,
)

def check_collision(stars):
    pos = stars[1].position - stars[0].position
    r = pos.length()
    if r <= stars[1].radius + stars[0].radius:
        return True
    else: return False

def kick_stars_tides(stars, dt):
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    rvec = pos / r
    rdot = (rvec * vel).sum() * rvec
    rdot_mag = rdot.length()

    inv_r = 1.0 / r
    inv_r_7 = inv_r**7
    # k = 0
    # j = 1
    r0_5 = stars[0].radius**5
    r1_5 = stars[1].radius**5
    m0_2 = stars[0].mass*stars[0].mass
    m1_2 = stars[1].mass*stars[1].mass

    ftr = -3.0 * inv_r_7 * constants.G * ((m1_2 * r0_5 * stars[0].kaps + m0_2 * r1_5 * stars[1].kaps) +    # Non-dissipative
                3 * inv_r * rdot_mag *
               (m1_2 * r0_5 * stars[0].kaps * stars[0].taulag + m0_2 * r1_5 * stars[1].kaps * stars[1].taulag))  # Dissipative

    hutforce = ftr * inv_r * pos

    acc0 = (1.0 / stars[0].mass) * hutforce
    acc1 = (-1.0 / stars[1].mass) * hutforce

    kick0 = acc0 * dt
    kick1 = acc1 * dt

    stars[0].velocity += kick0
    stars[1].velocity += kick1

def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    period = (
        2*numpy.pi
        * (
            double_star.semimajor_axis*double_star.semimajor_axis*double_star.semimajor_axis
            / (constants.G*double_star.mass)
        ).sqrt()
    )
    print("Period =", period.as_string_in(units.yr))

    dt = period/50.

    print("Mass loss timestep =", dt)
    print("Steps per period: = {:1.2f}".format(period/dt))

    print("Radii", stars.radius)
    print("Taulag", stars.taulag)
    print("K", stars.kaps)

    QT = QuickTides(double_star.semimajor_axis, double_star.eccentricity,
                    stars[0].mass, stars[1].mass,
                    stars[0].radius, stars[1].radius,
                    stars[0].kaps, stars[1].kaps,
                    stars[0].taulag, stars[1].taulag)

    a_an = [] | units.au
    e_an = []
    atemp = double_star.semimajor_axis
    etemp = double_star.eccentricity

    a = [] | units.au
    e = [] 
    m = [] | units.MSun
    t = [] | units.yr
    while time<end_time:
        time += dt
        gravity.evolve_model(time)
        to_stars.copy()

        dadt, dedt = QT.dadt_dedt(atemp, etemp)

        atemp = atemp + dadt*dt
        etemp = etemp + dedt*dt
        a_an.append(atemp)
        e_an.append(etemp)

        kick_stars_tides(stars, dt)

        from_stars.copy()
        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)

        a.append(orbital_elements[2])
        e.append(orbital_elements[3])
        m.append(stars.mass.sum())
        t.append(time)

        if check_collision(stars): break

        print("time=", time.in_(units.yr),
              "a=", a[-1].in_(units.RSun),
              "e=", e[-1],
              "m=", stars.mass.in_(units.MSun), end="\r")
    gravity.stop()
    from matplotlib import pyplot
    fig, axis = pyplot.subplots(nrows=2, ncols=2, sharex=True)
    axis[0][0].plot(t.value_in(units.yr), a.value_in(units.RSun), label="nbody")
    axis[0][0].plot(t.value_in(units.yr), a_an.value_in(units.RSun), label="analytic")
    axis[0][0].set_ylabel("a [$R_\odot$]")
    axis[0][0].legend()

    axis[0][1].plot(t.value_in(units.yr), m.value_in(units.MSun))
    axis[0][1].set_ylabel("M [$M_\odot$]")

    axis[1][1].plot(t.value_in(units.yr), e)
    axis[1][1].plot(t.value_in(units.yr), e_an)
    axis[1][1].set_ylabel("e")

    axis[1][1].set_xlabel("time [yr]")
    axis[1][0].set_xlabel("time [yr]")
    pyplot.savefig("mloss.png")
    pyplot.show()


def main():
    o, arguments = new_option_parser().parse_args()
    double_star, stars = make_binary_star(
        o.mprim, o.msec, o.semimajor_axis, o.eccentricity,
    )
    stars.kaps = o.kaps
    stars.taulag = o.taulag

    end_time = 20.0 | units.yr
    evolve_model(end_time, double_star, stars)


if __name__ == "__main__":
    main()
