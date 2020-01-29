#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
import numpy

from dynbin_common import (
    make_binary_star, new_option_parser,
    mass_loss_rate, dadt_massloss, dedt_massloss,
)


def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr
    dt = 0.5*end_time/10000.

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
    print("Mass loss timestep =", dt)
    print("Steps per period: = {:1.2f}".format(period/dt))

    a_an = [] | units.au
    e_an = []
    atemp = double_star.semimajor_axis
    etemp = double_star.eccentricity
    print(atemp)

    a = [] | units.au
    e = [] 
    m = [] | units.MSun
    t = [] | units.yr
    while time<end_time:
        time += dt
        gravity.evolve_model(time)
        to_stars.copy()

        dmdt = mass_loss_rate(stars.mass)

        dadt = dadt_massloss(atemp, stars.mass, dmdt)
        dedt = dedt_massloss(etemp, stars.mass, dmdt)

        atemp = atemp + dadt*dt
        etemp = etemp + dedt*dt
        a_an.append(atemp)
        e_an.append(etemp)

        stars.mass -= dmdt * dt
        from_stars.copy()
        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)


        a.append(orbital_elements[2])
        e.append(orbital_elements[3])
        m.append(stars.mass.sum())
        t.append(time)
        print("time=", time.in_(units.yr),
              "a=", a[-1].in_(units.RSun),
              "e=", e[-1],
              "m=", stars.mass.in_(units.MSun))
    gravity.stop()
    from matplotlib import pyplot
    fig, axis = pyplot.subplots(nrows=2, ncols=2, sharex=True)
    axis[0][0].plot(t.value_in(units.yr), a.value_in(units.RSun))
    axis[0][0].plot(t.value_in(units.yr), a_an.value_in(units.RSun))
    axis[0][0].set_ylabel("a [$R_\odot$]")

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
    end_time = 1000.0 | units.yr
    evolve_model(end_time, double_star, stars)


if __name__ == "__main__":
    main()
