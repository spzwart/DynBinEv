#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite

from dynbin_common import make_binary_star, new_option_parser


def evolve_model(end_time, double_star, stars):
    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    # from_stars = stars.new_channel_to(gravity.particles)

    time = 0 | units.yr
    dt = end_time/100.
    a = [] | units.au
    t = [] | units.yr
    while time < end_time:
        time += dt
        gravity.evolve_model(1 | units.yr)
        to_stars.copy()
        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)
        a.append(orbital_elements[2])
        t.append(time)
    gravity.stop()

    from matplotlib import pyplot
    pyplot.scatter(t.value_in(units.yr), a.value_in(units.au))
    pyplot.show()


def main():
    o, arguments = new_option_parser().parse_args()
    double_star, stars = make_binary_star(
        o.mprim, o.msec, o.semimajor_axis, o.eccentricity,
    )
    end_time = 10 | units.yr
    evolve_model(end_time, double_star, stars)


if __name__ == "__main__":
    main()
