import numpy
import os
from amuse.datamodel import Particle, Particles
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite

#class DoubleStar():

def make_binary_star(mprim, msec, semimajor_axis, eccentricity):
    double_star = Particle()
    double_star.is_binary = True
    double_star.mass = mprim + msec
    double_star.semimajor_axis = semimajor_axis
    double_star.eccentricity = eccentricity
    
    stars = new_binary_from_orbital_elements(mprim,
                                             msec,
                                             semimajor_axis,
                                             eccentricity,
                                             G=constants.G)
    stars.is_binary = False
    double_star.child1 = stars[0]
    double_star.child1.name = "primary"
    double_star.child2 = stars[1]
    double_star.child2.name = "secondary"
    
    return double_star, stars

def evolve_model(end_time, double_star, stars):

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    time = 0 | units.yr
    dt = end_time/100.
    a = [] | units.au
    t = [] | units.yr
    while time<end_time:
        time += dt
        gravity.evolve_model(1|units.yr)
        to_stars.copy()
        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)
        a.append(orbital_elements[2])
        t.append(time)
    gravity.stop()
    
    from matplotlib import pyplot
    pyplot.scatter(t.value_in(units.yr), a.value_in(units.au))
    pyplot.show()

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-M", unit=units.MSun, type="float",
                      dest="mprim", default = 15|units.MSun,
                      help="primary mass [%default]")
    result.add_option("-m", unit=units.MSun, type="float",
                      dest="msec", default = 15|units.MSun,
                      help="secondary mass [%default]")
    result.add_option("-a", unit=units.MSun, type="float",
                      dest="semimajor_axis", default = 138|units.RSun,
                      help="semi-major axis [%default]")
    result.add_option("-e", type="float",
                      dest="eccentricity", default = 0.68,
                      help="eccentricity [%default]")
    return result

if __name__ == "__main__":
    o, arguments  = new_option_parser().parse_args()
    double_star, stars = make_binary_star(o.mprim,
                                   o.msec,
                                   o.semimajor_axis,
                                   o.eccentricity)
    end_time = 10|units.yr
    evolve_model(end_time, double_star, stars)
