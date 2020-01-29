#!/usr/bin/env python3
import numpy
from amuse.datamodel import Particle
from amuse.units import units, constants
from amuse.ext.orbital_elements import new_binary_from_orbital_elements


def mass_loss_rate(m):
    dmdt = -(1.e-6 | units.MSun/units.yr) * (m/(1.0 | units.MSun))**2
    return dmdt


def dadt_massloss(a0, m0, dmdt):
    # dmdt is negative for mass loss
    dadt = a0 * -((dmdt[0] + dmdt[1])/(m0[0]+m0[1]))
    return dadt


def dedt_massloss(e0, m0, dmdt):
    dedt = 0 | 1/units.s
    return dedt


def make_binary_star(mprim, msec, semimajor_axis, eccentricity):
    double_star = Particle()
    double_star.is_binary = True
    double_star.mass = mprim + msec
    double_star.semimajor_axis = semimajor_axis
    double_star.eccentricity = eccentricity

    period = (
        2*numpy.pi
        * (
            semimajor_axis*semimajor_axis*semimajor_axis
            / (constants.G*double_star.mass)
        ).sqrt()
    )
    print("Period =", period.as_string_in(units.yr))

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


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-M", unit=units.MSun, type="float",
                      dest="mprim", default=15 | units.MSun,
                      help="primary mass [%default]")
    result.add_option("-m", unit=units.MSun, type="float",
                      dest="msec", default=15 | units.MSun,
                      help="secondary mass [%default]")
    result.add_option("-a", unit=units.RSun, type="float",
                      dest="semimajor_axis", default=10000 | units.RSun,
                      help="semi-major axis [%default]")
    result.add_option("-e", type="float",
                      dest="eccentricity", default=0.68,
                      help="eccentricity [%default]")
    return result
