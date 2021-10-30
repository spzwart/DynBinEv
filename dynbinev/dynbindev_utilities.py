#!/usr/bin/env python3
from amuse.datamodel import Particle
from amuse.units import units, constants
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from enum import IntEnum
import numpy as np

stateid = IntEnum("stateid", "m1 m2 a e i ome Ome nu", start=0)

def mod2pi(f):
    while f < 0:
        f += 2 * np.pi
    while f > 2 * np.pi:
        f -= 2 * np.pi
    return f

def get_period(double_star):
    P = (
        2*np.pi
        * (
            double_star.semimajor_axis*double_star.semimajor_axis*double_star.semimajor_axis
            / (constants.G*double_star.mass)
        ).sqrt()
    )
    return P

def check_collisions(stars):
    pos = stars[1].position - stars[0].position
    r = pos.length()
    sumrad = stars.radius.sum()
    if sumrad > r:
        print("Collided!")
        return True
    else:
        return False

def mass_loss_rate(m):
    dmdt = -(1.e-6 | units.MSun/units.yr) * (m/(1.0 | units.MSun))**2
    return dmdt


def dadt_masschange(a0, m0, dmdt):
    # dmdt is negative for mass loss
    dadt = a0 * -((dmdt[0] + dmdt[1])/(m0[0]+m0[1]))
    return dadt


def dedt_masschange(e0, m0, dmdt):
    dedt = 0 | 1/units.s
    return dedt


def make_binary_star(mprim, msec, semimajor_axis, eccentricity, argument_of_periapsis=360 | units.deg,true_anomaly=0|units.deg):
    double_star = Particle()
    double_star.is_binary = True
    double_star.mass = mprim + msec
    double_star.semimajor_axis = semimajor_axis
    double_star.eccentricity = eccentricity
    double_star.argument_of_periapsis = argument_of_periapsis
    double_star.true_anomaly = true_anomaly

    period = (
        2*np.pi
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
                                             G=constants.G,
                                             argument_of_periapsis=argument_of_periapsis,
                                             true_anomaly=true_anomaly)
    stars.is_binary = False
    double_star.child1 = stars[0]
    double_star.child1.name = "primary"

    double_star.child2 = stars[1]
    double_star.child2.name = "secondary"

    for star in stars:
        star.radius = (star.mass.value_in(units.MSun) ** 0.8) | units.RSun

    return double_star, stars


def new_option_parser(a_default=10000|units.RSun, e_default=0.68,
                      m_default=15|units.MSun, M_default=15|units.MSun):
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-M", unit=units.MSun, type="float",
                      dest="mprim", default=M_default,
                      help="primary mass [%default]")
    result.add_option("-m", unit=units.MSun, type="float",
                      dest="msec", default=m_default,
                      help="secondary mass [%default]")
    result.add_option("-a", unit=units.RSun, type="float",
                      dest="semimajor_axis", default=a_default,
                      help="semi-major axis [%default]")
    result.add_option("-e", type="float",
                      dest="eccentricity", default=e_default,
                      help="eccentricity [%default]")
    result.add_option("-k", type="float", nargs=2,
                      dest="kaps", default=[0.14, 0.14],
                      help="apsidal constant [%default]")
    result.add_option("--tau", type="float", nargs=2,
                      dest="taulag", default=[100, 100] | units.s, unit=units.s,
                      help="timelag in seconds [%default]")
    return result
