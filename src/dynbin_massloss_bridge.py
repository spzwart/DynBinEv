import numpy
import os
from amuse.couple import bridge
from amuse.datamodel import Particle, Particles
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite

class CodeWithMassLoss(bridge.GravityCodeInField):
    #def drift(self, tend):
    #    dt = tend-self.time
    #    #dt = timestep
    #    dmdt = mass_loss_rate(self.particles.mass)
    #    self.particles.mass -= dmdt*dt
    #    pass
    def kick(self, dt):
        kinetic_energy_before = self.particles.kinetic_energy()
        dmdt = mass_loss_rate(self.particles.mass)
        self.particles.mass -= dmdt*dt
        kinetic_energy_after = self.particles.kinetic_energy()
        return kinetic_energy_after - kinetic_energy_before

def mass_loss_rate(m):
    dmdt = (1.e-6 | units.MSun/units.yr) * (m/(1.0|units.MSun))**2
    return dmdt

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
    time = 0 | units.yr
    dt = end_time/100.

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    massloss_code = CodeWithMassLoss(gravity, ())
    gravml = bridge.Bridge(use_threading=False)
    gravml.timestep = 0.5*dt
    gravml.add_system(gravity,)
    gravml.add_code(massloss_code)
    
    a = [] | units.au
    e = [] 
    m = [] | units.MSun
    t = [] | units.yr
    while time<end_time:
        time += dt
        gravml.evolve_model(time)
        to_stars.copy()
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
    fig, axis = pyplot.subplots(nrows=2,ncols=2, sharex=True)
    axis[0][0].scatter(t.value_in(units.yr), a.value_in(units.RSun))
    axis[0][0].set_ylabel("a [$R_\odot$]")

    axis[0][1].scatter(t.value_in(units.yr), m.value_in(units.MSun))
    axis[0][1].set_ylabel("M [$M_\odot$]")

    axis[1][1].scatter(t.value_in(units.yr), e)
    axis[1][1].set_ylabel("e")

    axis[1][1].set_xlabel("time [yr]")
    axis[1][0].set_xlabel("time [yr]")
    pyplot.savefig("mloss_bridge.png")
    pyplot.show()

def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()
    result.add_option("-M", unit=units.MSun, type="float",
                      dest="mprim", default = 15|units.MSun,
                      help="primary mass [%default]")
    result.add_option("-m", unit=units.MSun, type="float",
                      dest="msec", default = 8|units.MSun,
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
    end_time = 1000.0|units.yr
    evolve_model(end_time, double_star, stars)
