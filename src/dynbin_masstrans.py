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


def dmdt_acc(dmlossdt):
    alpha = 0.9
    dmacc = (alpha-1)*dmlossdt
    return dmacc[::-1]

def dhdt_masstrans(mass, dmdtloss, dmdtacc, a):
    # BSE-like
    mtot = mass.sum()
    hcirc = (constants.G*mtot*a).sqrt()
    dhdt = mass[1]/mtot*(dmdtloss[0]*mass[1] - dmdtacc[1]*mass[0]) + mass[0]/mtot*(dmdtloss[1]*mass[0] - dmdtacc[0]*mass[1])
    dhdt = dhdt/(mass[0]*mass[1])*hcirc
    return dhdt

def dadt_masstrans(a, e, mass, dmdtacc):
    # BSE-like, masstrans only (no loss)
    mtot = mass.sum()
    dadt = -a*((2-e*e)/mass[1] + (1+e*e)/mtot)*dmdtacc[1]/(1-e*e)
    dadt = dadt - a*((2-e*e)/mass[0] + (1+e*e)/mtot)*dmdtacc[0]/(1-e*e)
    return dadt

def dedt_masstrans(e, mass, dmdtacc):
    # BSE-like
    mtot = mass.sum()
    dedt = -e*dmdtacc[1]*(1/mtot + 0.5/mass[1])
    dedt = dedt -e*dmdtacc[0]*(1/mtot + 0.5/mass[0])
    return dedt

def kick_binary(stars, dt, a, e, dmdtloss, dmdtacc):
    mtot = stars.mass.sum()
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v = vel.length()

    rvec = pos/r
    h = pos.cross(vel)
    h_mag = h.length()
    rdot = (rvec*vel).sum() * rvec
    rdot_mag = rdot.length()
    rdot_vec = rdot/rdot_mag

    vth = vel-rdot
    vth_mag = vth.length()
    vth_vec = vth/vth_mag

    dhdt = dhdt_masstrans(stars.mass, dmdtloss, dmdtacc, a)
    dadt = dadt_masstrans(a, e, stars.mass, dmdtacc)

    vth_acc = dhdt/h_mag * vth_mag
    print(vth_acc)

    rdot_acc = 0.5*dadt*constants.G*mtot/(a*a) - vth_acc*vth_mag
    rdot_acc = rdot_acc/rdot_mag
    print(rdot_acc)

    com_pos = stars.center_of_mass()
    com_vel = stars.center_of_mass_velocity()

    vth_kick = vth_acc * dt * vth_vec
    rdot_kick = vth_acc * dt * rdot_vec

def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr
    dt = 0.5*end_time/1000.

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
        dmdtacc = dmdt_acc(dmdt)

        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)

        dadt = dadt_massloss(atemp, stars.mass, dmdt)
        dedt = dedt_massloss(etemp, stars.mass, dmdt)

        atemp = atemp + dadt*dt
        etemp = etemp + dedt*dt
        a_an.append(atemp)
        e_an.append(etemp)

        kick_binary(stars, dt, orbital_elements[2], orbital_elements[3], dmdt, dmdtacc)

        stars.mass += dmdt * dt
        from_stars.copy()



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
    end_time = 1000.0 | units.yr
    evolve_model(end_time, double_star, stars)


if __name__ == "__main__":
    main()
