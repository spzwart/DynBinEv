#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
import numpy

from dynbin_common import (
    make_binary_star, new_option_parser,
    mass_loss_rate, dadt_masschange, dedt_masschange,
)


def dmdt_acc(dmlossdt):
    alpha = 0.9
    dmacc = (alpha - 1) * dmlossdt
    return dmacc[::-1]

def dhdt_momentum_secular(h, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    dhdt = -h * (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1])
    return dhdt

def dEdt_momentum_secular(h, E, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    mtot = mass.sum()
    dEdt = -(dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1]) * h/(constants.G*mtot) * (-2*E)**1.5
    return dEdt

def h_from_stars(stars):
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    h = pos.cross(vel)
    return h.length()

def E_from_stars(stars):
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v2 = vel.length_squared()

    E = 0.5*v2 - constants.G*stars.mass.sum()/r
    return E

def dadt_momentum_debug(a, e, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    dadt = -a * (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1]) * 2 * (1 - e * e) ** 0.5
    return dadt

def dedt_momentum_debug(a, dadt, e, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    dedt = dadt/a - 2 * (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1])
    dedt = dedt * 0.5*(1-e*e)/e
    return dedt

def dedt_momentum_old(e, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    ome2 = 1 - e * e
    dedt = - (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1]) * ((1 - e * e) ** 0.5 - 1) * (ome2 / e)
    return dedt

def kick_from_accretion(stars, dmdtacc, dt):
    # Assuming dVth1/Vth = -dm1/m1
    stars.move_to_center()

    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity
    r = pos.length()
    v = vel.length()

    rvec = pos / r
    rdot = (rvec * vel).sum() * rvec
    vth = vel - rdot
    vth_mag = vth.length()
    vth_vec = vth / vth_mag

    accth_mag = -vth_mag * dmdtacc / stars.mass
    kick0 = -vth_vec * accth_mag[0] * dt
    kick1 = vth_vec * accth_mag[1] * dt

    stars[0].velocity += kick0
    stars[1].velocity += kick1

def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr
    dt = 0.5 * end_time / 1000. * 2

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    period = (
            2 * numpy.pi
            * (
                    double_star.semimajor_axis * double_star.semimajor_axis * double_star.semimajor_axis
                    / (constants.G * double_star.mass)
            ).sqrt()
    )
    print("Period =", period.as_string_in(units.yr))
    print("Mass loss timestep =", dt)
    print("Steps per period: = {:1.2f}".format(period / dt))

    E_an = [] | (units.km/units.s)**2
    h_an = [] | units.km**2/units.s
    Etemp = -double_star.mass*constants.G / (2*double_star.semimajor_axis)
    htemp = (double_star.mass*constants.G*double_star.semimajor_axis*
             (1-double_star.eccentricity*double_star.eccentricity))**0.5

    E_num = [] | (units.km/units.s)**2
    h_num = [] | units.km**2/units.s
    m1 = [] | units.MSun
    m2 = [] | units.MSun
    t = [] | units.yr
    t_an = [] | units.yr
    dt_an = period*2.5
    time_an = 0 | units.yr
    while time < end_time:
        time += dt
        gravity.evolve_model(time)

        to_stars.copy()

        dmdtloss = mass_loss_rate(stars.mass)
        dmdtacc = dmdt_acc(dmdtloss)

        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)

        h = h_from_stars(stars)
        E = E_from_stars(stars)

        if time > time_an:
            #atemp = orbital_elements[2]
            #htemp = h
            E_an.append(Etemp)
            h_an.append(htemp)
            t_an.append(time)

            dEdt = dEdt_momentum_secular(htemp, Etemp, stars.mass, dmdtacc)
            dhdt = dhdt_momentum_secular(htemp, stars.mass, dmdtacc)
            Etemp = Etemp + dEdt * dt_an
            htemp = htemp + dhdt * dt_an
            E_an.append(Etemp)
            h_an.append(htemp)
            t_an.append(time+dt_an)
            time_an += dt_an

        kick_from_accretion(stars, dmdtacc, dt)

        #stars.mass += (dmdtloss + dmdtacc) * dt
        #stars2.mass += (dmdtloss + dmdtacc) * dt

        from_stars.copy()

        h_num.append(h)
        E_num.append(E)
        m1.append(stars[0].mass)
        m2.append(stars[1].mass)
        t.append(time)
        print("time=", time.in_(units.yr),
              "h=", h.in_(units.km**2/units.s),
              "E=", E.in_((units.km/units.s)**2),
              "m=", stars.mass.in_(units.MSun), end="\r")
    gravity.stop()

    from matplotlib import pyplot
    pyplot.rc('text', usetex=True)
    pyplot.rcParams.update({ 'font.size': 16 })
    fig, axis = pyplot.subplots(nrows=2, ncols=2, sharex=True, figsize=(13, 6))
    axis[0][0].plot(t.value_in(units.yr), E_num.value_in((units.km/units.s)**2), label="nbody (direct)", lw=2)
    axis[0][0].plot(t_an.value_in(units.yr), E_an.value_in((units.km/units.s)**2), label="analytic", lw=2, ls="-.")
    axis[0][0].set_ylabel("E [km/s]")
    axis[0][0].legend()

    axis[0][1].plot(t.value_in(units.yr), m1.value_in(units.MSun), label="m1", lw=2, c="tab:red")
    axis[0][1].plot(t.value_in(units.yr), m2.value_in(units.MSun), label="m2", lw=2, c="tab:cyan")
    axis[0][1].set_ylabel("M [$M_\odot$]")
    axis[0][1].legend()

    axis[1][1].plot(t.value_in(units.yr), h_num.value_in(units.km**2/units.s), lw=2)
    axis[1][1].plot(t_an.value_in(units.yr), h_an.value_in(units.km**2/units.s), lw=2, ls="-.")
    axis[1][1].set_ylabel("h [km$^2$/s]")

    #axis[1][0].plot(t.value_in(units.yr), ome, lw=2, label="nbody (direct)")
    #axis[1][0].set_ylabel("$\omega$ [degrees]")

    axis[1][1].set_xlabel("time [yr]")
    axis[1][0].set_xlabel("time [yr]")

    pyplot.tight_layout()
    pyplot.subplots_adjust(hspace=0, top=0.93, bottom=0.1)

    pyplot.suptitle("mloss+macc+momentum change, same model")
    pyplot.savefig("debug.png")
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
