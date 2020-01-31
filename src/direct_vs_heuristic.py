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


def dadt_masstrans_bse(a, e, mass, dmdtacc):
    # BSE-like, masstrans only (no loss)
    mtot = mass.sum()
    dadt = -a * ((2 - e * e) / mass[1] + (1 + e * e) / mtot) * dmdtacc[1] / (1 - e * e)
    dadt = dadt - a * ((2 - e * e) / mass[0] + (1 + e * e) / mtot) * dmdtacc[0] / (1 - e * e)
    return dadt


def dedt_masstrans_bse(e, mass, dmdtacc):
    # BSE-like
    mtot = mass.sum()
    dedt = -e * dmdtacc[1] * (1 / mtot + 0.5 / mass[1])
    dedt = dedt - e * dmdtacc[0] * (1 / mtot + 0.5 / mass[0])
    return dedt


def dhdt_momentumchange(h, mass, dmdtacc):
    dhdt = -h * (dmdtacc[1] / mass[1] + dmdtacc[0] / mass[0])
    return dhdt


def dadt_momentumchange(a, e, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    dadt = - a * (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1]) * 2 * (1 - e * e) ** 0.5
    return dadt


def dedt_momentumchange(e, mass, dmdtacc):
    # Assuming dVth1/V = -dm1/m1
    ome2 = 1 - e * e
    dedt = - (dmdtacc[0] / mass[0] + dmdtacc[1] / mass[1]) * ((1 - e * e) ** 0.5 - 1) * (ome2 / e)
    return dedt


def kick_from_accretion(stars, dmdtacc, dt):
    # Assuming dVth1/V = -dm1/m1
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

    accth_mag = -v * dmdtacc / stars.mass
    kick0 = -vth_vec * accth_mag[0] * dt
    kick1 = vth_vec * accth_mag[1] * dt

    stars[0].velocity += kick0
    stars[1].velocity += kick1


def dhdt_dadt_to_kick(stars, dhdt, dadt, dmdt, dt):
    stars.move_to_center()
    mtot = stars.mass.sum()
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v2 = vel.length_squared()

    vcirc2 = constants.G * mtot / r
    a = - constants.G * mtot / (v2 - 2. * vcirc2)

    rvec = pos / r
    rdot = (rvec * vel).sum() * rvec
    rdot_mag = rdot.length()
    rdot_vec = rdot / rdot_mag

    vth = vel - rdot
    vth_mag = vth.length()
    vth_vec = vth / vth_mag

    vth_acc = dhdt / r

    # removing contribution from mass loss
    dadt_corrected = dadt + a * (dmdt[0] + dmdt[1]) / mtot

    rdot_acc = 0.5 * dadt_corrected * constants.G * mtot / (a * a) - vth_acc * vth_mag
    rdot_acc = rdot_acc / rdot_mag

    vth_kick = vth_acc * dt * vth_vec
    rdot_kick = rdot_acc * dt * rdot_vec

    stars[1].velocity += vth_kick #+ rdot_kick


def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr
    dt = 0.5 * end_time / 1000. * 2

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    stars2 = stars.copy()

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    gravity2 = Hermite(converter)
    gravity2.particles.add_particle(stars2)
    to_stars2 = gravity2.particles.new_channel_to(stars2)
    from_stars2 = stars2.new_channel_to(gravity2.particles)

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

    a_an = [] | units.au
    e_an = []
    atemp = double_star.semimajor_axis
    etemp = double_star.eccentricity

    a = [] | units.au
    e = []
    ome = []
    a2 = [] | units.au
    e2 = []
    ome2 = []
    m1 = [] | units.MSun
    m2 = [] | units.MSun
    t = [] | units.yr
    while time < end_time:
        time += dt
        gravity.evolve_model(time)
        gravity2.evolve_model(time)
        to_stars.copy()
        to_stars2.copy()

        dmdtloss = mass_loss_rate(stars.mass)
        dmdtacc = dmdt_acc(dmdtloss)

        orbital_elements = orbital_elements_from_binary(stars,
                                                        G=constants.G)
        orbital_elements2 = orbital_elements_from_binary(stars2,
                                                         G=constants.G)

        dadt = dadt_masschange(atemp, stars.mass, dmdtloss + dmdtacc)
        dedt = dedt_masschange(etemp, stars.mass, dmdtloss + dmdtacc)
        dadt += dadt_momentumchange(atemp, etemp, stars.mass, dmdtacc)
        dedt += dedt_momentumchange(etemp, stars.mass, dmdtacc)

        h = (constants.G * stars.mass.sum() * atemp * (1 - etemp * etemp)) ** 0.5
        dhdt = dhdt_momentumchange(h, stars.mass, dmdtacc)

        atemp = atemp + dadt * dt
        etemp = etemp + dedt * dt
        a_an.append(atemp)
        e_an.append(etemp)

        kick_from_accretion(stars, dmdtacc, dt)

        dhdt_dadt_to_kick(stars2, dhdt, dadt, dmdtloss + dmdtacc, dt)

        stars.mass += (dmdtloss + dmdtacc) * dt
        stars2.mass += (dmdtloss + dmdtacc) * dt

        from_stars.copy()
        from_stars2.copy()

        a.append(orbital_elements[2])
        e.append(orbital_elements[3])
        ome.append(orbital_elements[7])
        a2.append(orbital_elements2[2])
        e2.append(orbital_elements2[3])
        ome2.append(orbital_elements2[7])
        m1.append(stars[0].mass)
        m2.append(stars[1].mass)
        t.append(time)
        print("time=", time.in_(units.yr),
              "a=", a[-1].in_(units.RSun),
              "e=", e[-1],
              "m=", stars.mass.in_(units.MSun), end="\r")
    gravity.stop()
    gravity2.stop()

    from matplotlib import pyplot
    pyplot.rc('text', usetex=True)
    pyplot.rcParams.update({ 'font.size': 16 })
    fig, axis = pyplot.subplots(nrows=2, ncols=2, sharex=True, figsize=(13, 6))
    axis[0][0].plot(t.value_in(units.yr), a.value_in(units.RSun), label="nbody (direct)", lw=2)
    axis[0][0].plot(t.value_in(units.yr), a2.value_in(units.RSun), label="nbody (heuristic)", lw=2, ls="--")
    axis[0][0].plot(t.value_in(units.yr), a_an.value_in(units.RSun), label="analytic", lw=2, ls="-.")
    axis[0][0].set_ylabel("a [$R_\odot$]")
    axis[0][0].legend()

    axis[0][1].plot(t.value_in(units.yr), m1.value_in(units.MSun), label="m1", lw=2, c="tab:red")
    axis[0][1].plot(t.value_in(units.yr), m2.value_in(units.MSun), label="m2", lw=2, c="tab:cyan")
    axis[0][1].set_ylabel("M [$M_\odot$]")
    axis[0][1].legend()

    axis[1][1].plot(t.value_in(units.yr), e, lw=2)
    axis[1][1].plot(t.value_in(units.yr), e2, lw=2, ls="--")
    axis[1][1].plot(t.value_in(units.yr), e_an, lw=2, ls="-.")
    axis[1][1].set_ylabel("e")

    axis[1][0].plot(t.value_in(units.yr), ome, lw=2, label="nbody (direct)")
    axis[1][0].plot(t.value_in(units.yr), ome2, lw=2, label="nbody (heuristic)")
    axis[1][0].set_ylabel("$\omega$ [degrees]")

    axis[1][1].set_xlabel("time [yr]")
    axis[1][0].set_xlabel("time [yr]")

    pyplot.tight_layout()
    pyplot.subplots_adjust(hspace=0, top=0.93, bottom=0.1)

    pyplot.suptitle("mloss+macc+momentum change, same model")
    pyplot.savefig("comparisons.png")
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
