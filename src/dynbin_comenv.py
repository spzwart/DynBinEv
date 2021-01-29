#!/usr/bin/env python3
from amuse.units import units, constants, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.ext.orbital_elements import orbital_elements_from_binary
from amuse.community.hermite.interface import Hermite
import numpy

from dynbin_common import (
    make_binary_star, new_option_parser,
    mass_loss_rate, dadt_masschange, dedt_masschange, get_period
)


def dadt_comenv_k2(a, e, C):
    e2 = e * e
    da_dt = C / a * 2 * (1 + e2) / (1 - e2) ** 1.5
    return da_dt


def dedt_comenv_k2(a, e, C):
    e2 = e * e
    de_dt = C / (a * a) * 2 * e / (1 - e2) ** 0.5
    return de_dt

def dadt_comenv_k0(a, e, C):
    da_dt = C * a * 2
    return da_dt


def dedt_comenv_k0(a, e, C):
    de_dt = 0.0 * C
    return de_dt

def kick_stars_comenv(stars, dt, C):
    stars.move_to_center()
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v = vel.length()
    vvec = vel / v

    acc = C / (r * r) * v * vvec
    mtot = stars[0].mass + stars[1].mass

    stars[0].velocity += -acc*dt * stars[1].mass/mtot
    stars[1].velocity += acc*dt * stars[0].mass/mtot

def kick_stars_comenv2(stars, dt, C, A):
    stars.move_to_center()
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v = vel.length()
    vvec = vel / v

    acc = C / A * v * vvec
    mtot = stars[0].mass + stars[1].mass

    stars[0].velocity += -acc*dt * stars[1].mass/mtot
    stars[1].velocity += acc*dt * stars[0].mass/mtot


def kick_stars_comenv3(stars, dt, C):
    stars.move_to_center()
    pos = stars[1].position - stars[0].position
    vel = stars[1].velocity - stars[0].velocity

    r = pos.length()
    v = vel.length()
    vvec = vel / v

    acc = -C * v * v * vvec
    mtot = stars[0].mass + stars[1].mass

    stars[0].velocity += -acc*dt * stars[1].mass/mtot
    stars[1].velocity += acc*dt * stars[0].mass/mtot

def C_from_eps(eps0, eps_ce, Tce, mu):
    epsf = eps0 + eps_ce
    C = mu**2 * (1/epsf**2 -  1/eps0**2) / (16*Tce)
    return C


def check_collisions(stars):
    pos = stars[1].position - stars[0].position
    r = pos.length()
    sumrad = stars.radius.sum()
    if sumrad > r:
        print("Collided!")
        return True
    else: return False

def evolve_model(end_time, double_star, stars):
    time = 0 | units.yr
    dt = 0.01*end_time/1000.

    converter = nbody_system.nbody_to_si(double_star.mass,
                                         double_star.semimajor_axis)

    gravity = Hermite(converter)
    gravity.particles.add_particle(stars)
    to_stars = gravity.particles.new_channel_to(stars)
    from_stars = stars.new_channel_to(gravity.particles)

    period = get_period(double_star)
    print("Period =", period.as_string_in(units.yr))
    print("Mass loss timestep =", dt)
    print("Steps per period: = {:1.2f}".format(period/dt))

    a_an = [] | units.au
    e_an = []
    atemp = double_star.semimajor_axis
    etemp = double_star.eccentricity

    ###### COMMON ENVELOPE STUFF ###############
    final_a = 40 | units.RSun

    mu = double_star.mass * constants.G
    Eps0 = mu / (2 * double_star.semimajor_axis)
    Eps1 = mu / (2 * final_a)

    # Eps_ce should come from alpha lambda model, but we just fix the final semimajor axis here for simplicity
    Eps_ce = Eps1 - Eps0
    print("Eps_ce/Eps0", Eps_ce / Eps0)

    Tce = 1000 | units.yr
    C_ce = C_from_eps(Eps0, Eps_ce, Tce, mu)
    print("C_ce", C_ce)
    Avisc = -C_ce * Tce
    print("Avisc", Avisc.as_string_in(units.RSun ** 2))
    Rvisc = Avisc.sqrt() / (4 * constants.pi)
    print("Rvisc", Rvisc.as_string_in(units.RSun))

    vorb = (mu / double_star.semimajor_axis).sqrt()

    #### l=2,k=0
    Cfact_l2k0 = 1e-5 | 1/units.RSun
    print("C:", Cfact_l2k0.as_string_in(1/units.RSun))

    ###### END COMMON ENVELOPE STUFF ###############

    collision = False
    a = [] | units.au
    e = [] 
    m = [] | units.MSun
    t = [] | units.yr
    while time < end_time:
        time += dt
        if not collision:
            gravity.evolve_model(time)
            to_stars.copy()
            kick_stars_comenv3(stars, dt, C=Cfact_l2k0)
            from_stars.copy()

            from_stars.copy()

            orbital_elements = orbital_elements_from_binary(stars,
                                                            G=constants.G)

            collision = check_collisions(stars)

        if atemp.number > 0:
            dadt = dadt_comenv_k0(atemp, etemp, C_ce/Avisc)
            dedt = dedt_comenv_k0(atemp, etemp, C_ce/Avisc)

            atemp = atemp + dadt*dt
            etemp = etemp + dedt*dt

        if collision and atemp.number < 0: break

        if orbital_elements[2] < final_a: break

        a_an.append(atemp)
        e_an.append(etemp)
        a.append(orbital_elements[2])
        e.append(orbital_elements[3])
        m.append(stars.mass.sum())
        t.append(time)
        print("time=", time.in_(units.yr),
              "a=", a[-1].in_(units.RSun),
              "e=", e[-1],
              "m=", stars.mass.in_(units.MSun), end="\r")

    gravity.stop()
    from matplotlib import pyplot
    import seaborn as sns
    sns.set(font_scale=1.33)
    sns.set_style("ticks")

    fig, axis = pyplot.subplots(nrows=2, sharex=True)
    axis[0].plot(t.value_in(units.yr), a.value_in(units.RSun), label="nbody k=0")
    axis[0].plot(t.value_in(units.yr), a_an.value_in(units.RSun), label="analytic")
    axis[0].set_ylabel("semimajor axis [$R_\odot$]")
    axis[0].set_yscale("log")
    axis[0].legend()

    axis[1].plot(t.value_in(units.yr), e)
    axis[1].plot(t.value_in(units.yr), e_an)
    axis[1].set_ylabel("eccentricity")

    axis[1].set_xlabel("time [yr]")
    axis[0].set_xlabel("time [yr]")

    pyplot.tight_layout()
    pyplot.subplots_adjust(hspace=0.0)
    pyplot.savefig("comenv2.png")
    pyplot.show()


def main():
    #m1, m2 = 80 | units.MSun, 55 | units.MSun
    m1, m2 = 15 | units.MSun, 15 | units.MSun
    a0 = 4000 | units.RSun
    e0 = 0.4

    o, arguments = new_option_parser(M_default=m1, m_default=m2,
                                     a_default=a0, e_default=e0).parse_args()
    double_star, stars = make_binary_star(
        o.mprim, o.msec, o.semimajor_axis, o.eccentricity,
    )
    end_time = 1000.0 | units.yr
    evolve_model(end_time, double_star, stars)


if __name__ == "__main__":
    main()
