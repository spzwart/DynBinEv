from amuse.community.bse.interface import BSE
from amuse.community.sse.interface import SSE
from amuse.community.hermite.interface import Hermite

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles, Particle, quantities
import numpy as np
import matplotlib.pyplot as plt
from dynbin_common import make_binary_star


def test_binary_evolution(a0=0.3|units.au, e0=0.6, m1=60|units.MSun, m2=40|units.MSun, t0=0 | units.yr,
                          tend=3|units.Myr, dt_out=1000|units.yr, dt_kick_fP=0.1):

    ###################
    ###     BSE     ###
    ###################
    bse = BSE()
    bse.parameters.metallicity = 0.02
    bse.parameters.neutron_star_mass_flag = 3

    stars_bse = Particles(2)
    stars_bse[0].mass = m1
    stars_bse[1].mass = m2

    mu = stars_bse.mass.sum() * constants.G
    P0 = np.pi * (a0*a0*a0/mu).sqrt()
    print("P0:", P0.as_string_in(units.yr))
    print("a0:", a0.as_string_in(units.au))
    print("e0:", e0)

    binaries = Particles(1)
    binary = binaries[0]
    binary.semi_major_axis = a0
    binary.eccentricity = e0
    binary.child1 = stars_bse[0]
    binary.child2 = stars_bse[1]

    bse.particles.add_particles(stars_bse)
    bse.binaries.add_particles(binaries)

    from_bse_to_model = bse.particles.new_channel_to(stars_bse)
    from_bse_to_model.copy()
    from_bse_to_model_binaries = bse.binaries.new_channel_to(binaries)
    from_bse_to_model_binaries.copy()


    ###################
    ###    NBODY    ###
    ###################
    converter = nbody_system.nbody_to_si(1 | units.MSun,
                                         1 | units.au)
    directcode = Hermite(converter)
    binary_nbd, stars_nbd = make_binary_star(m1, m2, a0, e0)
    directcode.particles.add_particle(stars_nbd)
    from_nbody_to_model = directcode.particles.new_channel_to(stars_nbd)
    from_model_to_nbody = stars_nbd.new_channel_to(directcode.particles)

    ##################
    ### BEGIN LOOP ###
    ##################
    t = t0
    t_next_out = t0 + dt_out
    dt_kick = P0 * dt_kick_fP*10000000
    data = []
    while t < tend:
        t = t + dt_kick
        bse.evolve_model(t)

        from_bse_to_model_binaries.copy()
        from_bse_to_model.copy()

        if t >= t_next_out:
            data.append((t.value_in(units.yr), binary.child1.mass.value_in(units.MSun), binary.child2.mass.value_in(units.MSun),
                         binary.child1.radius.value_in(units.RSun), binary.child2.radius.value_in(units.RSun),
                         binary.semi_major_axis.value_in(units.RSun), binary.eccentricity))
            t_next_out = t + dt_out

            print("{:g}%".format(t/tend*100), end="\r")

    bse.stop()


    #######################
    ###### PLOTTING #######
    #######################
    data = np.array(data)
    time = data[:,0]
    mass1 = data[:,1]
    mass2 = data[:,2]
    radius1 = data[:,3]
    radius2 = data[:,4]
    semi = data[:,5]
    ecc = data[:,6]
    peri = semi * (1 - ecc)

    figure = plt.figure(figsize=(16, 7))
    plot = figure.add_subplot(1, 2, 1)
    plot.plot(time, mass1, label='m1')
    plot.plot(time, mass2, label='m2')
    plot.set_xlabel('Time [yr]')
    plot.set_ylabel('Mass [MSun]')
    plot.legend(loc='best')

    plot = figure.add_subplot(1, 2, 2)
    plot.plot(time, radius1, label='R1')
    plot.plot(time, radius2, label='R2')
    plot.plot(time, semi, label='semi BSE', c="blue")
    plot.plot(time, peri, label='peri BSE', c="blue", ls="-.")
    plot.set_xlabel('Time [yr]')
    plot.set_ylabel('R [RSun]')
    plt.show()

if __name__ == "__main__":
    test_binary_evolution()