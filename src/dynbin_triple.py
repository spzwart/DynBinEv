#!/usr/bin/env python3


from amuse.ext.orbital_elements import new_binary_from_orbital_elements, rel_posvel_arrays_from_orbital_elements, \
                                        orbital_elements_from_binary, center_of_mass_array, get_orbital_elements_from_arrays
from amuse.units import units, nbody_system, constants
import numpy as np
from amuse.units.trigo import cos, sin, arccos, arctan2
from amuse.community.hermite.interface import Hermite
from amuse.community.sse.interface import SSE
from amuse.datamodel import Particles, Particle
import matplotlib.pyplot as plt


def i1_i2_from_imut(m1, m2, m3, a1, a2, e1, e2, imut):
    mtot1 = (m1 + m2)
    L1 = m1 * m2 / mtot1 * (constants.G * mtot1 * a1) ** 0.5
    mtot2 = (mtot1+m3)
    L2 = (mtot1 * m3) / mtot2 * (constants.G * mtot2 * a2) ** 0.5
    G1 = L1 * (1 - e1 * e1) ** 0.5
    G2 = L2 * (1 - e2 * e2) ** 0.5

    Gtot = (G1*G1 + G2*G2 + 2*cos(imut)*G1*G2).sqrt()

    H1_x = (Gtot * Gtot + G1 * G1 - G2 * G2) / (2 * Gtot)
    H2_x = (Gtot * Gtot + G2 * G2 - G1 * G1) / (2 * Gtot)

    i1 = arccos(H1_x / G1)
    i2 = arccos(H2_x / G2)
    return i1, i2

def get_orbit_of_triple(triplesys, inner1=0, inner2=1, outer=2):
    inner_pos_rel = triplesys[inner1].position - triplesys[inner2].position
    inner_vel_rel = triplesys[inner1].velocity - triplesys[inner2].velocity
    inner_mass = triplesys[inner1].mass + triplesys[inner2].mass

    inner_orb = get_orbital_elements_from_arrays(inner_pos_rel, inner_vel_rel, inner_mass, G=constants.G)

    inner_pos_com = (triplesys[inner1].position*triplesys[inner1].mass + triplesys[inner2].position*triplesys[inner2].mass) / inner_mass
    inner_vel_com = (triplesys[inner1].velocity*triplesys[inner1].mass + triplesys[inner2].velocity*triplesys[inner2].mass) / inner_mass

    outer_pos_rel = inner_pos_com - triplesys[outer].position
    outer_vel_rel = inner_vel_com - triplesys[outer].velocity

    outer_orb = get_orbital_elements_from_arrays(outer_pos_rel, outer_vel_rel, inner_mass + triplesys[outer].mass, G=constants.G)

    # Mutual inclination
    #Linn = inner_pos_rel.cross(inner_vel_rel)
    #Lout = outer_pos_rel.cross(outer_vel_rel)
    #CosNorm = Linn.dot(Lout) / (Linn.length() * Lout.length())
    #i_mut = np.arccos(CosNorm)
    #print(np.degrees(i_mut))


    # Better formatting
    outer_orb = [x[0]for x in outer_orb]
    inner_orb = [x[0]for x in inner_orb]
    outer_orb[0] = outer_orb[0].as_quantity_in(units.au)
    inner_orb[0] = inner_orb[0].as_quantity_in(units.au)

    return inner_orb, outer_orb


class TripleSystemWithCE:
    def __init__(self, length_unit=units.RSun, time_unit=units.yr):
        self.length_unit = 1 * length_unit
        self.time_unit = 1 * time_unit
        self.conv = nbody_system.nbody_to_si(self.length_unit, self.time_unit)

    def make_triple(
        self, m1, m2, m3, a1, a2, e1=0.0, e2=0.0, ome1=0.0 | units.deg,
        ome2=0.0 | units.deg, Ome1=0.0 | units.deg, i_mut=0.0 | units.deg,
        nu1=0.0 | units.deg, nu2=0.0 | units.deg
    ):

        if 1 - cos(i_mut) < 1e-15:
            i1 = i2 = 0.0
        else:
            i1, i2 = i1_i2_from_imut(m1, m2, m3, a1, a2, e1, e2, i_mut)

        position_vector1, velocity_vector1 = rel_posvel_arrays_from_orbital_elements(
            primary_mass=m1,
            secondary_mass=m2,
            semi_major_axis=a1,
            eccentricity=e1,
            true_anomaly=nu1,
            inclination=i1,
            longitude_of_the_ascending_node=Ome1,
            argument_of_periapsis=ome1,
            G=constants.G
        )

        position_vector2, velocity_vector2 = rel_posvel_arrays_from_orbital_elements(
            primary_mass=m1+m2,
            secondary_mass=m3,
            semi_major_axis=a2,
            eccentricity=e2,
            true_anomaly=nu2,
            inclination=i2,
            longitude_of_the_ascending_node=Ome1+units.pi,
            argument_of_periapsis=ome2,
            G=constants.G
        )

        pos_com1 = position_vector1[0] * m2/(m1+m2)
        vel_com1 = velocity_vector1[0] * m2/(m1+m2)

        triplesys = Particles(3)
        triplesys[0].mass = m1
        triplesys[0].position = - pos_com1
        triplesys[0].velocity = - vel_com1

        triplesys[1].mass = m2
        triplesys[1].position = position_vector1[0] - pos_com1
        triplesys[1].velocity = velocity_vector1[0] - vel_com1

        triplesys[2].mass = m3
        triplesys[2].position = position_vector2[0]
        triplesys[2].velocity = velocity_vector2[0]

        self.triplesys = triplesys
        self.inner1, self.inner2, self.outer = 0, 1, 2

        inn_orb, out_orb = get_orbit_of_triple(triplesys, inner1=self.inner1, inner2=self.inner2, outer=self.outer)
        Period1 = 2*np.pi * ( inn_orb[0]**3 / (constants.G * (m1+m2)) ).sqrt()
        Period2 = 2*np.pi * ( out_orb[0]**3 / (constants.G * (m1+m2+m3)) ).sqrt()

        print("Inner orbit:\nsemi={}, ecc={}, nu={}, inc={}, ome={}, Ome={}".format(*inn_orb))
        print("Outer orbit:\nsemi={}, ecc={}, nu={}, inc={}, ome={}, Ome={}".format(*out_orb))
        print("P1={}\nP2={}".format(Period1.as_string_in(units.yr), Period2.as_string_in(units.yr)))
        print("i1={}\ni2={}".format(i1.as_string_in(units.deg), i2.as_string_in(units.deg)))

        i_mut = arccos(cos(out_orb[3])*cos(inn_orb[3]) + cos(out_orb[4]-inn_orb[4])*sin(out_orb[3])*sin(inn_orb[3]))
        print("i_mut", i_mut.as_string_in(units.deg))

        return triplesys


    def initialize_simulation(self):
        # Add stellar radii with SSE
        #TODO

        self.stars = self.triplesys
        self.gravity = Hermite(self.conv)
        self.evolution = SSE()
        self.gravity.particles.add_particles(self.stars)
        self.evolution.particles.add_particles(self.stars)
        self.grav_to_stars = self.gravity.particles.new_channel_to(self.stars)
        self.evo_to_stars = self.evolution.particles.new_channel_to(self.stars)
        self.grav_from_stars = self.stars.new_channel_to(self.gravity.particles)

        print(self.stars)

    def run_model(self, tfin, dt_out, dt_interaction=None, tstart=0|units.yr, no_stevo=True):
        time = tstart
        dtout_next = time + dt_out

        if dt_interaction is None:
            dt_interaction = dt_out

        inn_orb, out_orb = get_orbit_of_triple(self.stars, inner1=self.inner1, inner2=self.inner2, outer=self.outer)

        a1 = [inn_orb[0].value_in(units.RSun)]
        e1 = [inn_orb[1]]
        a2 = [out_orb[0].value_in(units.RSun)]
        e2 = [out_orb[1]]
        i_mut = [arccos(cos(out_orb[3])*cos(inn_orb[3]) + cos(out_orb[4]-inn_orb[4])*sin(out_orb[3])*sin(inn_orb[3])).value_in(units.deg)]
        t = [tstart.value_in(self.time_unit)]
        while time < tfin:
            time += dt_interaction
            self.gravity.evolve_model(time)
            self.evolution.evolve_model(time)
            self.grav_to_stars.copy()
            if not no_stevo:
                self.evo_to_stars.copy_attributes(["mass", "radius"])
                self.grav_from_stars.copy()

            if time > dtout_next:
                inn_orb, out_orb = get_orbit_of_triple(
                    self.stars, inner1=self.inner1, inner2=self.inner2,
                    outer=self.outer)
                t.append(time.value_in(self.time_unit))
                a1.append(inn_orb[0].value_in(units.RSun))
                e1.append(inn_orb[1])
                a2.append(out_orb[0].value_in(units.RSun))
                e2.append(out_orb[1])
                i_mut.append(arccos(cos(out_orb[3])*cos(inn_orb[3]) + cos(out_orb[4]-inn_orb[4])*sin(out_orb[3])*sin(inn_orb[3])).value_in(units.deg))
                dtout_next = time + dt_out

            # Check collisions
            #TODO

            print("time=", time.as_string_in(self.time_unit), end="\r")

        self.gravity.stop()

        data = np.array((t, a1, e1, a2, e2, i_mut))

        return data

def plot_data(data):
    t = data[0,:]
    a1 = data[1,:]
    e1 = data[2,:]
    i_mut = data[5,:]

    f, ax = plt.subplots(3,1)
    ax[0].plot(t, a1)
    ax[1].plot(t, e1)
    ax[2].plot(t, i_mut)
    ax[2].set_xlabel("time [yr]")
    ax[2].set_ylabel("inclination [deg]")
    ax[1].set_ylabel("eccentricity")
    ax[0].set_ylabel("semimajor axis")
    plt.show()

if "__main__" == __name__:
    TriSys = TripleSystemWithCE()
    TriSys.make_triple(m1=110|units.MSun, m2=30|units.MSun, m3=30|units.MSun, a1=1|units.au, a2=25|units.au, i_mut=90|units.deg, e1=0.1, e2=0.2)
    TriSys.initialize_simulation()
    dataplot = TriSys.run_model(tfin=10000|units.yr, dt_out=10|units.yr)
    plot_data(dataplot)
