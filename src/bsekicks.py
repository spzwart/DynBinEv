import numpy as np
from amuse.units import constants, units
from amuse.datamodel import Particles






def eccentric_tides_kick(particle_pair, dt):
    """
    Kick from tides due to eccentric orbit.
    :param particle_pair: Amuse particle set with two particles
                          Needs to have following attributes:
                           * kaps: apsidal constant [no unit]
                           * taulag: time-lag [time]
                          Position and velocities need to be scaled to the pair center of mass
    :param dt: time step
    :return:
    """
    #particles.move_to_center()
    pos = particle_pair[1].position - particle_pair[0].position
    vel = particle_pair[1].velocity - particle_pair[0].velocity

    r = pos.length()
    rvec = pos / r
    rdot = (rvec * vel).sum() * rvec
    rdot_mag = rdot.length()

    inv_r = 1.0 / r
    inv_r_7 = inv_r**7
    r0_5 = particle_pair[0].radius ** 5
    r1_5 = particle_pair[1].radius ** 5
    m0_2 = particle_pair[0].mass * particle_pair[0].mass
    m1_2 = particle_pair[1].mass * particle_pair[1].mass

    ftr = -3.0 * inv_r_7 * constants.G * ((m1_2 * r0_5 * particle_pair[0].kaps + m0_2 * r1_5 * particle_pair[1].kaps) +  # Non-dissipative
                                          3 * inv_r * rdot_mag *
                                          (m1_2 * r0_5 * particle_pair[0].kaps * particle_pair[0].taulag + m0_2 * r1_5 * particle_pair[1].kaps * particle_pair[1].taulag))  # Dissipative

    hutforce = ftr * inv_r * pos

    acc0 = (1.0 / particle_pair[0].mass) * hutforce
    acc1 = (-1.0 / particle_pair[1].mass) * hutforce

    kick0 = acc0 * dt
    kick1 = acc1 * dt

    particle_pair[0].velocity += kick0
    particle_pair[1].velocity += kick1


def kick_from_accretion(particle_pair, dt):
    """
    Kick from accretion in the tangential direction. 
    The star recoils due to momentum conservation
    Assuming dVth1/Vrel = -dm1/m1
    :param particle_pair: Amuse particle set with two particles
                          Needs to have following attributes:
                           * dmdt_acc: instantaneus mass accretion rate [mass/time]
                          Position and velocities need to be scaled to the pair center of mass
    :param dt:
    :return:
    """
    particle_pair.move_to_center()

    pos = particle_pair[1].position - particle_pair[0].position
    vel = particle_pair[1].velocity - particle_pair[0].velocity
    r = pos.length()
    v = vel.length()

    rvec = pos/r
    rdot = (rvec*vel).sum() * rvec
    vth = vel-rdot
    vth_mag = vth.length()
    vth_vec = vth/vth_mag

    accth_mag = -v * particle_pair.dmdt_acc / particle_pair.mass
    kick0 = -vth_vec * accth_mag[0] * dt
    kick1 = vth_vec * accth_mag[1] * dt

    particle_pair[0].velocity += kick0
    particle_pair[1].velocity += kick1