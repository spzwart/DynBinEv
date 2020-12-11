import numpy as np
from amuse.units import constants, units
from amuse.datamodel import Particles
from comenv import CE_drag_averaged


class BSEPairKicker():

    def __init__(self, kicks_per_orbit=100.0, tides=False, mass_loss=False,
                 mass_accretion=False, common_envelope=False, forceform_comenv="l1k2"):
        """
        Initialized the kicker
        :param kicks_per_orbit: needed to estimate the next optimal timestep #TODO
        :param tides: enable/disable tides
        :param mass_loss: enable/disable
        :param mass_accretion: enable/disable
        :param common_envelope: enable/disable
        """

        self.mass_loss = mass_loss
        self.tides = tides
        self.common_envelope = common_envelope
        self.mass_accretion = mass_accretion
        self.kicks_per_orbit = kicks_per_orbit

        self.comenv_phase = False
        self.comenv_module = CE_drag_averaged(forceform=forceform_comenv)

    def kick_pair(self, stellar_pair, dt):
        """
        Kicks the pair of particle according to the enabled physical processes
        :param stellar_pair: the stellar pair or binary that is evolving via stellar evolution
        :param dt: the timestep of the kick step
        :return:
        """

        self.pair = stellar_pair
        self.pos_com = stellar_pair.center_of_mass()
        self.vel_com = stellar_pair.center_of_mass_velocity()
        self.dt = dt
        self.calculate_useful_stuff()

        if self.mass_accretion: self.kick_from_accretion()
        if self.tides: self.kick_from_eccentric_tides()

    def calculate_useful_stuff(self):
        """
        Utility function, run once per timestep
        """

        self.pos = self.pair[1].position - self.pair[0].position
        self.vel = self.pair[1].velocity - self.pair[0].velocity

        self.r = self.pos.length()
        self.v = self.vel.length()
        self.inv_r = 1.0 / self.r

        self.rvec = self.pos * self.inv_r
        self.rdot = (self.rvec * self.vel).sum() * self.rvec
        self.rdot_mag = self.rdot.length()

        self.vth = self.vel - self.rdot
        self.vth_mag = self.vth.length()
        self.vth_vec = self.vth / self.vth_mag

    def kick_from_accretion(self):
        """
        Kick from accretion in the tangential direction.
        The star recoils due to momentum conservation
        Assuming dVth1/Vrel = -dm1/m1

        Particles need to have this attribute:
        dmdt_acc: instantaneus mass accretion rate [mass/time]
        """

        accth_mag = -self.v * self.pair.dmdt_acc / self.pair.mass
        kick0 = -self.vth_vec * accth_mag[0] * self.dt
        kick1 = self.vth_vec * accth_mag[1] * self.dt

        self.pair[0].velocity += kick0
        self.pair[1].velocity += kick1


    def kick_from_eccentric_tides(self):
        """
        Kick from tides due to eccentric orbit (Equilibrium tide model, Hut 1981)

        Particles to have following attributes:
        radius: stellar radius [length]
        kaps: apsidal constant [no unit]
        taulag: time-lag [time]
        """

        inv_r_7 = self.inv_r**7
        r0_5 = self.pair[0].radius ** 5
        r1_5 = self.pair[1].radius ** 5
        m0_2 = self.pair[0].mass * self.pair[0].mass
        m1_2 = self.pair[1].mass * self.pair[1].mass

        ftr = -3.0 * inv_r_7 * constants.G * ((m1_2 * r0_5 * self.pair[0].kaps + m0_2 * r1_5 * self.pair[1].kaps) +  # Non-dissipative
                                              3 * self.inv_r * self.rdot_mag *
                                              (m1_2 * r0_5 * self.pair[0].kaps * self.pair[0].taulag
                                               + m0_2 * r1_5 * self.pair[1].kaps * self.pair[1].taulag))  # Dissipative

        hutforce = ftr * self.inv_r * self.pos
        kick0 = (1.0 / self.pair[0].mass) * hutforce * self.dt
        kick1 = (-1.0 / self.pair[1].mass) * hutforce * self.dt

        self.pair[0].velocity += kick0
        self.pair[1].velocity += kick1


    def begin_common_envelope(self, Ece, Tce):
        """
        Sets the beginning of the common envelope inspiral
        :param Ece: energy lost during the common envelope phase
        :param Tce: duration of the common envelope phase
        :return:
        """
        mu = (self.pair[0].mass + self.pair[1].mass) * constants.G
        Eps0 = 0.5 * self.v**2 - mu*self.inv_r
        redmass = (self.pair[0].mass * self.pair[1].mass)/(self.pair[0].mass + self.pair[1].mass)
        Eps_ce = Ece/redmass

        self.Kce = self.comenv_module.K_from_eps(Eps0, Eps_ce, Tce, mu)
        self.comenv_phase = True

