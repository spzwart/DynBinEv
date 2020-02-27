from amuse.units import units, constants
import numpy as np


def f1hut(e2):
    f = 1 + e2 * (15.5 + e2 * (31.875 + e2 * (11.5625 + e2 * 0.390625)))
    return f

def f2hut(e2):
    f = 1 + e2 * (7.5 + e2 * (5.625 + e2 * 0.3125))
    return f

def f3hut(e2):
    f = 1 + e2 * (3.75 + e2 * (1.875 + e2 * 7.8125e-2))
    return f

def f4hut(e2):
    f = 1 + e2 * (1.5 + e2 * 0.125)
    return f

def f5hut(e2):
    f = 1 + e2 * (3 + e2 * 0.375)
    return f


class QuickTides:
    def __init__(self,
                 a0, e0,
                 mstar1, mstar2,
                 rstar1, rstar2,
                 kaps1, kaps2,
                 taulag1, taulag2):
        self.a0 = self.a = a0
        self.e0 = self.e = e0
        self.mstar1 = mstar1
        self.rstar1 = rstar1
        self.mstar2 = mstar2
        self.rstar2 = rstar2
        self.kaps1 = kaps1  # k should be apsidal number, not love number
        self.kaps2 = kaps2  # kapsidal = klove/2
        self.taulag1 = taulag1
        self.taulag2 = taulag2

        self.q2 = self.mstar1 / self.mstar2
        self.q1 = self.mstar2 / self.mstar1

        self.n = (constants.G * (self.mstar1 + self.mstar2) / (self.a * self.a * self.a)).sqrt()

        self.star1_tide_diagnostic()

        self.star2_tide_diagnostic()

    @staticmethod
    def k_over_T_from_taulag(taulag, klove, mass, radius):
        k_over_T = (constants.G * mass) / (radius * radius * radius)  # Surface orbital frequency ^2
        k_over_T *= (klove / 2) * taulag  # kapsidal = klove/2
        return k_over_T

    @staticmethod
    def taulag_from_k_over_T(k_over_T, klove, mass, radius):
        taulag = k_over_T / (klove / 2)  # kapsidal = klove/2
        taulag *= (radius * radius * radius) / (constants.G * mass)
        return taulag

    @staticmethod
    def Thut(taulag, mass, radius):
        T = (radius * radius * radius) / (constants.G * mass)
        T /= taulag
        return T

    @staticmethod
    def Teggleton(taulag, mass, radius, klove):
        T = 3 * (1 + 1 / klove) * QuickTides.Thut(taulag, mass, radius)
        return T

    @staticmethod
    def taulag_from_Qprime(Qprime, ntide, klove):  # Qprime is comprises both Q and k
        Q = Qprime * klove  # 1/Qprime = klove/Q
        taulag = 1 / (Q * ntide)  # Q^-1 = taulag*ntide
        return taulag

    @staticmethod
    def Qprime_from_taulag(taulag, ntide, klove):  # Qprime is comprises both Q and k
        Q = 1 / (ntide * taulag)  # Q^-1 = taulag*ntide
        Qprime = Q / klove  # 1/Qprime = klove/Q
        return Qprime

    @staticmethod
    def circularization_timescale(k_over_T, a, mass, radius, mpert):
        q = mpert / mass
        tauc_inv = 10.5 * k_over_T * q * (q + 1) * (radius / a) ** 8
        tauc = 1 / tauc_inv
        return tauc

    def star1_tide_diagnostic(self):
        self.k_over_T1 = self.k_over_T_from_taulag(self.taulag1, self.kaps1, self.mstar1, self.rstar1)
        self.Thut1 = self.Thut(self.taulag1, self.mstar1, self.rstar1)
        self.Teggleton1 = self.Teggleton(self.taulag1, self.mstar1, self.rstar1, self.kaps1)

        print("\n-=== STAR1 TIDE ===-")
        #print("Qprime = {:e}".format(self.Qplanet))
        print("k_over_T: {:s}".format(self.k_over_T1.as_string_in(1 / units.yr)))
        print("taulag: {:s}".format(self.taulag1.as_string_in(units.s)))
        print("Thut: {:s}".format(self.Thut1.as_string_in(units.hour)))
        print("Teggleton: {:s}".format(self.Teggleton1.as_string_in(units.hour)))

        Tcirc = self.circularization_timescale(self.k_over_T1, self.a, self.mstar1, self.rstar1, self.mstar2)
        print("circ timescale: {:s}".format(Tcirc.as_string_in(units.Myr)))
        print("kaps {:g}".format(self.kaps1))

    def star2_tide_diagnostic(self):
        self.k_over_T2 = self.k_over_T_from_taulag(self.taulag2, self.kaps2, self.mstar2, self.rstar2)
        self.Thut2 = self.Thut(self.taulag2, self.mstar2, self.rstar2)
        self.Teggleton2 = self.Teggleton(self.taulag2, self.mstar2, self.rstar2, self.kaps2)

        print("\n-=== STAR2 TIDE ===-")
        print("k_over_T: {:s}".format(self.k_over_T2.as_string_in(1 / units.yr)))
        print("taulag: {:s}".format(self.taulag2.as_string_in(units.s)))
        print("Thut: {:s}".format(self.Thut2.as_string_in(units.hour)))
        print("Teggleton: {:s}".format(self.Teggleton2.as_string_in(units.hour)))

        Tcirc = self.circularization_timescale(self.k_over_T2, self.a, self.mstar2, self.rstar2, self.mstar1)
        print("circ timescale: {:s}".format(Tcirc.as_string_in(units.Myr)))
        print("kaps {:g}".format(self.kaps2))

    def compute_e_over_dedt(self):
        e2 = self.e * self.e
        self.update_f(e2)
        eqspin = self.equilibrium_spin(e2, self.n)

        # Stellar tide
        qstar = self.mstar1 / self.mstar2
        dedt = self.dedt(self.k_over_T2, eqspin, self.rstar2, qstar, e2)

        # Planet tide
        qplan = self.mstar2 / self.mstar1
        dedt += self.dedt(self.k_over_T1, eqspin, self.rstar1, qplan, e2)

        Tecc = self.e / np.abs(dedt)
        print("e/de = {:s}".format(Tecc.as_string_in(units.yr)))

    def dadt_dedt(self, a, e):
        e2 = e*e
        self.update_f(e2)
        n = (constants.G * (self.mstar1 + self.mstar2) / (a * a * a)).sqrt()
        eqspin = self.equilibrium_spin(e2, n)

        da_dt1 = self.dadt(self.k_over_T1, eqspin, self.rstar1, self.q1, e2)
        de_dt1 = self.dedt(self.k_over_T1, eqspin, self.rstar1, self.q1, e2)

        da_dt2 = self.dadt(self.k_over_T2, eqspin, self.rstar2, self.q2, e2)
        de_dt2 = self.dedt(self.k_over_T2, eqspin, self.rstar2, self.q2, e2)

        return da_dt1+da_dt2, de_dt1+de_dt2

    def dadt(self, k_over_T, spin, radius, q, e2):
        dad = -6 * k_over_T * q * (q + 1) * (radius / self.a) ** 8 * self.a / (1 - e2) ** 7.5
        dad = dad * (self.f1 - (1 - e2) ** 1.5 * self.f2 * spin / self.n)
        return dad

    def dedt(self, k_over_T, spin, radius, q, e2):
        ded = -27 * k_over_T * q * (q + 1) * (radius / self.a) ** 8 * self.e / (1 - e2) ** 6.5
        ded = ded * (self.f3 - 11. / 18 * (1 - e2) ** 1.5 * self.f4 * spin / self.n)
        return ded

    def equilibrium_spin(self, e2, n):
        ospin = self.f2 / (self.f5 * (1 - e2) ** 1.5) * n
        return ospin

    def update_f(self, e2):
        self.f1 = f1hut(e2)
        self.f2 = f2hut(e2)
        self.f3 = f3hut(e2)
        self.f4 = f4hut(e2)
        self.f5 = f5hut(e2)
