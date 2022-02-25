import numpy as np
from amuse.units import constants, units
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

def period_from_binary(binary):
    try:
        mu = binary.mass.sum()*constants.G
        r = (binary[0].position - binary[1].position).length()
        v2 = (binary[0].velocity - binary[1].velocity).length_squared()
        a = (mu * r / (2. * mu - r * v2) )
        period = (a*a*a/mu).sqrt()
    except RuntimeWarning:
        print("found error", a)
        quit()
    return period



def C_from_X(X, mu, a, l, k):
    """
    Obtains C coeffiecient from dimensionless inspiral parameter X (\chi)
    :param mu: standard gravitational parameter
    :param a: initial semimajor axis
    :param l: l exponent
    :param k: k exponent
    :return:
    """
    C = X / np.pi * mu ** (1 - l / 2) * a ** (l / 2 + k - 2)
    Cunit_L = 1 - l + k
    Cunit_T = l - 2
    return C, Cunit_L, Cunit_T

def kick_stars_l2k0(star1, star2, dt, C):
    """
    l=2, k=0 model
    :param star1:
    :param star2:
    :param dt:
    :param C:
    :return:
    """
    #vcom = stars.center_of_mass_velocity()
    #pos = star1.position - star2.position
    vel = star1.velocity - star2.velocity

    #r = pos.length()
    #print(vel)
    v = vel.length()

    vvec = vel / v

    acc = -C * v * v * vvec
    mtot = star1.mass + star2.mass
    print(v / (acc * dt).length())

    star1.velocity += -acc * dt * star2.mass / mtot
    star2.velocity += acc * dt * star1.mass / mtot
