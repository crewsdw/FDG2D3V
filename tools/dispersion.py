import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import pyvista as pv
# import cupy as cp
import scipy.optimize as opt

# "Global" parameters
om_pc = 1.0  # omega_p / omega_c


def Z(z):
    sol = 1j * np.sqrt(np.pi) * np.exp(-z ** 2.0) * (1.0 + sp.erf(1j * z))
    alt = 1j * np.sqrt(np.pi) * np.exp(-z ** 2.0) - (1.0 / z + 1.0 / (2.0 * (z ** 2.0)) + 3.0 / (4.0 * (z ** 3.0)))
    return np.where(np.isnan(sol), alt, sol)


def Zprime(z):
    return -2 * (1 + z * Z(z))


def fac(n):
    return np.math.factorial(n)


def shifted_disp_para(zeta, n, k_para):
    return sum([(-1) ** s / (fac(n + s) * fac(n - s)) * Zprime(zeta - s / k_para / om_pc)
                for s in range(-n + 1 - 1, n + 1)])


def shifted_disp_perp(zeta, n, k_para):
    return sum([s * (-1) ** s / (fac(n + s) * fac(n - s)) * Z(zeta - s / k_para / om_pc)
                for s in range(-n + 1 - 1, n + 1)])


def V_parallel(z, k_perp, k_para, j, terms):
    arg = -2.0 * k_perp ** 2.0
    return sum([
        sp.poch(0.5, n) * sp.poch(j + 1, n) * shifted_disp_para(z, n, k_para) * (arg ** n) / fac(n)
        for n in range(terms)
    ])


def V_perp(z, k_perp, k_para, j, terms):
    arg = -2.0 * k_perp ** 2.0
    return 1.0 * sum([
        (sp.poch(0.5, n) * sp.poch(j + 1, n) *  # (n / (n + j)) *
         shifted_disp_perp(z, n, k_para) * (arg ** n) / fac(n))
        for n in range(1, terms)
    ])


def dispersion(z, k_perp, k_para, ring_j, terms):
    ksq = k_perp ** 2.0 + k_para ** 2.0
    return (1 - V_parallel(z, k_perp, k_para, ring_j, terms=terms) / (ksq ** 2.0) +
     V_perp(z, k_perp, k_para, ring_j, terms=terms) / (ksq ** 2.0) / k_para / om_pc)


def standard(z, k_perp, k_para, terms):
    b = k_perp ** 2.0
    k = np.sqrt(k_perp ** 2.0 + k_para ** 2.0)
    return 1.0 - np.exp(-b ** 2.0) / (k ** 2.0) * sum([
        sp.iv(s, b) * (Zprime(z - s / k_para) - s / k_para * Z(z - s / k_para))
        for s in range(-terms + 1, terms)
    ])


def dispersion_fsolve(om, wave, ring_j, terms):
    freq = om[0] + 1j*om[1]
    d = dispersion(freq, wave[0], wave[1], ring_j, terms)
    return [np.real(d), np.imag(d)]


# Define complex plane
k_perp, k_para = 0.1, 0.1

z_r = np.linspace(15, 35.5, num=250)
z_i = np.linspace(-np.pi, np.pi, num=250)
z = (np.tensordot(z_r, np.ones_like(z_i), axes=0) +
     1.0j * np.tensordot(np.ones_like(z_r), z_i, axes=0))
X, Y = np.tensordot(z_r, np.ones_like(z_i), axes=0), np.tensordot(np.ones_like(z_r), z_i, axes=0)
cb = np.linspace(-1, 1, num=100)

func = dispersion(z, k_perp, k_para, 0, terms=20)
func2 = standard(z, k_perp, k_para, terms=15)

plt.figure()
# plt.contourf(X, Y, np.real(func), cb, extend='both')
plt.contour(X, Y, np.real(func), 0, colors='g')
plt.contour(X, Y, np.imag(func), 0, colors='r')

plt.figure()
# plt.contourf(X, Y, np.real(func2), cb, extend='both')
plt.contour(X, Y, np.real(func2), 0, colors='g')
plt.contour(X, Y, np.imag(func2), 0, colors='r')

plt.show()


# Root solve, analysis at 45 degrees to field
k_para, k_perp = np.linspace(0.075, 0.4, num=20), np.linspace(0.075, 0.4, num=20)
wave = np.sqrt(k_para**2.0 + k_perp**2.0)
waves = np.array([k_perp, k_para])

mode1 = np.zeros_like(k_para) + 0j
mode2 = np.zeros_like(k_para) + 0j
guess_r1, guess_i1 = np.zeros_like(k_para), np.zeros_like(k_para)
guess_r2, guess_i2 = np.zeros_like(k_para), np.zeros_like(k_para)


guess_r1[k_para <= 0.125] = 7.0
guess_r1[k_para >= 0.125] = 4.69
guess_r1[k_para >= 0.135] = 3.5
guess_r1[k_para >= 0.25] = 2

guess_i1[k_para <= 0.125] = -0.01
guess_i1[k_para >= 0.125] = -0.03
guess_i1[k_para >= 0.135] = -0.08
guess_i1[k_para >= 0.25] = -0.3

guess_r2[k_para <= 0.125] = 25.0
guess_r2[k_para >= 0.125] = 25
guess_r2[k_para >= 0.135] = 20
guess_r2[k_para >= 0.25] = 15

guess_i2[k_para <= 0.125] = -0.01
guess_i2[k_para >= 0.125] = -0.03
guess_i2[k_para >= 0.135] = -0.08
guess_i2[k_para >= 0.25] = -0.05

for idx in range(k_para.shape[0]):
    sol = opt.root(dispersion_fsolve, x0=np.array([guess_r1[idx], guess_i1[idx]]), args=(waves[:, idx], 0, 10))
    mode1[idx] = sol.x[0] + 1j*sol.x[1]
    sol = opt.root(dispersion_fsolve, x0=np.array([guess_r2[idx], guess_i2[idx]]), args=(waves[:, idx], 0, 10))
    mode2[idx] = sol.x[0] + 1j*sol.x[1]

# Scale solution to frequency
mode1_om = np.multiply(wave, mode1)
mode2_om = np.multiply(wave, mode2)

plt.figure()
plt.plot(wave, np.real(mode1_om), 'k')
plt.plot(wave, np.imag(mode1_om), 'k--')
plt.plot(wave, np.real(mode2_om), 'g')
plt.plot(wave, np.imag(mode2_om), 'g--')
plt.xlabel('Wavenumber'), plt.ylabel('Frequency')
plt.grid(True), plt.tight_layout()
plt.show()

quit()

# Run parameters
# k_para, k_perp = 0.5, 0.5


