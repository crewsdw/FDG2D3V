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
    sig = 1
    # if np.abs(np.imag(z)) < np.abs(1.0/np.real(z)):
    #     sig = 1
    # if np.imag(z) < 1.0 / np.abs(np.real(z)):
    #     sig = 2
    alt = 1j * sig * np.sqrt(np.pi) * np.exp(-z ** 2.0) - (1.0 / z + 1.0 / (2.0 * (z ** 3.0)) +
                                                     3.0 / (4.0 * (z ** 5.0)) + 15.0 / (8.0 * (z ** 7.0)))
    return np.where(np.isnan(sol), alt, sol)


def Zprime(z):
    return -2 * (1 + z * Z(z))


def Zdoubleprime(z):
    return -2 * (Z(z) + z * Zprime(z))


def fac(n):
    return np.math.factorial(n)


def shifted_disp_para(zeta, n, k_para):
    return sum([(-1) ** s / (fac(n + s) * fac(n - s)) * Zprime(zeta - s / k_para / om_pc)
                for s in range(-n + 1 - 1, n + 1)])


def shifted_disp_perp(zeta, n, k_para):
    return -1.0 * sum([s * (-1) ** s / (fac(n + s) * fac(n - s)) * Z(zeta - s / k_para / om_pc)
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


def hyp2f2(n, j, x):
    # compute coefficients
    out = 0
    a = n + 0.5
    for m in range(j + 1):
        A = sp.gamma(j + 1) / (sp.gamma(m + 1) * sp.gamma(j - m + 1))
        # B = sp.gamma(a + m) / sp.gamma(2 * a + m)
        # C = sp.gamma(n - m)
        B = sp.gamma(a + m) / (sp.gamma(m + 1) * sp.gamma(a))
        C = sp.gamma(n + m + 1) / (sp.gamma(m + 1) * sp.gamma(n + 1))
        D = sp.gamma(2 * a + m) / (sp.gamma(m + 1) * sp.gamma(2 * a))
        # print(A), print(B), print(C), print(D)
        out += (A * B) / (C * D) * sp.hyp1f1(a + m, 2 * a + m, x) * (x ** m / sp.gamma(m + 1))
    return out
    # return (2 ** n) * out / np.sqrt(np.pi)


def perp_integral(n, j, x):
    n = abs(n)
    return sp.gamma(n + j + 1) / (sp.gamma(n + 1) ** 2.0) * ((-x / 4.0) ** n) * hyp2f2(n, j, x) / sp.gamma(j + 1)


def modified(z, k_perp, k_para, ring_j, terms):
    x = -2 * k_perp ** 2.0
    ksq = k_perp ** 2.0 + k_para ** 2.0
    # compute hyper-geometric
    return 1.0 - sum([
        perp_integral(n=s, j=ring_j, x=x) * (0.5 * Zprime((z - s) / k_para) - s / k_para * Z((z - s) / k_para))
        for s in range(1 - terms, terms)]) / ksq


def analytic_jacobian(z, k_perp, k_para, ring_j, terms):
    x = -2 * k_perp ** 2.0
    ksq = k_perp ** 2.0 + k_para ** 2.0
    return -sum([
        perp_integral(n=s, j=ring_j, x=x) * (0.5 * Zdoubleprime((z - s) / k_para) / k_para -
                                             s / k_para * Zprime((z - s) / (k_para ** 2.0)))
        for s in range(1 - terms, terms)]) / ksq


def standard(z, k_perp, k_para, terms):
    b = k_perp ** 2.0
    ksq = k_perp ** 2.0 + k_para ** 2.0
    return 1.0 - np.exp(-b) / ksq * sum([
        sp.iv(abs(s), b) * (0.5 * Zprime((z - s) / k_para) - s / k_para * Z((z - s) / k_para))
        for s in range(1 - terms, terms)
    ])


def dispersion_fsolve(om, wave, ring_j, terms):
    freq = om[0] + 1j * om[1]
    # d = dispersion(freq, wave[0], wave[1], ring_j, terms)
    d = modified(freq, wave[0], wave[1], ring_j, terms)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(om, wave, ring_j, terms):
    freq = om[0] + 1j*om[1]
    jac = analytic_jacobian(freq, wave[0], wave[1], ring_j, terms)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]  # using cauchy-riemann equations


# Define complex plane
angle = 45 * np.pi / 180.0
k_perp = 0.55
k_para = k_perp / np.tan(angle)
print(k_para), print(k_perp)

z_r = np.linspace(-0.5, 3.5, num=500)
z_i = np.linspace(-0.8, 0.02, num=500)
z = (np.tensordot(z_r, np.ones_like(z_i), axes=0) +
     1.0j * np.tensordot(np.ones_like(z_r), z_i, axes=0))
X, Y = np.tensordot(z_r, np.ones_like(z_i), axes=0), np.tensordot(np.ones_like(z_r), z_i, axes=0)
cb = np.linspace(-1, 1, num=100)

# func = dispersion(z, k_perp, k_para, 0, terms=20)
func = modified(z, k_perp, k_para, ring_j=0, terms=4)
# func2 = standard(z, k_perp, k_para, terms=10)

plt.figure()
# plt.contourf(X, Y, np.real(func), cb, extend='both')
plt.contour(X, Y, np.real(func), 0, colors='g')
plt.contour(X, Y, np.imag(func), 0, colors='r')
plt.grid(True)

# plt.figure()
# # plt.contourf(X, Y, np.real(func2), cb, extend='both')
# plt.contour(X, Y, np.real(func2), 0, colors='g')
# plt.contour(X, Y, np.imag(func2), 0, colors='r')
# plt.grid(True)

plt.show()

# Root solve, analysis at 45 degrees to field
num = 75

angle = 45 * np.pi / 180.0
k_perp = np.linspace(0.05, 0.95, num=num)
k_para = k_perp / np.tan(angle)

wave = np.sqrt(k_para ** 2.0 + k_perp ** 2.0)
waves = np.array([k_perp, k_para])

mode1 = np.zeros_like(k_para) + 0j
mode2 = np.zeros_like(k_para) + 0j
mode3 = np.zeros_like(k_para) + 0j
guess_r1, guess_i1 = np.zeros_like(k_para), np.zeros_like(k_para)
guess_r2, guess_i2 = np.zeros_like(k_para), np.zeros_like(k_para)
guess_r3, guess_i3 = np.zeros_like(k_para), np.zeros_like(k_para)


# 80 degrees
guess_r1[k_perp <= 0.22] = 1.26
guess_r1[k_perp >= 0.22] = 1.38
guess_r1[k_perp >= 0.3] = 1.4
guess_r1[k_perp >= 0.4] = 1.5
guess_r1[k_perp >= 0.5] = 1.6
guess_r1[k_perp >= 0.6] = 1.7
guess_r1[k_perp >= 0.8] = 1.8

guess_i1[k_perp <= 0.2] = -0.01
guess_i1[k_perp >= 0.2] = -0.01
guess_i1[k_perp >= 0.4] = -0.25
guess_i1[k_perp >= 0.5] = -0.3
guess_i1[k_perp >= 0.6] = -0.5
guess_i1[k_perp >= 0.8] = -1

guess_r2[k_perp <= 0.2] = 0.37
guess_r2[k_perp >= 0.2] = 0.4
guess_r2[k_perp >= 0.4] = 0.6
guess_r2[k_perp >= 0.7] = 0.7
guess_r2[k_perp >= 0.8] = 0.8
# guess_r2[k_perp >= 0.9] = 0.7

guess_i2[k_perp <= 0.2] = -0.0001
guess_i2[k_perp >= 0.2] = -0.1
guess_i2[k_perp >= 0.4] = -0.45
guess_i2[k_perp >= 0.6] = -0.6
guess_i2[k_perp >= 0.7] = -0.9
guess_i2[k_perp >= 0.8] = -1.2
# guess_i2[k_perp >= 1.4] = -0.4

guess_r3[k_perp <= 0.2] = 2.01
guess_r3[k_perp >= 0.2] = 2.05
guess_r3[k_perp >= 0.2] = 2.2
guess_r3[k_perp >= 0.9] = 2.5

guess_i3[k_perp <= 0.2] = -0.15
guess_i3[k_perp >= 0.2] = -0.3
guess_i3[k_perp >= 0.5] = -0.5
guess_i3[k_perp >= 0.9] = -0.7
guess_i3[k_perp >= 1] = -1.2

for idx in range(k_para.shape[0]):
    sol = opt.root(dispersion_fsolve, x0=np.array([guess_r1[idx], guess_i1[idx]]),
                   args=(waves[:, idx], 0, 10), jac=jacobian_fsolve)
    mode1[idx] = sol.x[0] + 1j * sol.x[1]
    sol = opt.root(dispersion_fsolve, x0=np.array([guess_r2[idx], guess_i2[idx]]),
                   args=(waves[:, idx], 0, 10), jac=jacobian_fsolve)
    mode2[idx] = sol.x[0] + 1j * sol.x[1]
    # sol = opt.root(dispersion_fsolve, x0=np.array([guess_r3[idx], guess_i3[idx]]),
    #                args=(waves[:, idx], 0, 10), jac=jacobian_fsolve)
    # mode3[idx] = sol.x[0] + 1j * sol.x[1]

# Scale solution to frequency
# mode1_om = np.multiply(k_para, mode1)
# mode2_om = np.multiply(k_para, mode2)
mode1_om = np.append([1.26], mode1)
wave = np.append([0], wave)
mode2_om = np.append([0.4], mode2)
mode3_om = np.append([2], mode3)

plt.figure()
plt.plot(wave, np.real(mode1_om), 'k')
plt.plot(wave, np.imag(mode1_om), 'k--')
plt.plot(wave, np.real(mode2_om), 'g')
plt.plot(wave, np.imag(mode2_om), 'g--')
# plt.plot(wave, np.real(mode3_om), 'r')
# plt.plot(wave, np.imag(mode3_om), 'r--')
plt.axis([wave[0], wave[-1], -1.3, 2.4])
plt.xlabel(r'Wavenumber $\sqrt{k_\perp^2+k_\parallel^2}$'), plt.ylabel('Frequency')
plt.grid(True), plt.title(r'Angle $\theta=45^\circ$'), plt.tight_layout()
plt.show()

quit()

# Run parameters
# k_para, k_perp = 0.5, 0.5


# Guesses
# 80 degrees
# guess_r1[k_perp <= 0.22] = 1.4
# guess_r1[k_perp >= 0.22] = 1.38
# guess_r1[k_perp >= 0.3] = 1.3
# guess_r1[k_perp >= 0.5] = 1.3
# guess_r1[k_perp >= 0.6] = 1.25
# guess_r1[k_perp >= 0.7] = 1.23
#
# guess_i1[k_perp <= 0.2] = -0.000001
# guess_i1[k_perp >= 0.2] = -0.000001
# guess_i1[k_perp >= 0.7] = -0.03
# guess_i1[k_perp >= 1.0] = -0.2
#
# guess_r2[k_perp <= 0.2] = 0.1
# guess_r2[k_perp >= 0.2] = 0.1
# guess_r2[k_perp >= 0.9] = 0.2
# guess_r2[k_perp >= 1.4] = 0.3
#
# guess_i2[k_perp <= 0.2] = -0.001
# guess_i2[k_perp >= 0.2] = -0.01
# guess_i2[k_perp >= 0.9] = -0.1
# guess_i2[k_perp >= 1] = -0.2
# guess_i2[k_perp >= 1.4] = -0.4
#
# guess_r3[k_perp <= 0.2] = 2.01
# guess_r3[k_perp >= 0.2] = 2.05
# guess_r3[k_perp >= 0.9] = 2.2
#
# guess_i3[k_perp <= 0.2] = -0.01
# guess_i3[k_perp >= 0.2] = -0.05
# guess_i3[k_perp >= 0.9] = -0.1
# guess_i3[k_perp >= 1] = -0.2
