
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate, optimize

G = 6.674e-11
c = 2.998e9
M = 1.989e31

schwarzR = (2*G*M)/(c*c)

print('''Choose starting values for alpha and beta for a particle with non-zero mass and a beta value for particles with zero mass. \nFor non-zero mass alpha>1 and beta>0.\nFor the example alpha=1.3416407865 and beta=6.\nPhoton sphere example is always displayed by the prgram at the end for the given intiial beta value.\n''')

alpha = float(input('Alpha? '))
beta_M = float(input('Beta for non-zero mass? '))
beta_P = float(input('Beta for zero mass? '))


def inittial_M(a, B):
    r0M = (10*G*M)/(c*c)
    u0M = schwarzR/r0M
    return ((B*G*M)/c), a*c*c, u0M

def inittial_P(B):
    r0P = 1.5*schwarzR
    u0P = schwarzR/r0P
    return B/schwarzR, u0P

L, E, u0_M = inittial_M(alpha, beta_M)
l1, u0_P = inittial_P(beta_P)
#l1 = 2.597
#l1 = 1/(u0_P*np.sqrt(1-u0_P))


def f_P(phi, z):
    u, dudphi = z

    dzdt = [dudphi, (-u + (3/2)*(u**2))]

    return dzdt



kp = ((1/l1)*np.sqrt(1-(l1*l1*u0_P*u0_P*(1-u0_P))))

def f_M(phi, z):
    u, dudphi = z
    
    dzdt = [dudphi,
            ((3/2)*u*u) - u + ((schwarzR*schwarzR*c*c)/(2*L*L))]
    
    return dzdt

kM = ((schwarzR*c)/L)*np.sqrt(((E/(c*c))*(E/(c*c)))-((1-u0_M)*(1+(u0_M*u0_M*(L/(schwarzR*c))*(L/(schwarzR*c))))))

t_span = (0, 2*np.pi)
t_eval = np.linspace(0, 2*np.pi, 100)

sol_M = integrate.solve_ivp(f_M,t_span,[u0_M,kM],t_eval=t_eval)
sol_P = integrate.solve_ivp(f_P,t_span,[u0_P,kp],t_eval=t_eval)

r_M = schwarzR/sol_M.y[0]
r_P = schwarzR/sol_P.y[0]


#_______________ZERO MASS__________________________________
plt.plot(sol_P.t, r_P, c='r', lw='1', label='Particle')
plt.title('Motion of particle of zero mass')
plt.xlabel('Angular position on the equitorial plane')
plt.ylabel('Position of zero mass particle from centre of black hole')
plt.axhline(schwarzR, c='grey', label='Schwarzcschild Radius')
plt.legend()
plt.show()

plt.polar(sol_P.t,r_P,c="r",lw="1")
plt.title('Motion of particle of zero mass')
plt.fill(t_eval,np.full_like(t_eval,schwarzR),c="grey")
plt.grid(b=None,which="Major",axis="y")
plt.yticks([])
plt.show()

#_______________NON-ZERO MASS___________________________
plt.plot(sol_M.t, r_M, c='r', lw='1', label='Particle')
plt.title('Motion of particle of non-zero mass')
plt.axhline(schwarzR, c='grey', label='Schwarzschild Radius')
plt.xlabel('Angular position on the equitorial plane')
plt.ylabel('Position of non-zero mass particle from centre of black hole')
plt.legend()
plt.show()

plt.polar(sol_M.t,r_M,c="r",lw="1")
plt.title('Motion of particle of non-zero mass')
plt.fill(t_eval,np.full_like(t_eval,schwarzR),c="grey")
plt.grid(b=None,which="Major",axis="y")
plt.yticks([])
plt.show()

#____________PHOTON-SPHERE____________________________
l1 = 1/(u0_P*np.sqrt(1-u0_P))
kp = ((1/l1)*np.sqrt(1-(l1*l1*u0_P*u0_P*(1-u0_P))))
sol_P = integrate.solve_ivp(f_P,t_span,[u0_P,kp],t_eval=t_eval)
r_P = schwarzR/sol_P.y[0]

plt.plot(sol_P.t, r_P, c='r', lw='1', label='Particle')
plt.title('Motion of particle in photon sphere')
plt.xlabel('Angular position on the equitorial plane')
plt.ylabel('Position of non-zero mass particle from centre of black hole')
plt.axhline(schwarzR, c='grey', label='Schwarzcschild Radius')
plt.legend()
plt.show()

plt.polar(sol_P.t,r_P,c="r",lw="1")
plt.title('Motion of particle of zero mass in photon sphere')
plt.fill(t_eval,np.full_like(t_eval,schwarzR),c="grey")
plt.grid(b=None,which="Major",axis="y")
plt.yticks([])
plt.show()