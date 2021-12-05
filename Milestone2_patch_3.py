import numpy as np
from matplotlib import pyplot as plt
import sys

# only do longitudual friction
# print("x = {0}".format(x))

mad = np.array([2,3,4])

def grad_h(func, nodes): # Gradiant h
    """ Modified trapezoidal integration"""
    # Pads a 0 at the end of an array
    temp = pad_along_axis(func, nodes,axis = 1) # Using roll calculate the diff (ghost node of 0)

    return (temp - np.roll(temp, 1))

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

class crossrod:
    def __init__(self, T, dt, total_length, elements, density, radius, total_external_force,
                 G = 1E4, E = 1E6, dim = 3, **kwargs):
        # Plotting
        self.final_pos = []

        # Element Info
        self.e = elements
        self.n = self.e + 1 # nodes
        self.n_i = self.e - 1 # internal nodes

        # Initializing node mass
        self.area = np.pi * (radius**2) # Update?
        total_volume = self.area * total_length
        total_mass = density * total_volume
        self.m = np.zeros((1,self.n))
        element_mass = total_mass / self.e
        self.m[0][0] = element_mass/2
        self.m[0][1:self.n-1] = element_mass
        self.m[0][self.n-1] = element_mass/2

        # Initializing node radii
        self.r = np.full((1,self.n),radius) # Update?

        # Initializing node position
        self.pos = np.zeros((dim,self.n))
        for col in range(self.n):
            self.pos[2,col] = (total_length/self.e) * col

        # Length Info
        # UPDATE THIS AT EVERY TIME STEP
        self.l = self.pos[:,1:] - self.pos[:,:-1] # length vector
        self.l_mag = np.linalg.norm(self.l, axis = 0) # magnitude of length
        # DO NOT UPDATE THIS AT EVERY TIME STEP
        self.l_ref = self.pos[:,1:] - self.pos[:,:-1] # reference length (unstrecthed length of the rod)
        self.l_ref_mag = np.linalg.norm(self.l_ref, axis = 0) # magnitude of reference length as a scalar

        # Parameters determined by Length Info
        self.dil_fac = self.l_mag / self.l_ref_mag # dilatation factor
        self.tangents = self.l / self.l_mag # tangent vectors

        # Directors
        self.directors = np.zeros((3, 3, self.e))
        for idx in range(self.e):
            self.directors[:, :, idx] = np.eye(3) # maps from lab to material frame

        self.forces = np.zeros((dim,self.n)) # forces INITIALIZE
        self.forces[2,self.e] = total_external_force

        self.vel = np.zeros((dim,self.n)) # velocities
        self.ang_vel = np.zeros((dim,self.e)) # angular velocities

        # Shear/stretch diagonal matrix INITIALIZE INPUT FROM MATERIAL PROPERTIES
        self.S_hat = np.zeros((3,3,self.e))
        alpha_c = 4./3. # shape factor
        self.S_hat[0,0,:] = alpha_c * G * self.area
        self.S_hat[1,1,:] = alpha_c * G * self.area
        self.S_hat[2,2,:] = E * self.area

        # Moment of inertia diagonal matrix
        self.I = np.zeros((3,3,self.e))
        self.I[0,0,:] = self.area**2 / 4 * np.pi
        self.I[1,1,:] = self.area**2 / 4 * np.pi
        self.I[2,2,:] = self.area**2 / 4 * np.pi * 2

        # Bend diagonal matrix INITIALIZE INPUT FROM MATERIAL PROPERTIES
        self.B = np.zeros((3,3,self.n_i))
        self.B[0,0,:] = E * self.I[0,0,:]
        self.B[1,1,:] = E * self.I[1,1,:]
        self.B[2,2,:] = G * self.I[2,2,:]

        # J diagonal matrix.
        # **** if broken code, there might be some difference between dJ^ and J^
        # here i assume J is pI from dJ = pIds
        self.J = np.zeros((3,3,self.e))
        self.J[0,0,:] = density * self.I[0,0,:]
        self.J[1,1,:] = density * self.I[1,1,:]
        self.J[2,2,:] = density * self.I[2,2,:]

        # kappa here
        self.kappa = 0 # This has to be something, look to Evan for answer

        # shear/stress strain
        self.sigma = self.dil_fac * self.tangents - self.directors[2,:,:]

        # Governing Equations
        # pos += vel * dt # Equation 1
        # dv_dt = (grad_h(S_hat @ s / dil_fac) + f) / m # Equation 3

        for x in np.arange(0,T+dt,dt):
            self.pos, self.vel = self.position_verlet(dt, self.pos, self.vel)
            self.directors, self.ang_vel = self.angular_verlet(dt, self.directors, self.ang_vel)
            self.update(self.pos)
            self.final_pos.append(self.pos[2,-1])

    def position_verlet(self, dt, x, v):
        # x = position & v = velocity
        temp_x = x + 0.5*dt*v
        v_n = v + dt * self.force_rule(temp_x)
        x_n = temp_x + 0.5 * dt * v_n
        return x_n, v_n

    def angular_verlet(self, dt, Q, w):
        # Q = directors & w = angular velocity
        temp_Q = np.exp((-dt/2)*w) * Q
        w_n = w + dt * self.bending_rule(temp_Q)
        Q_n = np.exp((-dt/2)*w_n) * temp_Q
        return Q_n, w_n

    def bend_rule(self, temp_Q):
        self.update(temp_pos)

        matmul = np.zeros((3,self.e))
        matmul = np.einsum('ijk,jk->ik',self.B,self.kappa)

        self.bend_twist_internal_couple = grad_h(matmul, self.n)

        matmulfirst = np.zeros((3,self.e))
        matmulfirst = np.einsum('ijk,jk->ik',self.directors,self.tangents)

        matmulsecond = np.zeros((3,self.e))
        matmulsecond = np.einsum('ijk,jk->ik',self.S_hat,self.s)

        self.shear_stretch_internal_couple = np.cross(matmulfirst,matmulsecond)

        dw_dt = (self.bend_twist_internal_couple + self.shear_stretch_internal_couple)  / self.J
        return dw_dt

    def force_rule(self, temp_pos):
        # First update
        self.update(temp_pos)

        matmul = np.zeros((3,self.e))
        matmul = np.einsum('jil, jkl, kl -> il ', Q.T, self.S_hat, self.sigma)

        self.internal_force = grad_h(matmul / self.dil_fac, self.n)

        dv_dt = (self.internal_force + self.forces)  / self.m
        return dv_dt

    def update(self, temp_pos):
        # Constrain 1st node position
        temp_pos[:,0] = 0

        # Update Length
        self.l = temp_pos[:,1:] - temp_pos[:,:-1]
        self.l_mag = np.linalg.norm(self.l, axis = 0)

        # Update dilatation factor
        self.dil_fac = self.l_mag / self.l_ref_mag

        # Update tangents
        self.tangents = self.l / self.l_mag

        # Update shear/stress strain
        self.s = self.dil_fac * self.tangents - self.directors[2,:,:]
        pass

F = 15
E = 1E6
R = .1
A = np.pi * R**2
L = 1

T = 100
dt = 3E-4
test = crossrod(T = T, dt = dt, total_length = L, elements = 35, density = 5E3, radius = R, total_external_force = F)
print("position = {0}".format(test.pos))
real_strain = (F*L)/(E*A-F)
print(real_strain)
print(np.average(test.final_pos))
plt.plot(np.arange(0,T+dt,dt),test.final_pos)
plt.show()
