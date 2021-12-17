import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from numpy.random import multivariate_normal
import copy

from elastica import *

# Input problem here
def problem(force_1, force_2):
    class TimoshenkoBeamSimulator(BaseSystemCollection, Constraints, Forcing, Connections):
        pass

    timoshenko_sim = TimoshenkoBeamSimulator()

    n_elem = 100
    density = 1000
    nu = 0.1
    E = 1e6
    poisson_ratio = 0.31
    shear_modulus = E / (poisson_ratio + 1.0)

    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 4
    base_radius = 0.25
    base_area = np.pi * base_radius ** 2

    rod_1 = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus,
    )

    timoshenko_sim.append(rod_1)
    timoshenko_sim.constrain(rod_1).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    rod_2 = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus,
    )

    timoshenko_sim.append(rod_2)
    timoshenko_sim.constrain(rod_2).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    rod_3 = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus,
    )

    timoshenko_sim.append(rod_3)
    timoshenko_sim.constrain(rod_3).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    rod_4 = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        nu,
        E,
        shear_modulus,
    )

    timoshenko_sim.append(rod_4)
    timoshenko_sim.constrain(rod_4).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    timoshenko_sim.connect(rod_3,rod_4,first_connect_idx=99, second_connect_idx=99).using(FixedJoint, k=1e6, nu=0.2, kt=5e4)

    origin_force = np.array([0.0, 0.0, 0.0])
    end_force_1 = np.array([5.0, 0.0, 0.0])
    end_force_2 = np.array([0.0, -20.0, 0.0])
    end_force_3 = end_force_1
    end_force_4 = end_force_2
    ramp_up_time = 5.0

    timoshenko_sim.add_forcing_to(rod_1).using(
        EndpointForces, origin_force, end_force_1, ramp_up_time=ramp_up_time
    )
    timoshenko_sim.add_forcing_to(rod_2).using(
        EndpointForces, origin_force, end_force_2, ramp_up_time=ramp_up_time
    )
    timoshenko_sim.add_forcing_to(rod_3).using(
        EndpointForces, origin_force, end_force_3, ramp_up_time=ramp_up_time
    )
    timoshenko_sim.add_forcing_to(rod_4).using(
        EndpointForces, origin_force, end_force_4, ramp_up_time=ramp_up_time
    )

    timoshenko_sim.finalize()

    final_time = 20.0
    dl = base_length / n_elem
    dt = 0.01 * dl
    total_steps = int(final_time / dt)
    print("Total steps to take", total_steps)

    timestepper = PositionVerlet()

    integrate(timestepper, timoshenko_sim, final_time, total_steps)

    desired_position = [1.,2.,-8.]

    return rod_3.position_collection[:,-1] - desired_position

from functools import partial
current_problem = partial(problem)

# CMAES Class Definition
class CMAES:
    """Naive CMA implementation"""

    def __init__(self, initial_mean, sigma, popsize, **kwargs):
        """Please do all the initialization. The reserve space and
        code for collecting the statistics are already provided."""

        # Things that evolve : centroid, sigma, paths etc.
        self.centroid = np.asarray(initial_mean).copy()
        self.sigma = sigma
        self.pc = np.zeros_like(initial_mean)
        self.ps = np.zeros_like(initial_mean)
        self.C = np.eye(initial_mean.shape[0])
        self.B = np.eye(self.C.shape[0])
        self.diagD = np.ones(initial_mean.shape[0])

        # Optimal popsize
        self.popsize = popsize
        self.mu = popsize // 2

        # Update weights later on
        # Constant weight policy
        # self.weights = np.ones((self.mu, )) / self.mu

        # Decreasing weight policy
        self.weights = np.arange(self.mu, 0.0, -1.0)
        self.weights /= np.sum(self.weights)

        # Negative, Positive weight policy
        # unscaled_weights = np.arange(1.0 ,  1.0 + popsize)
        # unscaled_weights = np.log(0.5 * (popsize + 1.0) / unscaled_weights)

        # Utility variables
        self.dim = initial_mean.shape[0]

        # Expectation of a normal distribution
        self.chiN = np.sqrt(self.dim) * (1.0 - 0.25 / self.dim + 1.0/(21.0 * self.dim**2))
        self.mueff = 1.0 / np.linalg.norm(self.weights, 2)**2
        self.generations = 0

        # Options

        # Sigma adaptation
        # cs is short for c_sigma
        self.cs = kwargs.get("cs", (2.0 + self.mueff) / (self.dim + self.mueff + 5.0))
        # ds is short for d_sigma
        self.ds = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0)/ (self.dim + 1.0)) - 1.0) + self.cs

        # Covariance adaptation
        self.cc = kwargs.get("cc", (4.0 + self.mueff/self.dim) / (self.dim + 4.0 + 2.0 * self.mueff/self.dim))
        self.ccov = 0.0
        # If implementing the latest version of CMA according to the tutorial,
        # these parameters can be useful
        self.ccov1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.ccovmu = min(1.0 - self.ccov1, 2.0 * (self.mueff - 2.0 + 1.0/self.mueff)/((self.dim + 2.0)**2 + self.mueff))

        self.stats_centroids = []
        self.stats_new_centroids = []
        self.stats_covs = []
        self.stats_new_covs = []
        self.stats_offspring = []
        self.stats_offspring_weights = []
        self.stats_ps = []

    def update(self, problem, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        # -- store current state of the algorithm
        self.stats_centroids.append(copy.deepcopy(self.centroid))
        self.stats_covs.append(copy.deepcopy(self.C))

        population.sort(key=lambda ind: problem(*ind))
        # population.sort(key=lambda ind: problem(ind[0], ind[1]))
        # population.sort(key=problem)

        # -- store sorted offspring
        self.stats_offspring.append(copy.deepcopy(population))

        old_centroid = self.centroid
        # Note : the following does m <- <x>_w
        # Note : this is equivalent to doing m <- m + sigma * <z>_w
        # as x = m + sigma * z provided the weights sum to 1.0 which it
        # does
        self.centroid = np.dot(self.weights, population[0:self.mu])

        # -- store new centroid
        self.stats_new_centroids.append(copy.deepcopy(self.centroid))

        c_diff = self.centroid - old_centroid

        # Cumulation : update evolution path
        # Equivalent to in-class definition
        self.ps = (1 - self.cs) * self.ps \
             + np.sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
             * np.dot(self.B, (1. / self.diagD) * np.dot(self.B.T, c_diff))

        # -- store new evol path
        self.stats_ps.append(copy.deepcopy(self.ps))

        hsig = float((np.linalg.norm(self.ps) /
                np.sqrt(1. - (1. - self.cs)**(2. * (self.generations + 1.))) / self.chiN
                < (1.4 + 2. / (self.dim + 1.))))

        self.pc = (1 - self.cc) * self.pc + hsig \
                  * np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
                  * c_diff

        # Update covariance matrix
        artmp = population[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) \
                   * self.ccov1 * self.cc * (2 - self.cc)) * self.C \
                + self.ccov1 * np.outer(self.pc, self.pc) \
                + self.ccovmu * np.dot((self.weights * artmp.T), artmp) \
                / self.sigma**2

        # -- store new covs
        self.stats_new_covs.append(copy.deepcopy(self.C))

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1.) \
                                * self.cs / self.ds)

        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]]/self.diagD[indx[0]]

        self.diagD = self.diagD[indx]**0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def run(self, problem):
        # At the start, clear all stored cache and start a new campaign
        self.reset()
        while self.generations < 200:
            #Appending to array for graphing later
            sig_val.append(self.sigma)
            gen_val.append(self.generations)
            # Sample the population here!
            population = list(multivariate_normal(self.centroid, self.sigma**2 * self.C, self.popsize))
            # Pass the population to update, which computes all new parameters
            self.update(problem, population)
            # print(np.array(population).shape)
            self.generations += 1
        else:
            return population[0]

    def reset(self):
        # Clears everything to rerun the problem
        self.stats_centroids = []
        self.stats_new_centroids = []
        self.stats_covs = []
        self.stats_new_covs = []
        self.stats_offspring = []
        self.stats_offspring_weights = []
        self.stats_ps = []

# For graphing
sig_val = []
gen_val = []

# Running CMA on Problem
initial_centroid = np.random.randn(2)
cma_es = CMAES(initial_centroid, 0.5, 10)
result = cma_es.run(current_problem)
print(result)

# Plotting to find the smallest sigma value
plt.clf()
plt.plot(gen_val, sig_val)
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.title('Sigma Plot')
