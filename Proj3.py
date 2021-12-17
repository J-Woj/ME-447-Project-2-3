import numpy as np

from elastica import *

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
    40,
    start,
    direction,
    normal,
    1.6,
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


origin_force = np.array([0.0, 0.0, 0.0])
end_force_1 = np.array([20.0, 0.0, 0.0])
end_force_2 = np.array([-100.0, 0.0, 0.0])
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


rod_4 = CosseratRod.straight_rod(
    40,
    start,
    direction,
    normal,
    1.6,
    base_radius,
    density,
    nu,
    E,
    shear_modulus = shear_modulus,
)

timoshenko_sim.append(rod_4)
timoshenko_sim.constrain(rod_4).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

timoshenko_sim.add_forcing_to(rod_4).using(
    EndpointForces, origin_force, end_force_4, ramp_up_time=ramp_up_time
)


timoshenko_sim.connect(rod_3,rod_4,first_connect_idx=39, second_connect_idx=39).using(FixedJoint, k=1e6, nu=0.2, kt=5e4)

timoshenko_sim.finalize()

final_time = 20.0
dl = base_length / n_elem
dt = 0.01 * dl
total_steps = int(final_time / dt)
print("Total steps to take", total_steps)

timestepper = PositionVerlet()

integrate(timestepper, timoshenko_sim, final_time, total_steps)

def analytical_result(arg_rod, arg_end_force, shearing=True, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)
    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = (
        -acting_force
        / 2.0
        * np.sum(arg_rod.rest_lengths / arg_rod.bend_matrix[0, 0, 0])
    )
    cubic_prefactor = (acting_force / 6.0) / arg_rod.bend_matrix[0, 0, 0]
    if shearing:
        return (
            arg_s,
            arg_s * linear_prefactor
            + arg_s ** 2 * quadratic_prefactor
            + arg_s ** 3 * cubic_prefactor,
        )
    else:
        return arg_s, arg_s ** 2 * quadratic_prefactor + arg_s ** 3 * cubic_prefactor
    
# def plot_timoshenko(rod_1, rod_2, end_force_1, end_force_2):
#     import matplotlib.pyplot as plt

#     analytical_1_positon = analytical_result(
#         rod_1, end_force_1, shearing=True
#     )
#     analytical_2_positon = analytical_result(
#         rod_2, end_force_2, shearing=True
#     )
    
#     analytical_3_positon = analytical_result(
#         rod_3, end_force_3, shearing=True
#     )
    
#     analytical_4_positon = analytical_result(
#         rod_4, end_force_4, shearing=True
#     )
    
#     fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
#     ax = fig.add_subplot(111)
#     ax.grid(b=True, which="major", color="grey", linestyle="-", linewidth=0.25)


#     ax.plot(
#         rod_1.position_collection[1, :],
#         rod_1.position_collection[0, :],
#         "b-",
#         label="Rod 1 n=" + str(rod_1.n_elems),
#     )
#     ax.plot(
#         rod_2.position_collection[1, :],
#         rod_2.position_collection[0, :],
#         "r-",
#         label="Rod 2 n=" + str(rod_2.n_elems),
#     )
#     ax.plot(
#         rod_3.position_collection[1, :],
#         rod_3.position_collection[0, :],
#         "y-",
#         label="Combined n=" + str(rod_3.n_elems),
#     )
#     ax.plot(
#         rod_4.position_collection[1, :],
#         rod_4.position_collection[0, :],
#         "x-",
#         label="Combined n=" + str(rod_4.n_elems),
#     )
    
#     ax.legend(prop={"size": 12})
#     ax.set_ylabel("X Position (m)", fontsize=12)
#     ax.set_xlabel("Y Position (m)", fontsize=12)
#     plt.show()

# plot_timoshenko(rod_1, rod_2, end_force_1, end_force_2)

def plot_timoshenko(rod_1, rod_2, end_force_1, end_force_2):
    import matplotlib.pyplot as plt

    analytical_1_positon = analytical_result(
        rod_1, end_force_1, shearing=True
    )
    analytical_2_positon = analytical_result(
        rod_2, end_force_2, shearing=True
    )
    
    analytical_3_positon = analytical_result(
        rod_3, end_force_3, shearing=True
    )
    
    analytical_4_positon = analytical_result(
        rod_4, end_force_4, shearing=True
    )
    

    fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(b=True, which="major", color="grey", linestyle="-", linewidth=0.25)


    ax.plot(
        rod_1.position_collection[2, :],
        rod_1.position_collection[0, :],
        "b-",
        label="Rod 1 n=" + str(rod_1.n_elems),
    )
    ax.plot(
        rod_2.position_collection[2, :],
        rod_2.position_collection[0, :],
        "r-",
        label="Rod 2 n=" + str(rod_2.n_elems),
    )
    ax.plot(
        rod_3.position_collection[2, :],
        rod_3.position_collection[0, :],
        "y-",
        label="Combined n=" + str(rod_3.n_elems),
    )
    # ax.plot(
    #     rod_4.position_collection[2, :],
    #     rod_4.position_collection[0, :],
    #     "x-",
    #     label="Combined n=" + str(rod_4.n_elems),
    # )
    
    ax.legend(prop={"size": 12})
    ax.set_ylabel("Z Position (m)", fontsize=12)
    ax.set_xlabel("Y Position (m)", fontsize=12)
    plt.show()


plot_timoshenko(rod_1, rod_2, end_force_1, end_force_2)
