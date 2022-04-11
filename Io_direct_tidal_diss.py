import tudatpy.kernel.numerical_simulation.propagation_setup.integrator
from tudatpy.kernel.numerical_simulation.propagation import get_state_of_bodies
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
import numpy as np
import json
import pandas as pd
from pandas import DataFrame
################################################################################
# GENERAL SIMULATION SETUP #####################################################
################################################################################

    # Load spice kernels.
spice.load_standard_kernels()

    # Set simulation start epoch.
simulation_start_epoch = 1.0e7

    # Set simulation end epoch.
simulation_end_epoch = 1.0e7 + 10.0 * constants.JULIAN_YEAR

################################################################################
# SETUP ENVIRONMENT ############################################################
################################################################################

    # Define bodies in simulation.
bodies_to_create = ["Io","Jupiter"]

    # Create bodies in simulation.
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create,global_frame_origin,global_frame_orientation)

# Rotation model settings
body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous(
"Jupiter", global_frame_orientation, "Io_Fixed")

#Gravity settings
body_settings.get("Io").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic_triaxial_body(
    axis_a=1830000,
    axis_b=1818700,
    axis_c=1815300,
    density=3528,
    maximum_degree=2,
    maximum_order=2,
    associated_reference_frame="IAU_Io")

body_system = environment_setup.create_system_of_bodies(body_settings)

################################################################################
# SETUP PROPAGATION ############################################################
################################################################################
bodies_to_propagate = ["Io"]
central_bodies = ["Jupiter"]

# Define accelerations acting on Io
# Define tidal parameters
love_number_io = 0.7
love_number_jup = 0.379
time_lag_io = 517.5 # 7827.4
time_lag_jup = 0.104
# Add entry to acceleration settings dict
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.point_mass_gravity(),
        #       propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_io,time_lag_io,
        #                                                                       False, False), # Tide on satellite
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jup, time_lag_jup,
                                                                               False, True)  # Tide on planet
               ])

# Create global accelerations settings dictionary

acceleration_settings = {"Io": acceleration_settings_io}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    body_system,acceleration_settings,bodies_to_propagate,central_bodies
)

    ############################################################################
    # SETUP PROPAGATION : PROPAGATION SETTINGS #################################
    ############################################################################

    # Get system initial state.
system_initial_state = propagation.get_initial_state_of_bodies(
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies,
    body_system=body_system,
    initial_time=simulation_start_epoch,
)
    # Create termination settings.
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

    #create dependent variables
dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state("Io","Jupiter")
]

    # Create propagation settings.
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    termination_condition,
    output_variables = dependent_variables_to_save
)

    # Create numerical integrator settings.
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(simulation_start_epoch,200.0,
    tudatpy.kernel.numerical_simulation.propagation_setup.integrator.rkf_78,
   200.0, 200.0, 100000000.0, 100000000.0, 10, False, 0.8, 4.0, 0.1, True)

    ############################################################################
    # PROPAGATE ################################################################
    ############################################################################

    # Instantiate the dynamics simulator.
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    body_system, integrator_settings, propagator_settings)

    # Propagate and store results
states = dynamics_simulator.state_history
dep_var = dynamics_simulator.dependent_variable_history
states_array = result2array(states)

################################################################################
# VISUALISATION  ################################
################################################################################
from matplotlib import pyplot as plt

def plot_multi_body_system_state_history(states_array, bodies_to_propagate):
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title(f'Trajectory of Io w.r.t Jupiter')
    ax1.scatter(0, 0, 0, marker='x', label="Jupiter", color='black')
    ax1.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-.')

    ax1.legend()
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_zlim([-4E8, 4E8])
    ax1.set_aspect('auto', 'box')
    return fig1
figA = plot_multi_body_system_state_history(states_array, bodies_to_propagate)
plt.tight_layout()
#plt.show()

df_array = pd.DataFrame(data=states_array)

################################################################################
# KEPLERIAN ELEMENTS  ##########################################################
################################################################################

dep_var_array = result2array(dep_var)

#np.savetxt("output_spherical_Jupiter_and_tides.dat", dep_var_array)

time = dep_var_array[:,0]
time_step = time-1.0e7
time_day = time_step / (3600*24*365)

dep_var_array = pd.DataFrame(data=dep_var_array, columns ="t a e i Argument_periapsis RAAN true_anomaly".split())

fig, ((ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Kepler elements of Io during the propagation with tides on the planet, k2 = 0.7')

eccentricity = dep_var_array.loc[:,"e"]
#SEMI MAJOR AXIS
semi_major_axis = dep_var_array.loc[:,"a"]
c = 1.198e-17
D = 7588.2
dadt = 2/3*c*(1-7*D*(eccentricity[0])**2)*semi_major_axis[0]
yacc=semi_major_axis[0] + dadt*time
ax2.plot(time_day, semi_major_axis, 'r', label = "Simulation")
ax2.plot(time_day, yacc, 'g', label = "Theoretical")
ax2.legend(loc="upper left")
ax2.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([410000*1e3, 430000*1e3])

#ECCENTRICITY
dedt=-7/3*c*D*eccentricity[0]
yecc=eccentricity[0] + dedt*time
ax3.plot(time_day, eccentricity,'r', label="Simulation")
ax3.plot(time_day, yecc,'g', label="Theoretical")
ax3.set_ylabel('Eccentricity [-]')
ax3.legend(loc="upper right")

#INCLINATION
inclination = np.rad2deg(dep_var_array.loc[:,"i"])
ax4.plot(time_day, inclination)
ax4.set_ylabel('inclination [deg]')
#ax4.set_ylim(2.201, 2.203)

#RAAN
raan = np.rad2deg(dep_var_array.loc[:,"RAAN"])
ax5.plot(time_day, raan)
ax5.set_ylabel('RAAN [deg]')

#Argument of Pericenter
argument_of_pericenter = np.rad2deg(dep_var_array.loc[:,"Argument_periapsis"])
ax6.plot(time_day, argument_of_pericenter)
ax6.set_ylabel("Argument of pericenter [deg]")

#True Anomaly
true_anomaly = np.rad2deg(dep_var_array.loc[:,"true_anomaly"])
ax7.plot(time_day, true_anomaly)
ax7.set_ylabel("True anomaly [deg]")

for ax in fig.get_axes():
    ax.set_xlabel('Time [Years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
plt.show()
