import tudatpy.kernel.numerical_simulation.propagation_setup.integrator
from tudatpy.kernel.numerical_simulation.propagation import get_state_of_bodies
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy import bodies
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

simulation_start_epoch = 1.0e7

simulation_end_epoch = 1.0e7 + 0.1 * constants.JULIAN_YEAR

################################################################################
# SETUP ENVIRONMENT ############################################################
################################################################################

    # Define bodies in simulation.
bodies_to_create = ["Io","Jupiter"]

    # Create bodies in simulation.
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create,global_frame_origin,global_frame_orientation)

# Spherical harmonics variation in time
gravity_field_variation_settings = list()
tide_raising_body = "Jupiter"
degree = 2
love_number_Io = complex(0.7, -0.015)
love_number_Jup = complex(0.379, -1.102e-5)
gravity_field_variation_settings.append(environment_setup.gravity_field_variation.solid_body_tide_complex_k(
   tide_raising_body, love_number_Io, degree))
#body_settings.get("Io").gravity_field_variation_settings = gravity_field_variation_settings

#Change gravity field settings
gravity_field_settings = list()
normalized_cosine_coefficients = [
    [1,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0]]

normalized_sine_coefficients = [
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0]]
gravity_field_settings.append(environment_setup.gravity_field.SphericalHarmonicsGravityFieldSettings(
    "Io", True, 5.958e12, normalized_cosine_coefficients, normalized_sine_coefficients))
body_settings.get("Io").gravity_field_settings = gravity_field_settings

# Rotation model
initial_orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
initial_time = simulation_start_epoch
rotation_rate = 4.1106e-5
original_frame = global_frame_orientation
target_frame = "Io"
#body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.simple(
#    original_frame, target_frame, initial_orientation, initial_time, rotation_rate)

body_system = environment_setup.create_system_of_bodies(body_settings)

# Librations
scaled_libration_amplitude = 500.0 #1.378e-4
libration_calculator = environment.DirectLongitudeLibrationCalculator(scaled_libration_amplitude)
#body_system.get("Io").rotation_model.libration_calculator = libration_calculator

################################################################################
# SETUP PROPAGATION ############################################################
################################################################################
bodies_to_propagate = ["Io"]
central_bodies = ["Jupiter"]

# Define accelerations acting on the moons
# Add entry to acceleration settings dict
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,0,
        2,2)]
    )
# Create global accelerations settings dictionary

acceleration_settings = {"Io": acceleration_settings_io}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    body_system,acceleration_settings,bodies_to_propagate,central_bodies)

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
    propagation_setup.dependent_variable.keplerian_state("Io","Jupiter"),
    propagation_setup.dependent_variable.latitude("Jupiter","Io"),
    propagation_setup.dependent_variable.longitude("Jupiter","Io")
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

integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(simulation_start_epoch,300.0,
    tudatpy.kernel.numerical_simulation.propagation_setup.integrator.rkf_78,
   300.0, 300.0, 100000000.0, 100000000.0, 10, False, 0.8, 4.0, 0.1, True)

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

#np.savetxt("out_mutual_spherical_tidessat.dat", dep_var_array)
#np.savetxt("out_mutual_spherical.dat", dep_var_array)

time = dep_var_array[:,0]
time_step = time-1.0e7
time_day = time_step / (3600*24*365)

dep_var_array = pd.DataFrame(data=dep_var_array, columns ="t a e i Argument_periapsis RAAN true_anomaly Lat Lon".split())

fig, ((ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements of Io during the propagation without tides')

#SEMI MAJOR AXIS
semi_major_axis = dep_var_array.loc[:,"a"]
ax2.plot(time_day, semi_major_axis)
ax2.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([410000*1e3, 430000*1e3])

#ECCENTRICITY
eccentricity = dep_var_array.loc[:,"e"]
ax3.plot(time_day, eccentricity)
ax3.set_ylabel('Eccentricity [-]')

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
    ax.set_xlabel('Time [years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
#plt.show()

fig2, (ax8, ax9) = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('Latitude and longitude of Io with librations')

latitude = dep_var_array.loc[:,"Lat"]
longitude = dep_var_array.loc[:,"Lon"]

#LATITUDE
ax8.plot(time_day, latitude,'r')
ax8.set_ylabel('Latitude')

#LONGITUDE
ax9.plot(time_day, longitude, 'b')
ax9.set_ylabel('Longitude')

for ax in fig2.get_axes():
    ax.set_xlabel('Time [years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
plt.show()
