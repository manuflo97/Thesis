import tudatpy.kernel.numerical_simulation.propagation_setup.integrator
from tudatpy.kernel.numerical_simulation.propagation import get_state_of_bodies
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, environment
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
global_frame_origin = "Jupiter"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create,global_frame_origin,global_frame_orientation)

# Rotation model settings
body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter", global_frame_orientation, "IAU_Io")

#Change gravity field settings
sine_coefficients = [
    [0,                   0,                   0], #[degree 0]
    [0,                   0,                   0], #[10, 11, 12]
    [0,                   0,                   0]] #[21, 21, 22]
cosine_coefficients = [
    [1,                   0,                   0],
    [0,                   0,                   0],
    [0,                   0,                   0]]

body_settings.get("Io").gravity_field_settings.normalized_cosine_coefficients = cosine_coefficients
body_settings.get("Io").gravity_field_settings.normalized_sine_coefficients = sine_coefficients
body_settings.get("Jupiter").gravity_field_settings.normalized_cosine_coefficients = cosine_coefficients
body_settings.get("Jupiter").gravity_field_settings.normalized_sine_coefficients = sine_coefficients

body_system = environment_setup.create_system_of_bodies(body_settings)

# Librations
scaled_libration_amplitude = -6.65
libration_calculator = environment.DirectLongitudeLibrationCalculator(scaled_libration_amplitude)
#ody_system.get("Io").rotation_model.libration_calculator = libration_calculator

################################################################################
# SETUP PROPAGATION ############################################################
################################################################################
bodies_to_propagate = ["Io"]
central_bodies = ["Jupiter"]

# Define accelerations acting on Io
# Define tidal parameters
love_number_io = 0.7
love_number_jup = 0.379
time_lag_io = 529.0
time_lag_jup = 0.1016
# Add entry to acceleration settings dict
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.spherical_harmonic_gravity(2,2),
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_io,time_lag_io,
                                                                             False, False), # Tide on satellite
           #    propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jup, time_lag_jup,
           #                                                                  False, True)  # Tide on planet
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
    propagation_setup.dependent_variable.keplerian_state("Io","Jupiter"),
    propagation_setup.dependent_variable.latitude("Jupiter","Io"),
    propagation_setup.dependent_variable.longitude("Jupiter","Io"),
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
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(simulation_start_epoch, 300.0,
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

#np.savetxt("output_spherical_Jupiter_and_tides.dat", dep_var_array)

time = dep_var_array[:,0]
time_step = time-1.0e7
time_day = time_step / (3600*24*365)

dep_var_array = pd.DataFrame(data=dep_var_array, columns ="t a e i Argument_periapsis RAAN true_anomaly Lat Lon".split())

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Kepler elements of Io during the propagation with both tides, k2 = 0.7')

#THEORETICAL BEHAVIOR
eccentricity = dep_var_array.loc[:,"e"]
semi_major_axis = dep_var_array.loc[:,"a"]

c = 1.34e-17
D = 6603

#dadt = 2/3*c*(1-7*D*(eccentricity[0])**2)*semi_major_axis[0] # Planet + Satellite

#dadt = (2/3)*c*semi_major_axis[0] # Planet
#dedt = (57*0.379*8.931938e22*2*np.pi/152853.5)*eccentricity[0]*((69911000/semi_major_axis[0])**5)/(35600*8*1.8982e27) # Planet

dadt = -14/3*c*D*semi_major_axis[0]*(eccentricity[0])**2 # Satellite
dedt = -7/3*c*D*eccentricity[0] # Satellite (+ Planet)

#dadt = (3*0.379*(69173000**5)*8.931938e22*2*np.pi/152853.5)*(1+12.75*eccentricity[0]**2)/(35600*1.8982e27*semi_major_axis[0]**4) # Planet
#dadt = -(21*0.7*(1.8982e27)*2*np.pi/152853.5*1821600**5)*eccentricity[0]**2/(47*2*8.931938e22*semi_major_axis[0]**4) # Sat
#dedt = -(21*0.7*1.8982e27*2*np.pi/152853.5)*eccentricity[0]*((1821600/semi_major_axis[0])**5)/(2*47*8.931938e22) # Sat

#SEMI MAJOR AXIS
yacc = semi_major_axis[0] + dadt*time
ax2.plot(time_day, semi_major_axis, 'r', label = "Simulation")
ax2.plot(time_day, yacc, 'b', label = "Theoretical")
ax2.legend(loc="upper left")
ax2.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([4.22020296*1e8, 4.22020299*1e8])

#ECCENTRICITY
yecc= eccentricity[0] + dedt*time
ax3.plot(time_day, eccentricity,'r', label = "Simulation")
ax3.plot(time_day, yecc,'b', label = "Theoretical")
ax3.set_ylabel('Eccentricity [-]')
ax3.legend(loc="upper right")
#ax3.set_ylim([0.0036393, 0.003640])

#INCLINATION
inclination = np.rad2deg(dep_var_array.loc[:,"i"])
#ax4.plot(time_day, inclination)
#ax4.set_ylabel('inclination [deg]')
#ax4.set_ylim(2.201, 2.203)

#RAAN
raan = np.rad2deg(dep_var_array.loc[:,"RAAN"])
#ax5.plot(time_day, raan)
#ax5.set_ylabel('RAAN [deg]')

#Argument of Pericenter
#argument_of_pericenter = np.rad2deg(dep_var_array.loc[:,"Argument_periapsis"])
#ax6.plot(time_day, argument_of_pericenter)
#ax6.set_ylabel("Argument of pericenter [deg]")

#True Anomaly
#true_anomaly = np.rad2deg(dep_var_array.loc[:,"true_anomaly"])
#ax7.plot(time_day, true_anomaly)
#ax7.set_ylabel("True anomaly [deg]")

for ax in fig.get_axes():
    ax.set_xlabel('Time [Years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
#plt.show()

fig2, (ax8, ax9) = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('Latitude and longitude of Io during the propagation')

latitude = np.rad2deg(dep_var_array.loc[:,"Lat"])
longitude = np.rad2deg(dep_var_array.loc[:,"Lon"])

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
