import tudatpy.kernel.numerical_simulation.propagation_setup.integrator
from tudatpy.kernel.numerical_simulation.propagation import get_state_of_bodies
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, environment
from tudatpy import bodies
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
import numpy as np
from tudatpy.kernel.astro import frame_conversion, element_conversion
import json
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

################################################################################
# GENERAL SIMULATION SETUP #####################################################
################################################################################

# Load spice kernels.
spice.load_standard_kernels()
simulation_start_epoch = 1.0e7
simulation_end_epoch = simulation_start_epoch + 1.774*2000 * constants.JULIAN_DAY
# Define bodies in simulation.
bodies_to_create = ["Io","Jupiter"]
# Create bodies in simulation.
global_frame_origin = "Jupiter"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)
#body_system = environment_setup.create_system_of_bodies(body_settings)

# Spherical harmonics variation in time
tide_raising_body = "Jupiter"
#love_number_Io = complex(0.7, -0.015)
#love_number_Jup = complex(0.379, -1.102e-5)
love_numbers = dict()
love_numbers[ 2 ] = list()
love_numbers[ 2 ].append(complex(0.0, -0.0))
love_numbers[ 2 ].append(complex(0.0, -0.0))
love_numbers[ 2 ].append(complex(0.04, -0.015))
gravity_field_variation_list = list()
gravity_field_variation_list.append(environment_setup.gravity_field_variation.solid_body_tide_degree_order_variable_complex_k(
     tide_raising_body, love_numbers))
body_settings.get("Io").gravity_field_variation_settings = gravity_field_variation_list

#Change gravity field settings
sine_coefficients_io = [
    [0,                   0,                   0], #[degree 0]
    [0,                   0,                   0], #[10, 11, 12]
    [0,                  -0.0,                 -9.93746153432869e-06]] #[20, 21, 22]   -9.948105019330582e-06 for librations of 10
cosine_coefficients_io = [
    [1,                   0,                   0], #[degree 0]
    [0,                   0,                   0], #[10, 11, 12]
    [0,                   0,                   0]] #[20, 21, 22]
sine_coefficients_jup = [
    [0,                   0,                   0], #[degree 0]
    [0,                   0,                   0], #[10, 11, 12]
    [0,                   0,                   0]] #[20, 21, 22]
cosine_coefficients_jup = [
    [1,                   0,                   0],
    [0,                   0,                   0],
    [0,                   0,                   0]]

body_settings.get("Io").gravity_field_settings.normalized_cosine_coefficients = cosine_coefficients_io
body_settings.get("Io").gravity_field_settings.normalized_sine_coefficients = sine_coefficients_io
body_settings.get("Jupiter").gravity_field_settings.normalized_cosine_coefficients = cosine_coefficients_jup
body_settings.get("Jupiter").gravity_field_settings.normalized_sine_coefficients = sine_coefficients_jup

# Synchronous rotation model
body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter",global_frame_orientation,"IAU_Io")

body_system = environment_setup.create_system_of_bodies(body_settings)

# Librations
scaled_libration_amplitude = -6.66
libration_calculator = environment.DirectLongitudeLibrationCalculator(scaled_libration_amplitude)
body_system.get("Io").rotation_model.libration_calculator = libration_calculator

jupiter_gravitational_parameter = body_system.get("Jupiter").gravitational_parameter
io_gravitational_parameter = body_system.get("Io").gravitational_parameter

################################################################################
# SETUP PROPAGATION ############################################################
################################################################################
bodies_to_propagate = ["Io"]
central_bodies = ["Jupiter"]

# Define accelerations acting on the moons
# Add entry to acceleration settings dict
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,2,
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

# Create dependent variables
dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state("Io","Jupiter"),
    propagation_setup.dependent_variable.latitude("Jupiter","Io"),
    propagation_setup.dependent_variable.longitude("Jupiter","Io"),
    propagation_setup.dependent_variable.total_spherical_harmonic_sine_coefficien_variations("Io",2,2,0,2), #S20 S21 S22
    propagation_setup.dependent_variable.total_spherical_harmonic_cosine_coefficien_variations("Io",2,2,0,2), #C20, C21, C22
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

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(simulation_start_epoch,300.0,
    tudatpy.kernel.numerical_simulation.propagation_setup.integrator.rkf_78,
   300.0, 300.0, 100000000.0, 100000000.0, 10, False, 0.8, 4.0, 0.1, True)

    ############################################################################
    # PROPAGATE ################################################################
    ############################################################################

# Instantiate the dynamics simulator.
dynamics_simulator = numerical_simulation.SingleArcSimulator(body_system, integrator_settings, propagator_settings)

# Propagate and store results
states = dynamics_simulator.state_history
dep_var = dynamics_simulator.dependent_variable_history
states_array = result2array(states)

df_array = pd.DataFrame(data=states_array)

################################################################################
# KEPLERIAN ELEMENTS  ##########################################################
################################################################################

dep_var_array = result2array(dep_var)

#np.savetxt("Acceleration.dat",total_acceleration_norm)
#np.savetxt("out_mutual_spherical_tidessat.dat", dep_var_array)
#np.savetxt("out_mutual_spherical.dat", dep_var_array)

time = dep_var_array[:,0]
time_step = time-1.0e7
time_day = time_step / (3600*24*365)

dep_var_array = pd.DataFrame(data=dep_var_array, columns ="t a e i Argument_periapsis RAAN true_anomaly Lat Lon S20 S21 S22 C20 C21 C22".split())

fig, ((ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Kepler elements variation of Io due to tide raised on Io with librations')

#SEMI MAJOR AXIS
c = 1.33e-17##1.15986e-17#
D = 6603
semi_major_axis = dep_var_array.loc[:,"a"]
eccentricity = dep_var_array.loc[:,"e"]

dadt = -14/3*c*D*semi_major_axis*(eccentricity)**2 # Satellite
#dadt = (2/3)*c*semi_major_axis[0] # Planet
#dedt = (57*0.379*8.931938e22*2*np.pi/152853.5)*eccentricity[0]*((69911000/semi_major_axis[0])**5)/(35600*8*1.8982e27) # Planet
dedt=-7/3*c*D*eccentricity # Satellite (+ Planet)


yacc = semi_major_axis[0] + dadt*time
ax2.plot(time_day, semi_major_axis, 'r', label = "Simulation")
ax2.plot(time_day, yacc, 'b', label = "Theoretical")
ax2.legend()
ax2.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([410000*1e3, 430000*1e3])

#ECCENTRICITY
yecc= eccentricity[0] + dedt*time
ax3.plot(time_day, eccentricity, 'r', label = "Simulation")
ax3.plot(time_day, yecc, 'b', label= "Theoretical")
ax3.legend()
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

S22 = dep_var_array.loc[:,"S22"]
C22 = dep_var_array.loc[:,"C22"]
S21 = dep_var_array.loc[:,"S21"]
S20 = dep_var_array.loc[:,"S20"]

C = 0.6455*3/5*(jupiter_gravitational_parameter/io_gravitational_parameter)*((1821.6e3/semi_major_axis)**3)
k2 = 0.7
k2Q = 0.015
latitude = (dep_var_array.loc[:,"Lat"])
longitude = (dep_var_array.loc[:,"Lon"])

DeltaC22 = C*(k2*np.cos(2*longitude) - k2Q*np.sin(2*longitude))
DeltaS22 = C*(k2*np.sin(2*longitude) + k2Q*np.cos(2*longitude))

coeff=np.array([DeltaC22,S22])
coeff=coeff.transpose()
coeff=pd.DataFrame(data=coeff, columns="DS22 DS22_sim".split())

plt.figure(figsize=(10,6))
plt.title("Spherical harmonic coefficients variation over time")
plt.plot(time_day, S22, 'r--', label = "Simulated S22")
plt.plot(time_day, DeltaS22, 'b--', label = "Theoretical S22")

plt.plot(time_day, C22, 'g--', label = "Simulated C22")
plt.plot(time_day, DeltaC22, 'm--', label = "Theoretical C22")

plt.legend(loc = "right")
plt.ylabel("C22, S22")
plt.xlabel("Time [years]")
plt.grid()
#plt.tight_layout()
plt.xlim([min(time_day), max(time_day)])
#plt.show()

average=sum(S22)/len(S22)
print(average)

differenceC22=C22-DeltaC22
differenceS22=S22-DeltaS22

plt.figure(figsize=(10,6))
plt.title("DeltaS22 difference over time")
plt.plot(time_day, differenceC22, 'b', label="C22")
plt.plot(time_day, differenceS22, 'r', label = "S22")
plt.legend(loc = "upper right")
plt.xlabel("Time [years]")
plt.grid()
plt.xlim([min(time_day), max(time_day)])
#plt.show()

longitude = np.rad2deg(longitude)
plt.figure(figsize=(10,6))
plt.title("Longitude over time")
plt.plot(time_day, longitude, 'g')
plt.xlabel("Time [years]")
plt.grid()
plt.xlim([min(time_day), max(time_day)])
plt.show()

#np.savetxt("coefficients.txt", coeff)
