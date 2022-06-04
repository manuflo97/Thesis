import tudatpy.kernel.numerical_simulation.propagation_setup.integrator
from tudatpy.kernel.numerical_simulation.propagation import get_state_of_bodies
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel.astro import element_conversion
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
bodies_to_create = ["Io","Jupiter","Europa"]

    # Create bodies in simulation.
global_frame_origin = "Jupiter"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create,global_frame_origin,global_frame_orientation)

# Rotation model settings
body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter", global_frame_orientation, "IAU_Io")
body_settings.get("Europa").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter", global_frame_orientation, "IAU_Europa")

body_system = environment_setup.create_system_of_bodies(body_settings)

#Initial state
#Io
initial_cartesian_state_io=get_state_of_bodies(["Io"], ["Jupiter"], body_system, simulation_start_epoch)
jupiter_gravitational_parameter = body_system.get("Jupiter").gravitational_parameter
initial_keplerian_state_io=element_conversion.cartesian_to_keplerian(initial_cartesian_state_io, jupiter_gravitational_parameter)
a0_io=initial_keplerian_state_io[0]
n0_io=np.sqrt(jupiter_gravitational_parameter/(a0_io**3))

#Europa
initial_cartesian_state_eur=get_state_of_bodies(["Europa"], ["Jupiter"], body_system, simulation_start_epoch)
initial_keplerian_state_eur=element_conversion.cartesian_to_keplerian(initial_cartesian_state_io, jupiter_gravitational_parameter)
a0_eur=initial_keplerian_state_eur[0]
n0_eur=np.sqrt(jupiter_gravitational_parameter/(a0_eur**3))

################################################################################
# SETUP PROPAGATION ############################################################
################################################################################
bodies_to_propagate = ["Io","Europa"]
central_bodies = ["Jupiter","Jupiter"]

# Define accelerations acting on the moons
# Define tidal parameters
love_number_io = 0.7
love_number_jup = 0.379
time_lag_io = 529.0 # 7827.4
time_lag_jup_io = 0.104
love_number_eur = 0.2
time_lag_eur = 1.0724e3
time_lag_jup_eur = 0.09

# Add entry to acceleration settings dict
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.point_mass_gravity(),
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_io,time_lag_io,
                                                                             False, False), # Tide on Io by Jupiter
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jup, time_lag_jup_io,
                                                                             False, True)  # Tide on Jupiter by Io
],
   Europa = [propagation_setup.acceleration.point_mass_gravity()])

acceleration_settings_europa = dict(
    Jupiter = [propagation_setup.acceleration.point_mass_gravity(),
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_eur,time_lag_eur,
                                                                             False, False), # Tide on Europa by Jupiter
               propagation_setup.acceleration.direct_tidal_dissipation_acceleration(love_number_jup, time_lag_jup_eur,
                                                                             False, True)  # Tide on Jupiter by Europa
],
    Io = [propagation_setup.acceleration.point_mass_gravity()])

# Create global accelerations settings dictionary

acceleration_settings = {"Io": acceleration_settings_io,
                         "Europa": acceleration_settings_europa
                         }

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
    propagation_setup.dependent_variable.keplerian_state("Europa","Jupiter")
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
    ax1.set_title(f'Trajectory of Io and Europa w.r.t Jupiter')
    ax1.scatter(0, 0, 0, marker='x', label="Jupiter", color='black')
    ax1.plot(states_array[:,1], states_array[:,2], states_array[:,3], label=bodies_to_propagate[0]),
    ax1.plot(states_array[:,7], states_array[:,8],states_array[:,9], label=bodies_to_propagate[1] )

    ax1.legend()
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_zlim([-4E9, 4E9])
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

np.savetxt("out_all_tides.dat", dep_var_array)

time = dep_var_array[:,0]
time_step = time-1.0e7
time_day = time_step / (3600*24*365)

dep_var_array = pd.DataFrame(data=dep_var_array,
columns ="t a_Io e_Io i_Io Argument_periapsis_Io RAAN_Io true_anomaly_Io a_Eur e_Eur i_Eur Argument_periapsis_Eur RAAN_Eur true_anomaly_Eur".split())

fig, ((ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Kepler elements variation of Io')

#SEMI MAJOR AXIS
semi_major_axis_Io = dep_var_array.loc[:,"a_Io"]
ax2.plot(time_day, semi_major_axis_Io, 'r')
ax2.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([4.22020296*1e8, 4.22020299*1e8])

#ECCENTRICITY
eccentricity_Io = dep_var_array.loc[:,"e_Io"]
ax3.plot(time_day, eccentricity_Io,'r')
ax3.set_ylabel('Eccentricity [-]')
#ax3.set_ylim([0.0036393, 0.003640])

#INCLINATION
inclination_Io = np.rad2deg(dep_var_array.loc[:,"i_Io"])
ax4.plot(time_day, inclination_Io)
ax4.set_ylabel('inclination [deg]')
#ax4.set_ylim(2.201, 2.203)

#RAAN
raan_Io = np.rad2deg(dep_var_array.loc[:,"RAAN_Io"])
ax5.plot(time_day, raan_Io)
ax5.set_ylabel('RAAN [deg]')

#Argument of Pericenter
argument_of_pericenter_Io = np.rad2deg(dep_var_array.loc[:,"Argument_periapsis_Io"])
ax6.plot(time_day, argument_of_pericenter_Io)
ax6.set_ylabel("Argument of pericenter [deg]")

#True Anomaly
true_anomaly_Io = np.rad2deg(dep_var_array.loc[:,"true_anomaly_Io"])
ax7.plot(time_day, true_anomaly_Io)
ax7.set_ylabel("True anomaly [deg]")

for ax in fig.get_axes():
    ax.set_xlabel('Time [Years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
#plt.show()

fig2, ((ax8, ax9), (ax10, ax11), (ax12, ax13)) = plt.subplots(3, 2, figsize=(9, 12))
fig2.suptitle('Kepler elements variation of Europa')

semi_major_axis_eur = dep_var_array.loc[:,"a_Eur"]
eccentricity_eur = dep_var_array.loc[:,"e_Eur"]

#Theoretical behavior tides
c = 3.0089e-19
D = 3542.6

#dadt = 2/3*c*(1-7*D*(eccentricity_eur[0])**2)*semi_major_axis_eur[0] # Planet + Satellite
#dadt = (2/3)*c*semi_major_axis_eur[0] # Planet
#dedt = (57*0.379*4.799844e22*2*np.pi/310000.0)*eccentricity_eur[0]*((69911000/semi_major_axis_eur[0])**5)/(35600*8*1.8982e27) # Planet
#dadt = -14/3*c*D*semi_major_axis_eur[0]*(eccentricity_eur[0])**2 # Satellite
#dedt=-7/3*c*D*eccentricity_eur[0] # Satellite (+ Planet)

#SEMI MAJOR AXIS
#yacc = semi_major_axis_eur[0] + dadt*time
ax8.plot(time_day, semi_major_axis_eur, 'r', label = "Simulation")
#ax8.plot(time_day, yacc, 'g--', label = "Theoretical")
ax8.legend(loc="upper left")
ax8.set_ylabel('Semi-major axis [m]')
#ax2.set_ylim([4.22020296*1e8, 4.22020299*1e8])

#ECCENTRICITY
#yecc= eccentricity_eur[0] + dedt*time
ax9.plot(time_day, eccentricity_eur,'r',label="Simulation")
#ax9.plot(time_day, yecc,'g--', label="Theoretical")
ax9.set_ylabel('Eccentricity [-]')
ax9.legend(loc="upper right")
#ax3.set_ylim([0.0036393, 0.003640])

#INCLINATION
inclination_eur = np.rad2deg(dep_var_array.loc[:,"i_Eur"])
ax10.plot(time_day, inclination_eur)
ax10.set_ylabel('inclination [deg]')
#ax4.set_ylim(2.201, 2.203)

#RAAN
raan_eur = np.rad2deg(dep_var_array.loc[:,"RAAN_Eur"])
ax11.plot(time_day, raan_eur)
ax11.set_ylabel('RAAN [deg]')

#Argument of Pericenter
argument_of_pericenter_eur = np.rad2deg(dep_var_array.loc[:,"Argument_periapsis_Eur"])
ax12.plot(time_day, argument_of_pericenter_eur)
ax12.set_ylabel("Argument of pericenter [deg]")

#True Anomaly
true_anomaly_eur = np.rad2deg(dep_var_array.loc[:,"true_anomaly_Eur"])
ax13.plot(time_day, true_anomaly_eur)
ax13.set_ylabel("True anomaly [deg]")

for ax in fig2.get_axes():
    ax.set_xlabel('Time [Years]')
    ax.set_xlim([min(time_day), max(time_day)])
    ax.grid()
    ax.relim()
    ax.autoscale_view()
plt.tight_layout()
#plt.show()

#Mean motion over time
n2=np.sqrt(jupiter_gravitational_parameter/(semi_major_axis_Io[len(semi_major_axis_Io)-1]**3))
ndot_Io=(n2-n0_io)/(time[len(time)-1]-time[0])
n_Io=list()
n_Eur=list()
resonance=list()
j=0.0
while j<len(time):
    ni=np.sqrt(jupiter_gravitational_parameter/(semi_major_axis_Io[j]**3))
    ne=np.sqrt(jupiter_gravitational_parameter/(semi_major_axis_eur[j]**3))
    n_Io.append(ni)
    n_Eur.append(ne)
    resonance.append(ni/ne)
    j=j+1

plt.figure(figsize=(10,6))
plt.title("Mean motion variation over time")
plt.plot(time_day, n_Io, 'r', label="Io")
plt.plot(time_day, n_Eur, 'b', label="Europa")
plt.ylabel("Mean motion [rad/s]")
plt.xlabel("Time [years]")
plt.legend()
plt.grid()
plt.xlim([min(time_day), max(time_day)])
plt.ylim([0, 5e-5])
#plt.show()

plt.figure(figsize=(10,6))
plt.title("Resonance between mean motion of Io and Europa (n_Io/n_Europa) ")
plt.plot(time_day, resonance, 'r')
plt.xlabel("Time [years]")
plt.grid()
plt.xlim([min(time_day), max(time_day)])
plt.show()
