import time
import random
import numpy as np
import pygame
import math
from simulation.connection import carla
from simulation.sensors import CameraSensor, TrackEgoVehicleSensor, CollisionSensor
from simulation.settings import *

MAX_CHANGE_STEER_THRESHOLD = 0.4
MAX_JERK_THRESHOLD = 4
THRESHOLD_DISTANCE = 10
TTC_THRESHOLD = 4

class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        self.debug = world.debug
        self.prev_steering_angle = 0.0
        self.previous_acceleration = None
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()


    # A reset function for reseting our environment.
    def reset(self):
        try:
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()

            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town06":
                transform = self.map.get_spawn_points()[257]
                self.total_distance = 100
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)

            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = TrackEgoVehicleSensor(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)
            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 80 # previously 65 km/h
            self.max_speed = 85.0
            self.min_speed = 75.0
            self.max_distance_from_center = 3
            self.steer = float(0.0)
            self.throttle = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0

            if self.fresh_start:
                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    elif self.town == "Town06":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array([self.throttle, self.velocity, self.steer, self.distance_from_center, self.angle])

            time.sleep(0.5)

            self.collision_history.clear()
            self.episode_start_time = time.time()
            self.set_other_vehicles()

            return [self.image_obs, self.navigation_obs]

        except:
            print('Exception at reset method. Exiting ...')
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx, run_step):
        try:
            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Apply vehicle steering and acceleration
            steer = round(float(action_idx[0]), 1)
            steer = max(min(steer, 1.0), -1.0)

            throttle = round(float(action_idx[1]), 1)
            throttle = max(min(throttle, 1.0), -1.0)

            if throttle > 0:
                if throttle > 0.5:
                    self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=1))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=0.5))
            else:
                self.vehicle.apply_control(carla.VehicleControl(steer=steer, brake=abs(throttle)))

            self.throttle = throttle

            # Collect vehicle data
            self.collision_history = self.collision_obj.collision_data            
            self.rotation          = self.vehicle.get_transform().rotation.yaw
            self.location          = self.vehicle.get_location()

            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            # Calculate deviation from center of the lane
            self.current_waypoint_index = waypoint_index
            self.current_waypoint       = self.route_waypoints[ self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint          = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center   = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation  += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd         = self.vector(self.vehicle.get_velocity())
            wp_fwd      = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

            # Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            # Change in acceleration (required for jerk calculation)
            delta_acceleration = None
            if run_step:
                acceleration = self.get_current_acceleration()
                if self.previous_acceleration is not None:
                    delta_acceleration = acceleration - self.previous_acceleration
                self.previous_acceleration = acceleration

            # Get rewards
            done, reward, steering_penalty, jerk_penalty = self.reward(steer, delta_acceleration)

            if self.timesteps >= 3500:
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0

            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            # Observation parameters
            self.image_obs                  = self.camera_obj.front_camera.pop(-1)
            normalized_velocity             = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle                = abs(self.angle / np.deg2rad(20))
            vehicle_connectivity            = self.vehicle_connectivity()
            self.navigation_obs             = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])

            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()

            return [self.image_obs, self.navigation_obs, vehicle_connectivity], reward, done, [self.distance_covered, self.center_lane_deviation, steering_penalty, jerk_penalty]

        except:
            print('Exception at step method. Exiting ....')
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


# -------------------------------------------------
# Vehicle Connectivity |
# ------------------------------------------------

    def vehicle_connectivity(self):
        vehicle_connectivity = []
        for vehicle in self.actor_list:
            if len(self.actor_list) > 1 and vehicle != self.vehicle:
                vehicle_location = vehicle.get_location()
                velocity = math.sqrt(vehicle.get_velocity().x **2  + vehicle.get_velocity().y ** 2 + vehicle.get_velocity().z ** 2)
                vehicle_data = [vehicle_location.x, vehicle_location.y, vehicle_location.z, velocity]
                vehicle_connectivity.append(vehicle_data)

        return np.array(vehicle_connectivity)


# -------------------------------------------------
# Reward Function |
# -------------------------------------------------

    def reward(self, steer, delta_acceleration):
        done   = False
        reward = 0

        if len(self.collision_history) != 0:
            done   = True
            reward = -150
        elif self.distance_from_center > self.max_distance_from_center:
            done   = True
            reward = -10
        elif self.episode_start_time + 6 < time.time() and self.velocity < 1.0:
            reward = -150
            done   = True

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
        # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
        angle_factor     = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

        if not done:
            if self.continous_action_space:
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor
                elif self.velocity > self.target_speed:
                    reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor
                else:
                    reward = 1.0 * centering_factor * angle_factor 
            else:
                reward = 1.0 * centering_factor * angle_factor

        # Penalty for changes in strong steering wheel movements
        steering_angle_change       = abs(steer - self.prev_steering_angle)
        steering_smoothness_penalty = max(0, steering_angle_change - MAX_CHANGE_STEER_THRESHOLD)
        self.prev_steering_angle    = steer

        if 0.5 < steering_smoothness_penalty < 0.8:
            reward -= 0.15
        elif steering_smoothness_penalty > 0.8:
            reward -= 0.25

        # Penalty for strong jerk values (changes in acceleration)
        jerk_penalty = 0
        if delta_acceleration is not None and delta_acceleration > MAX_JERK_THRESHOLD: # jerk = delta acceleration / delta time, delta time = 1
            jerk_penalty = delta_acceleration
            reward -= jerk_penalty

        # Near miss penalty
        for vehicle in self.actor_list:
            if len(self.actor_list) > 1 and vehicle != self.vehicle:
                distance = self.distance_to_ego(vehicle.get_location())
                if distance < THRESHOLD_DISTANCE:
                    ttc = self.calculate_collision_time(vehicle)
                    print('ttc: {}'.format(ttc))
                    if ttc and ttc < TTC_THRESHOLD:
                        reward -= 2

        return done, reward, steering_smoothness_penalty, jerk_penalty


# ---------------------------------------------------
# Creating and Spawning other vehciles in the world|
# ---------------------------------------------------

    def set_other_vehicles(self):
        try:
            # spawn_points_town06 = [279, 12, 275, 280, 13, 276]
            spawn_points_town06 = [12, 275]
            for _ in range(0, len(spawn_points_town06)):
                spawn_point = self.map.get_spawn_points()[spawn_points_town06[_]]
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.apply_control(carla.VehicleControl(throttle=0.5))
                    self.actor_list.append(other_vehicle)
        except:
            print('Exception has occured in spawning vehicles')
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    def calculate_collision_time(self, secondary_vehicle):
        # Get the positions and velocities of the two vehicles
        xa0, ya0 = self.vehicle.get_location().x, self.vehicle.get_location().y
        xat, yat = self.vehicle.get_velocity().x, self.vehicle.get_velocity().y
        xb0, yb0 = secondary_vehicle.get_location().x, secondary_vehicle.get_location().y
        xbt, ybt = secondary_vehicle.get_velocity().x, secondary_vehicle.get_velocity().y

        # Calculate relative position and relative velocity
        relative_x = xb0 - xa0
        relative_y = yb0 - ya0
        relative_vx = xbt - xat
        relative_vy = ybt - yat

        # Calculate TTC using relative distance and relative velocity
        try:
            ttc = -(relative_x * relative_vx + relative_y * relative_vy) / (relative_vx**2 + relative_vy**2)
        except ZeroDivisionError:
            ttc = None

        if ttc is not None and 0 < ttc < 15:
            return ttc
        else:
            return None


    def get_current_acceleration(self):
        acceleration = self.vehicle.get_acceleration()
        return math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)


    def distance_to_ego(self, secondary_vehicle):
        ego_position = self.vehicle.get_location()
        return math.sqrt((ego_position.x - secondary_vehicle.x) ** 2 + (ego_position.y - secondary_vehicle.y) ** 2 + (ego_position.z - secondary_vehicle.z) ** 2)


    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    def get_world(self) -> object:
        return self.world


    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None