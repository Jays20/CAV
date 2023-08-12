import math
import numpy as np
import weakref
import pygame
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA


# ---------------------------------------------------------------------|
# ------------------------------- CAMERA |
# ---------------------------------------------------------------------|

class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)


# ---------------------------------------------------------------------|
# ------------------------------- Birds Eye View Sensor of Vehicle |
# ---------------------------------------------------------------------|

class TrackEgoVehicleSensor:

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode((720, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: TrackEgoVehicleSensor._get_third_person_camera(weak_self, image))

    def _set_camera_sensor(self, world):
        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))

        font = pygame.font.Font(None, 36)

        if round(self.parent.get_location().z) == 0:
            speed_text = font.render(f'Speed: {self.get_speed_km():.0f} km/h', True, (255, 255, 255))

            if hasattr(self, 'previous_location'):
                distance_covered = self.get_distance_between_locations()
                self.distance_traveled += distance_covered
            else:
                self.distance_traveled = 0.0
            self.previous_location = self.parent.get_location()
            distance_text = font.render(f'Distance: {self.distance_traveled:.0f} meters', True, (255, 255, 255))
        else:
            speed_text = font.render(f'Speed: {0:.0f} km/h', True, (255, 255, 255))
            distance_text = font.render(f'Distance: {0:.0f} meters', True, (255, 255, 255))

        self.display.blit(speed_text, (10, 10))
        self.display.blit(distance_text, (10, 50))

        pygame.display.flip()

    def get_speed_km(self):
        return math.sqrt(self.parent.get_velocity().x **2  + self.parent.get_velocity().y ** 2 + self.parent.get_velocity().z ** 2) * 3.6

    def get_distance_between_locations(self):
        current_location = self.parent.get_location()
        previous_location = self.previous_location

        return math.sqrt((previous_location.x - current_location.x) ** 2 + (previous_location.y - current_location.y) ** 2 + (previous_location.z - current_location.z) ** 2)


# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)
