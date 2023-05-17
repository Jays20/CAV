import math


class Vehicle:
    def __init__(self, world, spawn_point,  color, vehicle_type='vehicle.tesla.model3',):
        self.world = world
        self.spawn_point = spawn_point
        self.color = color
        self.vehicle_type = vehicle_type
        self.vehicle_bp = None
        self.vehicle_actor = None


    def spawn(self):
        blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = blueprint_library.find(self.vehicle_type)
        self.vehicle_bp.set_attribute('color', self.color)
        # self.vehicle_bp.set_attribute('role_name', 'autopilot')

        self.vehicle_actor = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_point)
        if self.vehicle_actor is None:
            raise ValueError('Failed to spawn NPC vehicle.')


    def enable_autopilot(self):
        self.vehicle_actor.set_autopilot(True)
        
        
    def disable_autopilot(self):
        self.vehicle_actor.set_autopilot(False)
        
        
    def get_velocity(self) -> int:
        velocity_vector = self.vehicle_actor.get_velocity()
        return round(math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2) * 3.6)
        
        
    def destroy(self):
        if self.vehicle_actor is not None:
            self.vehicle_actor.destroy()