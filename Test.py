from simulation.connection import carla
from vehicle import Vehicle


def main():
    HOST = "localhost"
    PORT = 2000
    TIMEOUT = 20.0

    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)
    world = client.load_world("Town06")
    map = world.get_map()
    
    spawn_points = map.get_spawn_points()
    for index, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(index), color=carla.Color(255, 255, 0), life_time=500)
        
    waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)
    draw_waypoints(waypoints, world, road_id=10, life_time=20)
    
    # Add traffic manager
    traffic_manager = client.get_trafficmanager()
    # traffic_manager_port = traffic_manager.get_port()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(0.05)
    
    # Create and spawn Ego vehicle
    ego = Vehicle(world, spawn_points[257], '170, 74, 68')
    ego.spawn()
    ego.enable_autopilot()

    while True:
        world.tick()

def draw_waypoints(waypoints, world, road_id=None, life_time=50.0):
    for waypoint in waypoints:
        if(waypoint.road_id == road_id):
            world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=life_time, persistent_lines=True)


    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass