import carla
import random
import time

def main():
    try:
        # 1. Connect to the server
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        
        # 2. Get the virtual world
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 3. Choose a vehicle blueprint (e.g., a Tesla Model 3)
        bp = blueprint_library.filter('model3')[0]
        
        # 4. Find a valid random spawn point on the map
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # 5. Spawn the vehicle actor
        vehicle = world.spawn_actor(bp, spawn_point)
        print(f"Spawned vehicle: {vehicle.type_id} at {spawn_point.location}")
        
        # 6. Put it on Autopilot so it drives itself!
        vehicle.set_autopilot(True)
        
        # Let it drive around for 15 seconds
        time.sleep(15)

    finally:
        # ALWAYS clean up your actors when done, otherwise the server keeps them alive!
        if 'vehicle' in locals():
            print("Destroying vehicle...")
            vehicle.destroy()

if __name__ == '__main__':
    main()