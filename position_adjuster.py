'''# position_adjuster.py
import os
import math
import pandas as pd
import numpy as np

def create_v_formation(reference_data, num_drones, v_spacing=7, v_angle=30):
    """
    Creates a V-formation pattern relative to the reference drone's position.
    """
    adjusted_drones = []
    
    # Convert angle to radians
    angle_rad = math.radians(v_angle)
    
    for drone_idx in range(num_drones):
        adjusted_drone = reference_data.copy()
        
        # Calculate position in formation
        side = 1 if drone_idx % 2 == 0 else -1  # Alternate sides
        position = (drone_idx + 1) // 2  # Position from center
        
        # Calculate offsets based on V-formation geometry
        x_offset = position * v_spacing * math.cos(angle_rad) * side
        y_offset = -position * v_spacing * math.sin(angle_rad)  # Negative to go backwards
        
        # Apply offsets to all positions
        adjusted_drone['position_x'] = adjusted_drone['position_x'] + x_offset
        adjusted_drone['position_y'] = adjusted_drone['position_y'] + y_offset
        
        # Keep same altitude but add slight variation
        altitude_variation = np.sin(np.linspace(0, 4*np.pi, len(adjusted_drone))) * 0.5
        adjusted_drone['position_z'] = adjusted_drone['position_z'] + altitude_variation
        
        adjusted_drones.append(adjusted_drone)
    
    return adjusted_drones

def calculate_metrics(df):
    """
    Calculate proper velocity and acceleration metrics.
    """
    # Time differences
    dt = df['time'].diff()
    
    # Calculate velocities
    df['velocity_x'] = df['position_x'].diff() / dt
    df['velocity_y'] = df['position_y'].diff() / dt
    df['velocity_z'] = df['position_z'].diff() / dt
    
    # Calculate total velocity
    df['velocity_total'] = np.sqrt(
        df['velocity_x']**2 + 
        df['velocity_y']**2 + 
        df['velocity_z']**2
    )
    
    # Calculate accelerations
    df['acceleration_x'] = df['velocity_x'].diff() / dt
    df['acceleration_y'] = df['velocity_y'].diff() / dt
    df['acceleration_z'] = df['velocity_z'].diff() / dt
    
    # Calculate total acceleration
    df['acceleration_total'] = np.sqrt(
        df['acceleration_x']**2 + 
        df['acceleration_y']**2 + 
        df['acceleration_z']**2
    )
    
    # Apply smoothing
    window = 5
    for col in ['velocity_total', 'acceleration_total']:
        df[col] = df[col].rolling(window=window, center=True).mean()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    # Use normalized path with os.path.join
    base_dir = os.path.join("data files train")
    reference_file = os.path.join(base_dir, "drone_1_data.csv")
    output_directory = "adjusted_positions"
    
    try:
        print("Starting drone formation process...")
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Load reference positions
        if not os.path.exists(reference_file):
            raise ValueError(f"Reference file not found: {reference_file}")
            
        reference_data = pd.read_csv(reference_file)
        
        # Number of drones in formation (excluding reference drone)
        num_drones = 8  # This will create a 9-drone V-formation like in the image
        
        print("\nCreating V-formation pattern...")
        adjusted_positions = create_v_formation(reference_data, num_drones)
        
        # Calculate metrics for each drone
        print("\nCalculating flight metrics...")
        for i, drone_df in enumerate(adjusted_positions):
            adjusted_positions[i] = calculate_metrics(drone_df)
            
            # Save adjusted positions
            output_path = os.path.join(output_directory, f"adjusted_drone_{i+1}.csv")
            adjusted_positions[i].to_csv(output_path, index=False)
            print(f"Saved: adjusted_drone_{i+1}.csv")
        
        print(f"\nFormation complete. All files saved in '{output_directory}'.")
        
    except Exception as e:
        print("\nAn error occurred:")
        print(str(e))
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()'''

'''
import os
import math
import pandas as pd
import numpy as np

def calculate_ground_station_metrics(positions, ground_station_pos=(0, 0, 0)):
    """Calculate distance and communication status with ground station"""
    x, y, z = positions
    dx = x - ground_station_pos[0]
    dy = y - ground_station_pos[1]
    dz = z - ground_station_pos[2]
    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    comm_range = 100  # meters
    return distance, (distance <= comm_range).astype(int)

def create_v_formation(reference_data, num_drones, v_spacing=7, v_angle=30):
    """
    Creates a V-formation pattern relative to the reference drone's position.
    Modified to handle larger number of drones in double-V formation.
    """
    adjusted_drones = []
    
    # Convert angle to radians
    angle_rad = math.radians(v_angle)
    
    # For 16 drones, create a double-V formation (8 drones per V)
    drones_per_v = 8
    
    for drone_idx in range(num_drones):
        adjusted_drone = reference_data.copy()
        
        # Determine which V formation this drone belongs to
        v_formation = drone_idx // drones_per_v  # 0 for first V, 1 for second V
        position_in_v = drone_idx % drones_per_v
        
        # Calculate position in formation
        side = 1 if position_in_v % 2 == 0 else -1  # Alternate sides
        position = (position_in_v + 1) // 2  # Position from center
        
        # Calculate offsets based on V-formation geometry
        x_offset = position * v_spacing * math.cos(angle_rad) * side
        y_offset = -position * v_spacing * math.sin(angle_rad)  # Negative to go backwards
        
        # Add offset for second V-formation
        if v_formation == 1:
            y_offset -= 15  # Space between V formations
        
        # Apply offsets to all positions
        adjusted_drone['position_x'] = adjusted_drone['position_x'] + x_offset
        adjusted_drone['position_y'] = adjusted_drone['position_y'] + y_offset
        
        # Keep same altitude but add slight variation
        altitude_variation = np.sin(np.linspace(0, 4*np.pi, len(adjusted_drone))) * 0.5
        adjusted_drone['position_z'] = adjusted_drone['position_z'] + altitude_variation
        
        # Calculate ground station metrics
        distances, comm_status = calculate_ground_station_metrics(
            (adjusted_drone['position_x'], 
             adjusted_drone['position_y'], 
             adjusted_drone['position_z'])
        )
        adjusted_drone['ground_station_distance'] = distances
        adjusted_drone['comm_range_status'] = comm_status
        
        # Add wind parameters
        adjusted_drone['wind_speed'] = 5 + 2 * np.sin(adjusted_drone['time']/10) + adjusted_drone['position_z']/100
        adjusted_drone['wind_angle'] = (180 + 45 * np.sin(adjusted_drone['time']/15)) % 360
        
        # Add battery parameters
        adjusted_drone['battery_voltage'] = 11.1 - 0.1 * (adjusted_drone['position_z']/400)
        base_current = 10.0
        adjusted_drone['battery_current'] = base_current * (1 + adjusted_drone['position_z']/300)
        
        adjusted_drones.append(adjusted_drone)
    
    return adjusted_drones

def calculate_metrics(df):
    """
    Calculate proper velocity and acceleration metrics.
    """
    # Time differences
    dt = df['time'].diff()
    
    # Calculate velocities
    df['velocity_x'] = df['position_x'].diff() / dt
    df['velocity_y'] = df['position_y'].diff() / dt
    df['velocity_z'] = df['position_z'].diff() / dt
    
    # Calculate total velocity
    df['velocity_total'] = np.sqrt(
        df['velocity_x']**2 + 
        df['velocity_y']**2 + 
        df['velocity_z']**2
    )
    
    # Calculate accelerations
    df['acceleration_x'] = df['velocity_x'].diff() / dt
    df['acceleration_y'] = df['velocity_y'].diff() / dt
    df['acceleration_z'] = df['velocity_z'].diff() / dt
    
    # Calculate total acceleration
    df['acceleration_total'] = np.sqrt(
        df['acceleration_x']**2 + 
        df['acceleration_y']**2 + 
        df['acceleration_z']**2
    )
    
    # Calculate angular and orientation parameters
    t = df['time'].values
    df['angular_x'] = np.sin(t/10) * 0.1
    df['angular_y'] = np.sin(t/12) * 0.1
    df['angular_z'] = np.sin(t/15) * 0.1
    
    # Calculate orientation quaternion
    pitch = np.arctan2(df['velocity_z'], df['velocity_total'])
    roll = np.arctan2(-df['velocity_x'], np.sqrt(df['velocity_y']**2 + df['velocity_z']**2))
    yaw = np.arctan2(df['velocity_y'], df['velocity_x'])
    
    # Convert to quaternion (simplified)
    df['orientation_x'] = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    df['orientation_y'] = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    df['orientation_z'] = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    df['orientation_w'] = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    # Calculate linear accelerations
    df['linear_acceleration_x'] = df['acceleration_x']
    df['linear_acceleration_y'] = df['acceleration_y']
    df['linear_acceleration_z'] = df['acceleration_z']
    
    # Apply smoothing
    window = 5
    smooth_columns = ['velocity_total', 'acceleration_total', 'angular_x', 'angular_y', 'angular_z']
    for col in smooth_columns:
        df[col] = df[col].rolling(window=window, center=True).mean()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    base_dir = os.path.join("data files train")
    reference_file = os.path.join(base_dir, "drone_1_data.csv")
    output_directory = "adjusted_positions"
    
    try:
        print("Starting drone formation process...")
        
        os.makedirs(output_directory, exist_ok=True)
        
        if not os.path.exists(reference_file):
            raise ValueError(f"Reference file not found: {reference_file}")
            
        reference_data = pd.read_csv(reference_file)
        
        # Updated to handle 16 drones
        num_drones = 16
        
        print("\nCreating double V-formation pattern...")
        adjusted_positions = create_v_formation(reference_data, num_drones)
        
        print("\nCalculating flight metrics...")
        for i, drone_df in enumerate(adjusted_positions):
            adjusted_positions[i] = calculate_metrics(drone_df)
            
            output_path = os.path.join(output_directory, f"adjusted_drone_{i+1}.csv")
            adjusted_positions[i].to_csv(output_path, index=False)
            print(f"Saved: adjusted_drone_{i+1}.csv")
        
        print(f"\nFormation complete. All files saved in '{output_directory}'.")
        
    except Exception as e:
        print("\nAn error occurred:")
        print(str(e))
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()'''

import os
import math
import pandas as pd
import numpy as np

def calculate_ground_station_metrics(positions, ground_station_pos=(0, 0, 0)):
    """Calculate distance and communication status with ground station"""
    x, y, z = positions
    dx = x - ground_station_pos[0]
    dy = y - ground_station_pos[1]
    dz = z - ground_station_pos[2]
    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    comm_range = 100  # meters
    return distance, (distance <= comm_range).astype(int)

def create_v_formation(reference_data, num_drones, v_spacing=7, v_angle=30):
    """
    Creates a V-formation pattern while preserving original movement patterns.
    """
    adjusted_drones = []
    
    # Convert angle to radians
    angle_rad = math.radians(v_angle)
    
    # For 16 drones, create a double-V formation (8 drones per V)
    drones_per_v = 8
    
    # Extract original movement pattern
    original_movement = {
        'dx': reference_data['position_x'].diff().fillna(0),
        'dy': reference_data['position_y'].diff().fillna(0),
        'dz': reference_data['position_z'].diff().fillna(0)
    }
    
    for drone_idx in range(num_drones):
        adjusted_drone = pd.DataFrame()
        adjusted_drone['time'] = reference_data['time'].copy()
        
        # Determine which V formation this drone belongs to
        v_formation = drone_idx // drones_per_v  # 0 for first V, 1 for second V
        position_in_v = drone_idx % drones_per_v
        
        # Calculate position in formation
        side = 1 if position_in_v % 2 == 0 else -1  # Alternate sides
        position = (position_in_v + 1) // 2  # Position from center
        
        # Calculate base offsets for formation
        x_offset = position * v_spacing * math.cos(angle_rad) * side
        y_offset = -position * v_spacing * math.sin(angle_rad)
        
        # Add offset for second V-formation
        if v_formation == 1:
            y_offset -= 15  # Space between V formations
        
        # Initialize starting positions with offsets
        start_x = reference_data['position_x'].iloc[0] + x_offset
        start_y = reference_data['position_y'].iloc[0] + y_offset
        start_z = reference_data['position_z'].iloc[0]
        
        # Calculate new positions while maintaining relative movement
        adjusted_drone['position_x'] = start_x + np.cumsum(original_movement['dx'])
        adjusted_drone['position_y'] = start_y + np.cumsum(original_movement['dy'])
        adjusted_drone['position_z'] = start_z + np.cumsum(original_movement['dz'])
        
        # Add formation-specific variations
        time_factor = 2 * np.pi * np.linspace(0, 1, len(adjusted_drone))
        
        # Add subtle periodic variations unique to each drone
        phase_x = drone_idx * (2 * np.pi / num_drones)
        phase_y = drone_idx * (2 * np.pi / num_drones) + np.pi/4
        
        # Add controlled variation to maintain formation while allowing individual movement
        adjusted_drone['position_x'] += np.sin(time_factor + phase_x) * 0.5
        adjusted_drone['position_y'] += np.sin(time_factor + phase_y) * 0.5
        adjusted_drone['position_z'] += np.sin(time_factor * 2) * 0.3
        
        # Calculate ground station metrics
        distances, comm_status = calculate_ground_station_metrics(
            (adjusted_drone['position_x'], 
             adjusted_drone['position_y'], 
             adjusted_drone['position_z'])
        )
        adjusted_drone['ground_station_distance'] = distances
        adjusted_drone['comm_range_status'] = comm_status
        
        # Add wind effect
        adjusted_drone['wind_speed'] = 5 + 2 * np.sin(adjusted_drone['time']/10) + adjusted_drone['position_z']/100
        adjusted_drone['wind_angle'] = (180 + 45 * np.sin(adjusted_drone['time']/15)) % 360
        
        # Add battery parameters
        adjusted_drone['battery_voltage'] = 11.1 - 0.1 * (adjusted_drone['position_z']/400)
        base_current = 10.0
        adjusted_drone['battery_current'] = base_current * (1 + adjusted_drone['position_z']/300)
        
        adjusted_drones.append(adjusted_drone)
    
    return adjusted_drones

def calculate_metrics(df):
    """Calculate proper velocity and acceleration metrics."""
    # Time differences
    dt = df['time'].diff().fillna(df['time'].diff().mean())
    
    # Calculate velocities
    df['velocity_x'] = df['position_x'].diff() / dt
    df['velocity_y'] = df['position_y'].diff() / dt
    df['velocity_z'] = df['position_z'].diff() / dt
    
    # Calculate total velocity
    df['velocity_total'] = np.sqrt(
        df['velocity_x']**2 + 
        df['velocity_y']**2 + 
        df['velocity_z']**2
    )
    
    # Calculate accelerations
    df['acceleration_x'] = df['velocity_x'].diff() / dt
    df['acceleration_y'] = df['velocity_y'].diff() / dt
    df['acceleration_z'] = df['velocity_z'].diff() / dt
    
    # Calculate total acceleration
    df['acceleration_total'] = np.sqrt(
        df['acceleration_x']**2 + 
        df['acceleration_y']**2 + 
        df['acceleration_z']**2
    )
    
    # Calculate angular and orientation parameters based on actual movement
    heading = np.arctan2(df['velocity_y'], df['velocity_x'])
    pitch = np.arctan2(df['velocity_z'], 
                      np.sqrt(df['velocity_x']**2 + df['velocity_y']**2))
    
    # Add realistic angular velocities based on movement
    df['angular_x'] = np.gradient(pitch, dt)
    df['angular_y'] = np.zeros_like(dt)  # Assume minimal roll
    df['angular_z'] = np.gradient(heading, dt)
    
    # Calculate orientation quaternion
    roll = np.zeros_like(heading)  # Assume minimal roll for stability
    
    # Convert to quaternion
    cy = np.cos(heading * 0.5)
    sy = np.sin(heading * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    df['orientation_w'] = cr * cp * cy + sr * sp * sy
    df['orientation_x'] = sr * cp * cy - cr * sp * sy
    df['orientation_y'] = cr * sp * cy + sr * cp * sy
    df['orientation_z'] = cr * cp * sy - sr * sp * cy
    
    # Calculate linear accelerations (in body frame)
    df['linear_acceleration_x'] = df['acceleration_x']
    df['linear_acceleration_y'] = df['acceleration_y']
    df['linear_acceleration_z'] = df['acceleration_z']
    
    # Apply smoothing to reduce noise
    window = 5
    smooth_columns = ['velocity_total', 'acceleration_total', 
                     'angular_x', 'angular_y', 'angular_z']
    
    for col in smooth_columns:
        df[col] = df[col].rolling(window=window, center=True).mean()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def main():
    base_dir = os.path.join("data files train")
    reference_file = os.path.join(base_dir, "drone_1_data.csv")
    output_directory = "adjusted_positions"
    
    try:
        print("Starting drone formation process...")
        
        os.makedirs(output_directory, exist_ok=True)
        
        if not os.path.exists(reference_file):
            raise ValueError(f"Reference file not found: {reference_file}")
            
        reference_data = pd.read_csv(reference_file)
        
        num_drones = 16
        
        print("\nCreating double V-formation pattern...")
        adjusted_positions = create_v_formation(reference_data, num_drones)
        
        print("\nCalculating flight metrics...")
        for i, drone_df in enumerate(adjusted_positions):
            adjusted_positions[i] = calculate_metrics(drone_df)
            
            output_path = os.path.join(output_directory, f"adjusted_drone_{i+1}.csv")
            adjusted_positions[i].to_csv(output_path, index=False)
            print(f"Saved: adjusted_drone_{i+1}.csv")
        
        print(f"\nFormation complete. All files saved in '{output_directory}'.")
        
    except Exception as e:
        print("\nAn error occurred:")
        print(str(e))
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()