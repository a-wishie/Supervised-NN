import os
import math
import pandas as pd
import numpy as np

def adjust_positions(reference_positions, all_drones_positions, min_distance=7):
    """
    Adjusts the positions of all drones to ensure minimum distance between them at similar timestamps.
    """
    adjusted_drones = []
    
    for drone_df in all_drones_positions:
        adjusted_drone = drone_df.copy()
        
        for idx, row in adjusted_drone.iterrows():
            current_time = row['time']
            other_positions = []
            
            # Add reference drone position at this timestamp
            ref_matches = reference_positions[np.isclose(reference_positions['time'], current_time, atol=0.1)]
            if not ref_matches.empty:
                ref_row = ref_matches.iloc[0]
                other_positions.append((ref_row['position_x'], ref_row['position_y']))
            
            # Add other drones' positions at this timestamp
            for other_drone in adjusted_drones:
                other_matches = other_drone[np.isclose(other_drone['time'], current_time, atol=0.1)]
                if not other_matches.empty:
                    other_row = other_matches.iloc[0]
                    other_positions.append((other_row['position_x'], other_row['position_y']))
            
            # Current drone position
            x, y = row['position_x'], row['position_y']
            original_x, original_y = x, y
            
            # Check and adjust position if too close to other drones
            attempts = 0
            while attempts < 100:
                too_close = False
                
                for other_x, other_y in other_positions:
                    distance = math.sqrt((x - other_x)**2 + (y - other_y)**2)
                    if distance < min_distance:
                        too_close = True
                        break
                
                if too_close:
                    angle = 2 * math.pi * (attempts / 8)
                    radius = (attempts // 8 + 1) * min_distance
                    x = original_x + radius * math.cos(angle)
                    y = original_y + radius * math.sin(angle)
                    attempts += 1
                else:
                    break
            
            adjusted_drone.at[idx, 'position_x'] = x
            adjusted_drone.at[idx, 'position_y'] = y
        
        adjusted_drones.append(adjusted_drone)
    
    return adjusted_drones

def read_drones_from_directory(directory):
    """
    Reads drone position data from a directory of CSV files.
    """
    drone_positions = []
    file_names = []
    
    if not os.path.exists(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file in sorted(csv_files):
        file_path = os.path.join(directory, file)
        try:
            data = pd.read_csv(file_path)
            required_columns = ['time', 'position_x', 'position_y']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Warning: File {file} missing columns: {missing_columns}")
                print(f"Available columns: {data.columns.tolist()}")
                continue
                
            file_names.append(file)
            drone_positions.append(data)
            print(f"Successfully loaded {file}")
            
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
            continue
    
    return drone_positions, file_names

def main():
    # Use normalized path with os.path.join to avoid escape sequence issues
    base_dir = os.path.join("data files train")
    reference_file = os.path.join(base_dir, "drone_1_data.csv")
    drone_directory = os.path.join(base_dir, "drones")
    output_directory = "adjusted_positions"
    
    try:
        print("Starting drone position adjustment process...")
        
        os.makedirs(output_directory, exist_ok=True)
        print(f"Output directory created/verified: {output_directory}")
        
        # Load reference positions
        print(f"Reading reference file: {reference_file}")
        if not os.path.exists(reference_file):
            raise ValueError(f"Reference file not found: {reference_file}")
            
        reference_data = pd.read_csv(reference_file)
        print("Available columns in reference file:", reference_data.columns.tolist())
        
        # Check for required columns with position_x and position_y
        required_columns = ['time', 'position_x', 'position_y']
        missing_columns = [col for col in required_columns if col not in reference_data.columns]
        
        if missing_columns:
            raise ValueError(f"Reference file missing required columns: {missing_columns}. Available columns: {reference_data.columns.tolist()}")
        
        print("\nReading drone files from:", drone_directory)
        all_drones_positions, drone_files = read_drones_from_directory(drone_directory)
        
        if not all_drones_positions:
            raise ValueError("No valid drone position files found")
        
        print(f"\nFound {len(all_drones_positions)} valid drone files")
        
        print("\nAdjusting drone positions...")
        adjusted_positions = adjust_positions(reference_data, all_drones_positions)
        
        print("\nSaving adjusted positions...")
        for i, drone_file in enumerate(drone_files):
            output_path = os.path.join(output_directory, f"adjusted_{drone_file}")
            adjusted_positions[i].to_csv(output_path, index=False)
            print(f"Saved: adjusted_{drone_file}")
        
        print(f"\nAdjustment complete. All files saved in '{output_directory}'.")
        
    except Exception as e:
        print("\nAn error occurred:")
        print(str(e))
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()