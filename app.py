'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_and_combine_data():
    """Load and process the flight data"""
    df = pd.read_csv('flights/1.csv', low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Sample data for smoother animation
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def get_safe_limits(data):
    """Calculate limits handling NaN/Inf values"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0, 1  # Default limits if no valid data
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_animated_metrics(df):
    """Create animated metrics visualization with proper error handling"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drone Flight Metrics', fontsize=16)
    
    # Convert time to numpy array
    time_data = df['time'].to_numpy()
    
    # Define metrics and their units
    metrics = [
        ('altitude', 'Altitude (m)'),
        ('wind_speed', 'Wind Speed (m/s)'),
        ('battery_voltage', 'Battery Voltage (V)'),
        ('wind_angle', 'Wind Angle (degrees)')
    ]
    
    lines = []
    points = []
    
    # Store metric data for animation
    metric_data_dict = {}
    for metric, _ in metrics:
        if metric in df.columns:
            # Use newer pandas methods for filling missing values
            metric_data = df[metric].ffill().bfill().to_numpy()
            metric_data_dict[metric] = metric_data
        else:
            print(f"Warning: {metric} not found in data, using zeros")
            metric_data_dict[metric] = np.zeros_like(time_data)
    
    # Initialize subplots
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        metric_data = metric_data_dict[metric]
        
        # Create line and point
        line, = ax.plot([], [], 'b-', linewidth=2)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        # Set limits safely
        y_min, y_max = get_safe_limits(metric_data)
        ax.set_xlim(time_data[0], time_data[-1])
        ax.set_ylim(y_min, y_max)
        
        # Add labels and grid
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        
        lines.append(line)
        points.append(point)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(frame):
        # Calculate index based on frame
        idx = int((frame / 1000) * len(df))
        idx = min(idx, len(df) - 1)
        
        # Update each subplot
        for i, (metric, _) in enumerate(metrics):
            metric_data = metric_data_dict[metric]
            lines[i].set_data(time_data[:idx], metric_data[:idx])
            points[i].set_data([time_data[idx]], [metric_data[idx]])
        
        return lines + points
    
    # Create animation
    frames = 1000
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                 interval=20, blit=True)
    
    print("Saving metrics animation...")
    anim.save('drone_metrics_animated.gif', writer='pillow', fps=30)
    print("Metrics animation saved!")
    
    return fig

def main():
    print("Loading data...")
    try:
        df = load_and_combine_data()
        
        print("\nDataset information:")
        print(f"Total number of records: {len(df)}")
        print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
        
        # Print coordinate ranges for verification
        print(f"\nPosition ranges:")
        print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
        print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
        print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        if 'altitude' in df.columns:
            print(f"Altitude range: {df['altitude'].min():.2f}m to {df['altitude'].max():.2f}m")
        
        print("\nCreating animated visualizations...")
        
        # Create metrics visualization
        metrics_fig = create_animated_metrics(df)
        plt.close(metrics_fig)
        
        print("\nAnimation file created:")
        print("- drone_metrics_animated.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def load_and_combine_data():
    """Load and process the flight data"""
    df = pd.read_csv('flights/1.csv', low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Sample data for smoother animation if needed
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def get_safe_limits(data):
    """Calculate limits handling NaN/Inf values"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_metrics_visualization(df):
    """Create static visualizations of drone metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drone Flight Metrics', fontsize=16)
    
    metrics = [
        ('altitude', 'Altitude (m)'),
        ('wind_speed', 'Wind Speed (m/s)'),
        ('battery_voltage', 'Battery Voltage (V)'),
        ('wind_angle', 'Wind Angle (degrees)')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in df.columns:
            # Plot the data
            metric_data = df[metric].ffill().bfill()
            ax.plot(df['time'], metric_data, 'b-', linewidth=2, label='Actual')
            
            # Add trend line
            z = np.polyfit(df['time'], metric_data, 1)
            p = np.poly1d(z)
            ax.plot(df['time'], p(df['time']), "r--", alpha=0.8, label='Trend')
            
            # Set limits
            y_min, y_max = get_safe_limits(metric_data)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(df['time'].min(), df['time'].max())
        else:
            ax.text(0.5, 0.5, f'No {metric} data available', 
                   ha='center', va='center')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('drone_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_trajectory_animation(df):
    """Create animated visualization of drone trajectory"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get position data
    x_data = df['position_x'].values
    y_data = df['position_y'].values
    z_data = df['position_z'].values
    
    # Calculate limits for each axis
    x_min, x_max = get_safe_limits(x_data)
    y_min, y_max = get_safe_limits(y_data)
    z_min, z_max = get_safe_limits(z_data)
    
    # Initialize empty line and point
    line, = ax.plot([], [], [], 'b-', label='Flight Path', linewidth=2)
    point, = ax.plot([], [], [], 'ro', label='Current Position', markersize=10)
    
    # Set axis labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Drone Flight Trajectory')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Add legend
    ax.legend()
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    def update(frame):
        # Calculate index based on frame
        idx = int((frame / 1000) * len(df))
        idx = min(idx, len(df) - 1)
        
        # Update trajectory line
        line.set_data(x_data[:idx], y_data[:idx])
        line.set_3d_properties(z_data[:idx])
        
        # Update current position point
        point.set_data([x_data[idx]], [y_data[idx]])
        point.set_3d_properties([z_data[idx]])
        
        # Rotate view slightly for better 3D perception
        ax.view_init(elev=30, azim=frame/10)
        
        return line, point
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    # Save animation
    print("Saving trajectory animation...")
    anim.save('drone_trajectory.gif', writer='pillow', fps=30)
    plt.close()

def main():
    print("Loading data...")
    try:
        df = load_and_combine_data()
        
        print("\nDataset information:")
        print(f"Total number of records: {len(df)}")
        print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
        
        # Print coordinate ranges
        print(f"\nPosition ranges:")
        print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
        print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
        print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        if 'altitude' in df.columns:
            print(f"Altitude range: {df['altitude'].min():.2f}m to {df['altitude'].max():.2f}m")
        
        # Create metrics visualization
        print("\nCreating metric visualizations...")
        create_metrics_visualization(df)
        
        # Create trajectory animation
        print("\nCreating trajectory animation...")
        create_trajectory_animation(df)
        
        print("\nFiles created:")
        print("- drone_metrics.png")
        print("- drone_trajectory.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_combine_data():
    """Load and process the flight data"""
    df = pd.read_csv('flights.csv', low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    return df

def get_safe_limits(data):
    """Calculate limits handling NaN/Inf values"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return 0, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_metrics_visualization(df):
    """Create static visualizations of drone metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drone Flight Metrics Over Time', fontsize=16)
    
    metrics = [
        ('altitude', 'Altitude (m)'),
        ('wind_speed', 'Wind Speed (m/s)'),
        ('battery_voltage', 'Battery Voltage (V)'),
        ('wind_angle', 'Wind Angle (degrees)')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        if metric in df.columns:
            # Plot the data
            metric_data = df[metric].ffill().bfill()
            ax.plot(df['time'], metric_data, 'b-', linewidth=2, label='Actual')
            
            # Add trend line
            z = np.polyfit(df['time'], metric_data, 1)
            p = np.poly1d(z)
            ax.plot(df['time'], p(df['time']), "r--", alpha=0.8, label='Trend')
            
            # Add statistical annotations
            mean_val = metric_data.mean()
            ax.axhline(y=mean_val, color='g', linestyle=':', label='Mean')
            
            # Set limits
            y_min, y_max = get_safe_limits(metric_data)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(df['time'].min(), df['time'].max())
            
            # Add statistics text
            stats_text = f'Mean: {mean_val:.2f}\nMin: {metric_data.min():.2f}\nMax: {metric_data.max():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No {metric} data available', 
                   ha='center', va='center')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_trajectory_visualization(df):
    """Create 2D visualizations of drone trajectory"""
    fig = plt.figure(figsize=(15, 15))
    
    # Create subplots for different views
    gs = plt.GridSpec(2, 2, figure=fig)
    ax_top = fig.add_subplot(gs[0, :])    # Top view (X-Y)
    ax_side1 = fig.add_subplot(gs[1, 0])  # Side view 1 (X-Z)
    ax_side2 = fig.add_subplot(gs[1, 1])  # Side view 2 (Y-Z)
    
    fig.suptitle('Drone Flight Trajectory Views', fontsize=16)
    
    # Color map based on time for trajectory visualization
    norm = plt.Normalize(df['time'].min(), df['time'].max())
    
    # Top view (X-Y plane)
    scatter_top = ax_top.scatter(df['position_x'], df['position_y'], 
                               c=df['time'], cmap='viridis', norm=norm)
    ax_top.plot(df['position_x'], df['position_y'], 'gray', alpha=0.3)
    ax_top.set_title('Top View (X-Y Plane)')
    ax_top.set_xlabel('X Position (m)')
    ax_top.set_ylabel('Y Position (m)')
    ax_top.grid(True)
    
    # Mark start and end points
    ax_top.plot(df['position_x'].iloc[0], df['position_y'].iloc[0], 'go', 
                label='Start', markersize=10)
    ax_top.plot(df['position_x'].iloc[-1], df['position_y'].iloc[-1], 'ro', 
                label='End', markersize=10)
    
    # Side view 1 (X-Z plane)
    ax_side1.scatter(df['position_x'], df['position_z'], 
                    c=df['time'], cmap='viridis', norm=norm)
    ax_side1.plot(df['position_x'], df['position_z'], 'gray', alpha=0.3)
    ax_side1.set_title('Side View (X-Z Plane)')
    ax_side1.set_xlabel('X Position (m)')
    ax_side1.set_ylabel('Z Position (m)')
    ax_side1.grid(True)
    
    # Side view 2 (Y-Z plane)
    scatter_side2 = ax_side2.scatter(df['position_y'], df['position_z'], 
                                   c=df['time'], cmap='viridis', norm=norm)
    ax_side2.plot(df['position_y'], df['position_z'], 'gray', alpha=0.3)
    ax_side2.set_title('Side View (Y-Z Plane)')
    ax_side2.set_xlabel('Y Position (m)')
    ax_side2.set_ylabel('Z Position (m)')
    ax_side2.grid(True)
    
    # Add colorbar
    cbar = fig.colorbar(scatter_side2, ax=[ax_top, ax_side1, ax_side2])
    cbar.set_label('Time (s)')
    
    # Add legend to top view
    ax_top.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def main():
    print("Loading data...")
    try:
        df = load_and_combine_data()
        
        print("\nDataset information:")
        print(f"Total number of records: {len(df)}")
        print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
        
        # Print coordinate ranges
        print(f"\nPosition ranges:")
        print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
        print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
        print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        if 'altitude' in df.columns:
            print(f"Altitude range: {df['altitude'].min():.2f}m to {df['altitude'].max():.2f}m")
        
        # Create and save metrics visualization
        print("\nCreating metric visualizations...")
        metrics_fig = create_metrics_visualization(df)
        metrics_fig.savefig('drone_metrics.png', dpi=300, bbox_inches='tight')
        
        # Create and save trajectory visualization
        print("\nCreating trajectory visualizations...")
        trajectory_fig = create_trajectory_visualization(df)
        trajectory_fig.savefig('drone_trajectory.png', dpi=300, bbox_inches='tight')
        
        print("\nVisualization files created:")
        print("- drone_metrics.png")
        print("- drone_trajectory.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Sample data for smoother animation if needed
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def load_multiple_drone_data(file_paths):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    if not file_paths:
        raise ValueError("No drone data files provided!")
    
    for file_path in file_paths:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize all drones to the same time scale
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column):
    """Calculate limits handling NaN/Inf values for multiple drones"""
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return 0, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_trajectory_animation(drones_data, drone_names=None):
    """Create animated visualization of multiple drone trajectories"""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x')
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y')
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z')
    
    # Define colors for each drone
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    # Initialize empty lines and points for each drone
    lines = []
    points = []
    
    for i, _ in enumerate(drones_data):
        color = colors[i % len(colors)]
        drone_name = drone_names[i] if drone_names else f'Drone {i+1}'
        line, = ax.plot([], [], [], f'{color}-', label=f'{drone_name} Path', linewidth=2)
        point, = ax.plot([], [], [], f'{color}o', label=f'{drone_name} Position', markersize=10)
        lines.append(line)
        points.append(point)
    
    # Set axis labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Multiple Drone Flight Trajectories')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Add legend with smaller font and more columns
    ax.legend(prop={'size': 8}, ncol=2)
    
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return lines + points
    
    def update(frame):
        # Calculate progress based on frame
        progress = frame / 1000
        
        # Update each drone's position
        for i, df in enumerate(drones_data):
            # Get position data
            x_data = df['position_x'].values
            y_data = df['position_y'].values
            z_data = df['position_z'].values
            
            # Calculate current index based on progress
            idx = int(progress * len(df))
            idx = min(idx, len(df) - 1)
            
            # Update trajectory line
            lines[i].set_data(x_data[:idx], y_data[:idx])
            lines[i].set_3d_properties(z_data[:idx])
            
            # Update current position point
            points[i].set_data([x_data[idx]], [y_data[idx]])
            points[i].set_3d_properties([z_data[idx]])
        
        # Rotate view slightly for better 3D perception
        ax.view_init(elev=30, azim=frame/10)
        
        return lines + points
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    # Save animation
    print("Saving trajectory animation...")
    anim.save('multi_drone_trajectory.gif', writer='pillow', fps=30)
    plt.close()

def main():
    # List of CSV files for each drone
    drone_files = [
        'flights/1.csv',
        'flights/2.csv',
        'flights/8.csv',
        'flights/59.csv',
        'flights/23.csv',
    ]
    
    # Optional: Custom names for each drone
    drone_names = [
        'Drone Alpha',
        'Drone Beta',
        'Drone Gamma',
        'Drone Delta',
        'Drone Epsilon'
    ]
    
    print("Loading data...")
    try:
        drones_data = load_multiple_drone_data(drone_files)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        
        for i, df in enumerate(drones_data):
            print(f"\n{drone_names[i]} information:")
            print(f"Records: {len(df)}")
            print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
            print(f"Position ranges:")
            print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
            print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
            print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        # Create trajectory animation for all drones
        print("\nCreating trajectory animation...")
        create_trajectory_animation(drones_data, drone_names)
        
        print("\nFiles created:")
        print("- multi_drone_trajectory.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# [Previous functions remain the same until create_trajectory_animation]

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Sample data for smoother animation if needed
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def load_multiple_drone_data(file_paths):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    if not file_paths:
        raise ValueError("No drone data files provided!")
    
    for file_path in file_paths:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize all drones to the same time scale
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column):
    """Calculate limits handling NaN/Inf values for multiple drones"""
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return 0, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_enhanced_visualization(drones_data, drone_names=None):
    """Create animated visualization of multiple drone trajectories and derived metrics"""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits for trajectory
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x')
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y')
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z')
    
    # Pre-calculate derived metrics for each drone
    processed_data = []
    for df in drones_data:
        processed_df = df.copy()
        
        # Calculate velocity (change in position over time)
        processed_df['velocity_z'] = processed_df['position_z'].diff() / processed_df['time'].diff()
        
        # Calculate acceleration (change in velocity over time)
        processed_df['acceleration_z'] = processed_df['velocity_z'].diff() / processed_df['time'].diff()
        
        # Replace inf/nan values with 0
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        processed_df = processed_df.fillna(0)
        
        processed_data.append(processed_df)
    
    # Calculate global limits for metrics
    height_min, height_max = get_safe_limits_multiple(processed_data, 'position_z')
    vel_min, vel_max = get_safe_limits_multiple(processed_data, 'velocity_z')
    acc_min, acc_max = get_safe_limits_multiple(processed_data, 'acceleration_z')
    
    # Define colors for each drone
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    # Initialize empty lines and points
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    
    # Create lines for each drone
    for i, df in enumerate(processed_data):
        color = colors[i % len(colors)]
        drone_name = drone_names[i] if drone_names else f'Drone {i+1}'
        
        # Trajectory
        line, = ax_traj.plot([], [], [], f'{color}-', label=f'{drone_name}', linewidth=2)
        point, = ax_traj.plot([], [], [], f'{color}o', markersize=10)
        traj_lines.append(line)
        traj_points.append(point)
        
        # Metrics
        height_line, = ax_height.plot([], [], f'{color}-', label=drone_name)
        velocity_line, = ax_velocity.plot([], [], f'{color}-', label=drone_name)
        acceleration_line, = ax_acceleration.plot([], [], f'{color}-', label=drone_name)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    # Set up trajectory plot
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Flight Trajectories')
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_zlim(z_min, z_max)
    
    # Set up metrics plots
    ax_height.set_title('Height')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_ylim(height_min, height_max)
    ax_height.grid(True)
    
    ax_velocity.set_title('Vertical Velocity')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_ylim(vel_min, vel_max)
    ax_velocity.grid(True)
    
    ax_acceleration.set_title('Vertical Acceleration')
    ax_acceleration.set_xlabel('Time (s)')
    ax_acceleration.set_ylabel('Acceleration (m/s²)')
    ax_acceleration.set_ylim(acc_min, acc_max)
    ax_acceleration.grid(True)
    
    # Add legends
    for ax in [ax_traj, ax_height, ax_velocity, ax_acceleration]:
        ax.legend(prop={'size': 8}, loc='upper right')
    
    def init():
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        # Calculate progress based on frame
        progress = frame / 1000
        
        # Update each drone's visualizations
        for i, df in enumerate(processed_data):
            # Get position data
            x_data = df['position_x'].values
            y_data = df['position_y'].values
            z_data = df['position_z'].values
            time_data = df['time'].values
            
            # Calculate current index based on progress
            idx = int(progress * len(df))
            idx = min(idx, len(df) - 1)
            
            # Update trajectory
            traj_lines[i].set_data(x_data[:idx], y_data[:idx])
            traj_lines[i].set_3d_properties(z_data[:idx])
            traj_points[i].set_data([x_data[idx]], [y_data[idx]])
            traj_points[i].set_3d_properties([z_data[idx]])
            
            # Update metrics
            current_time = time_data[:idx]
            height_lines[i].set_data(current_time, z_data[:idx])
            velocity_lines[i].set_data(current_time, df['velocity_z'].values[:idx])
            acceleration_lines[i].set_data(current_time, df['acceleration_z'].values[:idx])
        
        # Rotate trajectory view
        ax_traj.view_init(elev=30, azim=frame/10)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save animation
    print("Saving enhanced visualization...")
    anim.save('multi_drone_enhanced.gif', writer='pillow', fps=30)
    plt.close()
def main():
    # List of CSV files for each drone
    drone_files = [
        'data files train/drone_1_data.csv',
        'data files train/drones/drone_2_data.csv',
        'data files train/drones/drone_3_data.csv',
        'data files train/drones/drone_4_data.csv',
        'data files train/drones/drone_5_data.csv',
    ]
    
    # Optional: Custom names for each drone
    drone_names = [
        'Drone Alpha',
        'Drone Beta',
        'Drone Gamma',
        'Drone Delta',
        'Drone Epsilon'
    ]
    
    print("Loading data...")
    try:
        drones_data = load_multiple_drone_data(drone_files)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        
        for i, df in enumerate(drones_data):
            print(f"\n{drone_names[i]} information:")
            print(f"Records: {len(df)}")
            print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
            print(f"Position ranges:")
            print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
            print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
            print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        # Create enhanced visualization
        print("\nCreating enhanced visualization...")
        create_enhanced_visualization(drones_data, drone_names)
        
        print("\nFiles created:")
        print("- multi_drone_enhanced.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Sample data for smoother animation if needed
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def load_multiple_drone_data(file_paths):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    if not file_paths:
        raise ValueError("No drone data files provided!")
    
    for file_path in file_paths:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize all drones to the same time scale
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column):
    """Calculate limits handling NaN/Inf values for multiple drones"""
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return 0, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * 0.1
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_enhanced_visualization(drones_data, drone_names=None):
    """Create animated visualization of multiple drone trajectories and derived metrics"""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_distance = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits for trajectory
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x')
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y')
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z')
    
    # Pre-calculate derived metrics for each drone
    processed_data = []
    max_time = 0
    
    for df in drones_data:
        processed_df = df.copy()
        
        # Calculate instantaneous velocity
        dx = processed_df['position_x'].diff()
        dy = processed_df['position_y'].diff()
        dz = processed_df['position_z'].diff()
        dt = processed_df['time'].diff()
        
        processed_df['velocity'] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        
        # Calculate cumulative distance traveled
        processed_df['distance'] = np.sqrt(dx**2 + dy**2 + dz**2).cumsum()
        
        # Replace inf/nan values with 0
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        processed_df = processed_df.fillna(method='ffill').fillna(0)
        
        max_time = max(max_time, processed_df['time'].max())
        processed_data.append(processed_df)
    
    # Calculate global limits for metrics
    height_min = min(df['position_z'].min() for df in processed_data)
    height_max = max(df['position_z'].max() for df in processed_data)
    vel_min = min(df['velocity'].min() for df in processed_data)
    vel_max = max(df['velocity'].max() for df in processed_data)
    dist_min = 0
    dist_max = max(df['distance'].max() for df in processed_data)
    
    # Define colors for each drone
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    # Initialize empty lines and points
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    distance_lines = []
    
    # Create lines for each drone
    for i, df in enumerate(processed_data):
        color = colors[i % len(colors)]
        drone_name = drone_names[i] if drone_names else f'Drone {i+1}'
        
        # Trajectory
        line, = ax_traj.plot([], [], [], f'{color}-', label=f'{drone_name}', linewidth=2)
        point, = ax_traj.plot([], [], [], f'{color}o', markersize=10)
        traj_lines.append(line)
        traj_points.append(point)
        
        # Metrics - using alpha for better visibility of line progression
        height_line, = ax_height.plot([], [], color=color, label=drone_name, linewidth=2, alpha=0.8)
        velocity_line, = ax_velocity.plot([], [], color=color, label=drone_name, linewidth=2, alpha=0.8)
        distance_line, = ax_distance.plot([], [], color=color, label=drone_name, linewidth=2, alpha=0.8)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        distance_lines.append(distance_line)
    
    # Set up trajectory plot
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Flight Trajectories')
    ax_traj.set_xlim(x_min - 1, x_max + 1)
    ax_traj.set_ylim(y_min - 1, y_max + 1)
    ax_traj.set_zlim(z_min - 1, z_max + 1)
    
    # Set up metrics plots with improved styling
    def setup_metric_plot(ax, title, ylabel, ymin, ymax):
        ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0, max_time)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    setup_metric_plot(ax_height, 'Height Over Time', 'Height (m)', height_min - 1, height_max + 1)
    setup_metric_plot(ax_velocity, 'Velocity Over Time', 'Velocity (m/s)', vel_min - 0.1, vel_max + 0.1)
    setup_metric_plot(ax_distance, 'Cumulative Distance', 'Distance (m)', dist_min, dist_max + 1)
    
    # Add legends with improved styling
    for ax in [ax_traj, ax_height, ax_velocity, ax_distance]:
        ax.legend(prop={'size': 8}, loc='upper right', framealpha=0.9)
    
    def init():
        lines = traj_lines + traj_points + height_lines + velocity_lines + distance_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        # Calculate progress based on frame
        progress = frame / 1000
        
        # Update each drone's visualizations
        for i, df in enumerate(processed_data):
            # Calculate current index based on progress
            max_idx = len(df)
            idx = min(int(progress * max_idx), max_idx - 1)
            
            # Get current data up to idx
            current_time = df['time'].values[:idx+1]
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            
            # Update trajectory
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            # Update metrics with smooth line progression
            height_lines[i].set_data(current_time, z_data)
            velocity_lines[i].set_data(current_time, df['velocity'].values[:idx+1])
            distance_lines[i].set_data(current_time, df['distance'].values[:idx+1])
        
        # Rotate trajectory view
        ax_traj.view_init(elev=30, azim=frame/10)
        
        return traj_lines + traj_points + height_lines + velocity_lines + distance_lines
    
    # Create animation with smoother updates
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    # Adjust layout and spacing
    plt.tight_layout()
    
    # Save animation with higher quality
    print("Saving enhanced visualization...")
    anim.save('multi_drone_enhanced.gif', writer='pillow', fps=30, 
              dpi=150, savefig_kwargs={'facecolor': 'white'})
    plt.close()

def main():
    # List of CSV files for each drone
    drone_files = [
        'flights/1.csv',
        'flights/2.csv',
        'flights/8.csv',
        'flights/59.csv',
        'flights/23.csv',
    ]
    
    # Optional: Custom names for each drone
    drone_names = [
        'Drone Alpha',
        'Drone Beta',
        'Drone Gamma',
        'Drone Delta',
        'Drone Epsilon'
    ]
    
    print("Loading data...")
    try:
        drones_data = load_multiple_drone_data(drone_files)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        
        for i, df in enumerate(drones_data):
            print(f"\n{drone_names[i]} information:")
            print(f"Records: {len(df)}")
            print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
            print(f"Position ranges:")
            print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
            print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
            print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        # Create enhanced visualization
        print("\nCreating enhanced visualization...")
        create_enhanced_visualization(drones_data, drone_names)
        
        print("\nFiles created:")
        print("- multi_drone_enhanced.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
/'''

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def load_drone_data(file_path):
    # Load and process flight data for a single drone
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert columns to numeric
    numeric_columns = ['position_x', 'position_y', 'position_z', 'altitude',
                      'wind_speed', 'battery_voltage', 'wind_angle', 'time']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN values in essential columns
    df = df.dropna(subset=['position_x', 'position_y', 'position_z', 'time'])
    
    # Sort by time and reset time to start from 0
    df = df.sort_values('time')
    df['time'] = df['time'] - df['time'].min()
    
    # Calculate velocity and acceleration using rolling windows for smoother results
    # Calculate velocity components
    window_size = 5  # Adjust this value to change smoothing
    df['velocity_x'] = df['position_x'].diff() / df['time'].diff()
    df['velocity_y'] = df['position_y'].diff() / df['time'].diff()
    df['velocity_z'] = df['position_z'].diff() / df['time'].diff()
    
    # Apply rolling mean to smooth velocity
    df['velocity_z'] = df['velocity_z'].rolling(window=window_size, center=True).mean()
    
    # Calculate acceleration from smoothed velocity
    df['acceleration_z'] = df['velocity_z'].diff() / df['time'].diff()
    df['acceleration_z'] = df['acceleration_z'].rolling(window=window_size, center=True).mean()
    
    # Replace NaN values with forward/backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Sample data for smoother animation if needed
    if len(df) > 10000:
        df = df.iloc[::len(df)//10000].copy()
    
    return df

def load_multiple_drone_data(file_paths):
    # Load and process flight data for multiple drones
    all_drones_data = []
    max_time = 0
    
    if not file_paths:
        raise ValueError("No drone data files provided!")
    
    for file_path in file_paths:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize all drones to the same time scale
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column, padding_factor=0.2):
    # # Calculate limits handling NaN/Inf values for multiple drones with more padding
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return -1, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * padding_factor
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_enhanced_visualization(drones_data, drone_names=None):
    # Create animated visualization of multiple drone trajectories and derived metrics # 
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x')
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y')
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z')
    vel_min, vel_max = get_safe_limits_multiple(drones_data, 'velocity_z', padding_factor=0.3)
    acc_min, acc_max = get_safe_limits_multiple(drones_data, 'acceleration_z', padding_factor=0.3)
    
    # Define colors for each drone
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    # Initialize empty lines and points
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    
    # Create lines for each drone
    for i, df in enumerate(drones_data):
        color = colors[i % len(colors)]
        drone_name = drone_names[i] if drone_names else f'Drone {i+1}'
        
        # Trajectory
        line, = ax_traj.plot([], [], [], f'{color}-', label=f'{drone_name}', linewidth=2)
        point, = ax_traj.plot([], [], [], f'{color}o', markersize=10)
        traj_lines.append(line)
        traj_points.append(point)
        
        # Metrics
        height_line, = ax_height.plot([], [], f'{color}-', label=drone_name)
        velocity_line, = ax_velocity.plot([], [], f'{color}-', label=drone_name)
        acceleration_line, = ax_acceleration.plot([], [], f'{color}-', label=drone_name)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    # Set up plots
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Flight Trajectories')
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_zlim(z_min, z_max)
    
    ax_height.set_title('Height')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_ylim(z_min, z_max)
    ax_height.grid(True)
    
    ax_velocity.set_title('Vertical Velocity')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_ylim(vel_min, vel_max)
    ax_velocity.grid(True)
    
    ax_acceleration.set_title('Vertical Acceleration')
    ax_acceleration.set_xlabel('Time (s)')
    ax_acceleration.set_ylabel('Acceleration (m/s²)')
    ax_acceleration.set_ylim(acc_min, acc_max)
    ax_acceleration.grid(True)
    
    # Add legends
    for ax in [ax_traj, ax_height, ax_velocity, ax_acceleration]:
        ax.legend(prop={'size': 8}, loc='upper right')
    
    def init():
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        progress = frame / 1000
        
        for i, df in enumerate(drones_data):
            idx = int(progress * len(df))
            idx = min(idx, len(df) - 1)
            
            # Update trajectory
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            time_data = df['time'].values[:idx+1]
            
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            # Update metrics
            height_lines[i].set_data(time_data, z_data)
            velocity_lines[i].set_data(time_data, df['velocity_z'].values[:idx+1])
            acceleration_lines[i].set_data(time_data, df['acceleration_z'].values[:idx+1])
        
        # Rotate trajectory view
        ax_traj.view_init(elev=30, azim=frame/10)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    plt.tight_layout()
    
    print("Saving enhanced visualization...")
    anim.save('multi_drone_enhanced.gif', writer='pillow', fps=30)
    plt.close()

# Main function remains the same
def main():
    drone_files = [
        'data files train/drone_1_data.csv',
        'data files train/drones/drone_2_data.csv',
        'data files train/drones/drone_3_data.csv',
        'data files train/drones/drone_4_data.csv',
        'data files train/drones/drone_5_data.csv',
    ]
    
    drone_names = [
        'Drone Alpha',
        'Drone Beta',
        'Drone Gamma',
        'Drone Delta',
        'Drone Epsilon'
    ]
    
    try:
        print("Loading data...")
        drones_data = load_multiple_drone_data(drone_files)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        for i, df in enumerate(drones_data):
            print(f"\n{drone_names[i]} information:")
            print(f"Records: {len(df)}")
            print(f"Time span: {df['time'].min():.2f}s to {df['time'].max():.2f}s")
            print(f"Position ranges:")
            print(f"X: {df['position_x'].min():.2f}m to {df['position_x'].max():.2f}m")
            print(f"Y: {df['position_y'].min():.2f}m to {df['position_y'].max():.2f}m")
            print(f"Z: {df['position_z'].min():.2f}m to {df['position_z'].max():.2f}m")
        
        print("\nCreating enhanced visualization...")
        create_enhanced_visualization(drones_data, drone_names)
        
        print("\nFiles created:")
        print("- multi_drone_enhanced.gif")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()"""

'''# visualization.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path)
    
    # Ensure time starts from 0
    df['time'] = df['time'] - df['time'].min()
    
    return df

def load_multiple_drone_data(file_paths):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    for file_path in file_paths:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Ensure all drones have the same time scale
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column, padding_factor=0.2):
    """Calculate limits with padding"""
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return -1, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * padding_factor
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_enhanced_visualization(drones_data, drone_names=None):
    """Create animated visualization of drone swarm"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x')
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y')
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z')
    vel_min, vel_max = get_safe_limits_multiple(drones_data, 'velocity_total')
    acc_min, acc_max = get_safe_limits_multiple(drones_data, 'acceleration_total')
    
    # Define colors for each drone
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown']
    
    # Initialize lines and points
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    
    # Create lines for each drone
    for i, df in enumerate(drones_data):
        color = colors[i % len(colors)]
        drone_name = drone_names[i] if drone_names else f'Drone {i+1}'
        
        line, = ax_traj.plot([], [], [], f'{color}-', label=drone_name, linewidth=2)
        point, = ax_traj.plot([], [], [], f'{color}o', markersize=10)
        traj_lines.append(line)
        traj_points.append(point)
        
        height_line, = ax_height.plot([], [], f'{color}-', label=drone_name)
        velocity_line, = ax_velocity.plot([], [], f'{color}-', label=drone_name)
        acceleration_line, = ax_acceleration.plot([], [], f'{color}-', label=drone_name)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    # Set up plots
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Swarm Formation')
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_zlim(z_min, z_max)
    
    ax_height.set_title('Height')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_ylim(z_min, z_max)
    ax_height.grid(True)
    
    ax_velocity.set_title('Total Velocity')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_ylim(vel_min, vel_max)
    ax_velocity.grid(True)
    
    ax_acceleration.set_title('Total Acceleration')
    ax_acceleration.set_xlabel('Time (s)')
    ax_acceleration.set_ylabel('Acceleration (m/s²)')
    ax_acceleration.set_ylim(acc_min, acc_max)
    ax_acceleration.grid(True)
    
    for ax in [ax_traj, ax_height, ax_velocity, ax_acceleration]:
        ax.legend(prop={'size': 8}, loc='upper right')
    
    def init():
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        progress = frame / 1000
        
        for i, df in enumerate(drones_data):
            idx = int(progress * len(df))
            idx = min(idx, len(df) - 1)
            
            # Update trajectory
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            time_data = df['time'].values[:idx+1]
            
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            # Update metrics
            height_lines[i].set_data(time_data, z_data)
            velocity_lines[i].set_data(time_data, df['velocity_total'].values[:idx+1])
            acceleration_lines[i].set_data(time_data, df['acceleration_total'].values[:idx+1])
        
        # Rotate trajectory view
        ax_traj.view_init(elev=30, azim=frame/10)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    plt.tight_layout()
    print("Saving enhanced visualization...")
    anim.save('drone_swarm_formation.gif', writer='pillow', fps=30)
    plt.close()

def main():
    base_dir = "adjusted_positions"
    drone_files = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir)) 
                  if f.startswith("adjusted_drone_") and f.endswith(".csv")]
    
    drone_names = [f'Drone {i+1}' for i in range(len(drone_files))]
    
    try:
        print("Loading data...")
        drones_data = load_multiple_drone_data(drone_files)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        print("\nCreating visualization...")
        create_enhanced_visualization(drones_data, drone_names)
        
        print("\nVisualization saved as 'drone_swarm_formation.gif'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()'''

'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import glob
from scipy.spatial.distance import cdist

class SwarmOptimizer:
    def __init__(self, n_drones):
        self.n_drones = n_drones
        self.current_leader = 0
        self.fitness_history = []
        
    def calculate_fitness(self, drone_positions, velocities, accelerations):
        """Calculate fitness for each drone based on multiple parameters"""
        fitness_scores = np.zeros(len(drone_positions))
        
        # Calculate centroid of the formation
        centroid = np.mean(drone_positions, axis=0)
        
        # Calculate distances to centroid
        distances = np.linalg.norm(drone_positions - centroid, axis=1)
        
        # Calculate velocity and acceleration magnitudes
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Combine metrics for fitness
        for i in range(len(drone_positions)):
            position_score = 1.0 / (1.0 + distances[i])
            velocity_score = 1.0 / (1.0 + vel_magnitudes[i])
            acceleration_score = 1.0 / (1.0 + acc_magnitudes[i])
            
            fitness_scores[i] = (0.4 * position_score + 
                               0.3 * velocity_score + 
                               0.3 * acceleration_score)
        
        return fitness_scores

    def spider_monkey_phase(self, fitness_scores):
        """Spider Monkey Optimization phase"""
        # Local leader learning phase
        local_groups = np.array_split(np.arange(self.n_drones), 2)
        local_leaders = []
        
        for group in local_groups:
            group_fitness = fitness_scores[group]
            local_leaders.append(group[np.argmax(group_fitness)])
        
        # Global leader learning phase
        global_leader = np.argmax(fitness_scores)
        
        return global_leader, local_leaders

    def whale_optimization_phase(self, drone_positions, fitness_scores):
        """Whale Optimization Algorithm phase"""
        best_position = drone_positions[np.argmax(fitness_scores)]
        a = 2 * (1 - len(self.fitness_history) / 1000)  # Decreasing from 2 to 0
        
        for i in range(len(drone_positions)):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            
            if np.abs(A) < 1:
                D = np.abs(C * best_position - drone_positions[i])
                drone_positions[i] = best_position - A * D
            else:
                D = np.abs(best_position - drone_positions[i])
                drone_positions[i] = D * np.exp(2*r) * np.cos(2*np.pi*r) + best_position
        
        return drone_positions

    def update_leader(self, drone_positions, velocities, accelerations):
        """Update leader using hybrid SMO-WOA approach"""
        fitness_scores = self.calculate_fitness(drone_positions, velocities, accelerations)
        self.fitness_history.append(np.max(fitness_scores))
        
        # Spider Monkey Phase
        global_leader, local_leaders = self.spider_monkey_phase(fitness_scores)
        
        # Whale Optimization Phase
        optimized_positions = self.whale_optimization_phase(drone_positions.copy(), fitness_scores)
        
        # Update leader based on optimization results
        new_fitness = self.calculate_fitness(optimized_positions, velocities, accelerations)
        self.current_leader = np.argmax(new_fitness)
        
        return self.current_leader, optimized_positions

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path)
    
    # Calculate velocities and accelerations if not present
    if 'velocity_total' not in df.columns:
        calculate_metrics(df)
    
    # Ensure time starts from 0
    df['time'] = df['time'] - df['time'].min()
    
    return df

def calculate_metrics(df):
    """Calculate velocity and acceleration metrics"""
    # Time differences
    dt = df['time'].diff().fillna(df['time'][1] - df['time'][0])
    
    # Positions
    positions = df[['position_x', 'position_y', 'position_z']].values
    
    # Calculate velocities
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / dt[1:, np.newaxis]
    
    df['velocity_x'] = velocities[:, 0]
    df['velocity_y'] = velocities[:, 1]
    df['velocity_z'] = velocities[:, 2]
    df['velocity_total'] = np.linalg.norm(velocities, axis=1)
    
    # Calculate accelerations
    accelerations = np.zeros_like(velocities)
    accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt[1:, np.newaxis]
    
    df['acceleration_x'] = accelerations[:, 0]
    df['acceleration_y'] = accelerations[:, 1]
    df['acceleration_z'] = accelerations[:, 2]
    df['acceleration_total'] = np.linalg.norm(accelerations, axis=1)
    
    # Smooth the data
    window = 5
    for col in df.columns:
        if col != 'time':
            df[col] = df[col].rolling(window=window, center=True).mean()
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def load_multiple_drone_data(directory):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    # Find all CSV files in the directory
    csv_files = sorted(glob.glob(os.path.join('adjusted_positions\fixed 1', "adjusted_drone_*.csv")))
    
    # Process each file
    for file_path in csv_files:
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize time scales
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def get_safe_limits_multiple(drones_data, column, padding_factor=0.2):
    """Calculate limits with padding"""
    all_values = np.concatenate([df[column].values for df in drones_data])
    valid_data = all_values[np.isfinite(all_values)]
    
    if len(valid_data) == 0:
        return -1, 1
    
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    else:
        range_val = max_val - min_val
        padding = range_val * padding_factor
        min_val -= padding
        max_val += padding
    
    return min_val, max_val

def create_enhanced_visualization(drones_data, output_path='drone_swarm_formation.gif'):
    """Create animated visualization with dynamic leader selection"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Initialize optimizer
    optimizer = SwarmOptimizer(len(drones_data))
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x', 0.1)
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y', 0.1)
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z', 0.1)
    vel_min, vel_max = get_safe_limits_multiple(drones_data, 'velocity_total', 0.1)
    acc_min, acc_max = get_safe_limits_multiple(drones_data, 'acceleration_total', 0.1)
    
    # Define colors for drones
    colors = plt.cm.tab20(np.linspace(0, 1, len(drones_data)))
    
    # Initialize visualization elements
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    leader_indicator = None
    
    # Create lines for each drone
    for i, df in enumerate(drones_data):
        color = colors[i]
        drone_name = f'Drone {i+1}'
        
        line, = ax_traj.plot([], [], [], color=color, label=drone_name, linewidth=1.5)
        point, = ax_traj.plot([], [], [], color=color, marker='o', markersize=8)
        traj_lines.append(line)
        traj_points.append(point)
        
        height_line, = ax_height.plot([], [], color=color, label=drone_name, linewidth=1)
        velocity_line, = ax_velocity.plot([], [], color=color, label=drone_name, linewidth=1)
        acceleration_line, = ax_acceleration.plot([], [], color=color, label=drone_name, linewidth=1)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    # Set up plots
    setup_plots(ax_traj, ax_height, ax_velocity, ax_acceleration,
               x_min, x_max, y_min, y_max, z_min, z_max,
               vel_min, vel_max, acc_min, acc_max)
    
    def init():
        """Initialize animation"""
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        """Update animation frame"""
        progress = frame / 1000
        
        # Get current positions, velocities, and accelerations
        current_positions = np.array([[df['position_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_z'].iloc[min(int(progress * len(df)), len(df)-1)]] 
                                    for df in drones_data])
        
        current_velocities = np.array([[df['velocity_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                     for df in drones_data])
        
        current_accelerations = np.array([[df['acceleration_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                        for df in drones_data])
        
        # Update leader using optimization
        leader_idx, optimized_positions = optimizer.update_leader(
            current_positions, current_velocities, current_accelerations)
        
        # Update visualization for each drone
        for i, df in enumerate(drones_data):
            idx = min(int(progress * len(df)), len(df)-1)
            
            # Update trajectory
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            time_data = df['time'].values[:idx+1]
            
            # Highlight leader
            if i == leader_idx:
                traj_points[i].set_markersize(12)
                traj_points[i].set_markeredgecolor('red')
            else:
                traj_points[i].set_markersize(8)
                traj_points[i].set_markeredgecolor(colors[i])
            
            # Update plots
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            height_lines[i].set_data(time_data, z_data)
            velocity_lines[i].set_data(time_data, df['velocity_total'].values[:idx+1])
            acceleration_lines[i].set_data(time_data, df['acceleration_total'].values[:idx+1])
        
        # Dynamic view angle
        ax_traj.view_init(elev=20 + 10*np.sin(frame/200), azim=frame/8)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    plt.tight_layout()
    print(f"Saving visualization to {output_path}...")
    anim.save(output_path, writer='pillow', fps=30)
    plt.close()

def setup_plots(ax_traj, ax_height, ax_velocity, ax_acceleration,
                x_min, x_max, y_min, y_max, z_min, z_max,
                vel_min, vel_max, acc_min, acc_max):
    """Set up plot styling and limits"""
    # 3D Trajectory plot
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Swarm Formation with Dynamic Leader')
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_zlim(z_min, z_max)
    ax_traj.grid(True)
    
    # Metrics plots
    for ax, title, ylim in [
        (ax_height, 'Altitude', (z_min, z_max)),
        (ax_velocity, 'Velocity', (vel_min, vel_max)),
        (ax_acceleration, 'Acceleration', (acc_min, acc_max))
    ]:
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{title} ({["m", "m/s", "m/s²"][["Altitude", "Velocity", "Acceleration"].index(title)]})')
        ax.set_ylim(ylim)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

def main():
    output_directory = "adjusted_positions"
    
    try:
        print("Loading drone data...")
        # Load and preprocess the data
        drones_data = load_multiple_drone_data(output_directory)
        
        # Ensure velocity calculations for all drones
        for df in drones_data:
            if 'velocity_total' not in df.columns:
                calculate_metrics(df)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        print("\nCreating visualization...")
        create_enhanced_visualization(drones_data)
        
        print("\nVisualization saved as 'drone_swarm_formation.gif'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_metrics(df):
    """Calculate velocity and acceleration metrics"""
    # Time differences
    dt = df['time'].diff().fillna(df['time'][1] - df['time'][0])
    
    # Calculate positions differences
    pos_cols = ['position_x', 'position_y', 'position_z']
    positions = df[pos_cols].values
    
    # Calculate velocities using central differences
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt[1:-1, np.newaxis])
    velocities[0] = (positions[1] - positions[0]) / dt[1]
    velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    
    # Add velocity columns
    for i, col in enumerate(['velocity_x', 'velocity_y', 'velocity_z']):
        df[col] = velocities[:, i]
    df['velocity_total'] = np.linalg.norm(velocities, axis=1)
    
    # Calculate accelerations using central differences
    accelerations = np.zeros_like(velocities)
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt[1:-1, np.newaxis])
    accelerations[0] = (velocities[1] - velocities[0]) / dt[1]
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt[-1]
    
    # Add acceleration columns
    for i, col in enumerate(['acceleration_x', 'acceleration_y', 'acceleration_z']):
        df[col] = accelerations[:, i]
    df['acceleration_total'] = np.linalg.norm(accelerations, axis=1)
    
    # Smooth the data using Savitzky-Golay filter
    from scipy.signal import savgol_filter
    window = min(15, len(df) // 4)
    if window % 2 == 0:
        window += 1  # Ensure odd window length
    
    for col in df.columns:
        if col != 'time':
            df[col] = savgol_filter(df[col], window, 3, mode='nearest')
    
    return df

class SwarmOptimizer:
    def __init__(self, n_drones):
        self.n_drones = n_drones
        self.current_leader = 0
        self.fitness_history = []
        self.local_search_prob = 0.5
        self.global_search_prob = 0.5
        self.inertia_weight = 0.9
        self.iteration = 0
        self.max_iterations = 1000
        
    def update_search_parameters(self):
        """Update search parameters based on iteration"""
        self.iteration += 1
        progress = self.iteration / self.max_iterations
        
        # Decrease inertia weight linearly
        self.inertia_weight = 0.9 - 0.5 * progress
        
        # Update search probabilities
        self.local_search_prob = 0.5 * (1 - progress)
        self.global_search_prob = 0.5 * (1 + progress)
    
    def calculate_fitness(self, drone_positions, velocities, accelerations):
        """Calculate fitness with enhanced metrics"""
        fitness_scores = np.zeros(len(drone_positions))
        
        # Calculate formation metrics
        centroid = np.mean(drone_positions, axis=0)
        distances = np.linalg.norm(drone_positions - centroid, axis=1)
        
        # Calculate pairwise distances for collision avoidance
        pairwise_distances = cdist(drone_positions, drone_positions)
        np.fill_diagonal(pairwise_distances, np.inf)
        min_distances = np.min(pairwise_distances, axis=1)
        
        # Calculate velocity alignment
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        mean_velocity = np.mean(velocities, axis=0)
        velocity_alignments = np.abs(np.dot(velocities, mean_velocity)) / (vel_magnitudes * np.linalg.norm(mean_velocity) + 1e-10)
        
        # Calculate energy efficiency
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        for i in range(len(drone_positions)):
            # Combine multiple objectives
            formation_score = 1.0 / (1.0 + distances[i])
            collision_score = 1.0 / (1.0 + np.exp(-min_distances[i] + 2))
            velocity_score = velocity_alignments[i]
            efficiency_score = 1.0 / (1.0 + acc_magnitudes[i])
            
            # Weighted combination
            fitness_scores[i] = (0.35 * formation_score + 
                               0.25 * collision_score +
                               0.25 * velocity_score +
                               0.15 * efficiency_score)
        
        return fitness_scores
'''

'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import glob
from scipy.spatial.distance import cdist


class SwarmOptimizer:
    def __init__(self, n_drones):
        self.n_drones = n_drones
        self.current_leader = 0
        self.fitness_history = []
        
    def calculate_fitness(self, drone_positions, velocities, accelerations):
        """Calculate fitness for each drone based on multiple parameters"""
        fitness_scores = np.zeros(len(drone_positions))
        
        # Calculate centroid of the formation
        centroid = np.mean(drone_positions, axis=0)
        
        # Calculate distances to centroid
        distances = np.linalg.norm(drone_positions - centroid, axis=1)
        
        # Calculate velocity and acceleration magnitudes
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Combine metrics for fitness
        for i in range(len(drone_positions)):
            position_score = 1.0 / (1.0 + distances[i])
            velocity_score = 1.0 / (1.0 + vel_magnitudes[i])
            acceleration_score = 1.0 / (1.0 + acc_magnitudes[i])
            
            fitness_scores[i] = (0.4 * position_score + 
                               0.3 * velocity_score + 
                               0.3 * acceleration_score)
        
        return fitness_scores

    def spider_monkey_phase(self, fitness_scores):
        """Spider Monkey Optimization phase"""
        # Local leader learning phase
        local_groups = np.array_split(np.arange(self.n_drones), 2)
        local_leaders = []
        
        for group in local_groups:
            group_fitness = fitness_scores[group]
            local_leaders.append(group[np.argmax(group_fitness)])
        
        # Global leader learning phase
        global_leader = np.argmax(fitness_scores)
        
        return global_leader, local_leaders

    def whale_optimization_phase(self, drone_positions, fitness_scores):
        """Whale Optimization Algorithm phase"""
        best_position = drone_positions[np.argmax(fitness_scores)]
        a = 2 * (1 - len(self.fitness_history) / 1000)  # Decreasing from 2 to 0
        
        for i in range(len(drone_positions)):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            
            if np.abs(A) < 1:
                D = np.abs(C * best_position - drone_positions[i])
                drone_positions[i] = best_position - A * D
            else:
                D = np.abs(best_position - drone_positions[i])
                drone_positions[i] = D * np.exp(2*r) * np.cos(2*np.pi*r) + best_position
        
        return drone_positions

    def update_leader(self, drone_positions, velocities, accelerations):
        """Update leader using hybrid SMO-WOA approach"""
        fitness_scores = self.calculate_fitness(drone_positions, velocities, accelerations)
        self.fitness_history.append(np.max(fitness_scores))
        
        # Spider Monkey Phase
        global_leader, local_leaders = self.spider_monkey_phase(fitness_scores)
        
        # Whale Optimization Phase
        optimized_positions = self.whale_optimization_phase(drone_positions.copy(), fitness_scores)
        
        # Update leader based on optimization results
        new_fitness = self.calculate_fitness(optimized_positions, velocities, accelerations)
        self.current_leader = np.argmax(new_fitness)
        
        return self.current_leader, optimized_positions

def load_drone_data(file_path):
    """Load and process flight data for a single drone"""
    df = pd.read_csv(file_path)
    
    # Calculate velocities and accelerations if not present
    if 'velocity_total' not in df.columns:
        calculate_metrics(df)
    
    # Ensure time starts from 0
    df['time'] = df['time'] - df['time'].min()
    
    return df

def calculate_metrics(df):
    """Calculate velocity and acceleration metrics"""
    # Time differences
    dt = df['time'].diff().fillna(df['time'][1] - df['time'][0])
    
    # Positions
    positions = df[['position_x', 'position_y', 'position_z']].values
    
    # Calculate velocities
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / dt[1:, np.newaxis]
    
    df['velocity_x'] = velocities[:, 0]
    df['velocity_y'] = velocities[:, 1]
    df['velocity_z'] = velocities[:, 2]
    df['velocity_total'] = np.linalg.norm(velocities, axis=1)
    
    # Calculate accelerations
    accelerations = np.zeros_like(velocities)
    accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt[1:, np.newaxis]
    
    df['acceleration_x'] = accelerations[:, 0]
    df['acceleration_y'] = accelerations[:, 1]
    df['acceleration_z'] = accelerations[:, 2]
    df['acceleration_total'] = np.linalg.norm(accelerations, axis=1)
    
    # Smooth the data
    window = 5
    for col in df.columns:
        if col != 'time':
            df[col] = df[col].rolling(window=window, center=True).mean()
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def load_multiple_drone_data(directory):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    # Find all CSV files in the directory
    # Fix: Use os.path.join properly and handle directory existence
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{'adjusted_positions/fixed 1'}' not found")
        
    csv_files = sorted(glob.glob(os.path.join('adjusted_positions/fixed 1', "adjusted_drone_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No drone data files found in '{'adjusted_positions/fixed 1'}'")
    
    print(f"Found {len(csv_files)} drone data files")
    
    # Process each file
    for file_path in csv_files:
        print(f"Loading {os.path.basename(file_path)}")
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize time scales
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def create_enhanced_visualization(drones_data, output_path='drone_swarm_formation.gif'):
    """Create animated visualization with dynamic leader selection"""
    # Validation
    if not drones_data or len(drones_data) == 0:
        raise ValueError("No drone data provided")
    
    print("Setting up visualization...")
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Initialize optimizer
    optimizer = SwarmOptimizer(len(drones_data))
    
    # Trajectory plot (3D)
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Metrics plots (2D)
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x', 0.1)
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y', 0.1)
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z', 0.1)
    vel_min, vel_max = get_safe_limits_multiple(drones_data, 'velocity_total', 0.1)
    acc_min, acc_max = get_safe_limits_multiple(drones_data, 'acceleration_total', 0.1)
    
    # Define colors for drones
    colors = plt.cm.tab20(np.linspace(0, 1, len(drones_data)))
    
    # Initialize visualization elements
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    
    # Create lines for each drone
    for i, df in enumerate(drones_data):
        color = colors[i]
        drone_name = f'Drone {i+1}'
        
        # Fix: Add proper initialization of lines
        line, = ax_traj.plot([], [], [], color=color, label=drone_name, linewidth=1.5)
        point, = ax_traj.plot([], [], [], color=color, marker='o', markersize=8)
        traj_lines.append(line)
        traj_points.append(point)
        
        height_line, = ax_height.plot([], [], color=color, label=drone_name, linewidth=1)
        velocity_line, = ax_velocity.plot([], [], color=color, label=drone_name, linewidth=1)
        acceleration_line, = ax_acceleration.plot([], [], color=color, label=drone_name, linewidth=1)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    # Set up plots
    setup_plots(ax_traj, ax_height, ax_velocity, ax_acceleration,
               x_min, x_max, y_min, y_max, z_min, z_max,
               vel_min, vel_max, acc_min, acc_max)
    
    def init():
        """Initialize animation"""
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        """Update animation frame"""
        progress = frame / 1000
        
        # Get current positions, velocities, and accelerations
        current_positions = np.array([[df['position_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_z'].iloc[min(int(progress * len(df)), len(df)-1)]] 
                                    for df in drones_data])
        
        current_velocities = np.array([[df['velocity_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                     for df in drones_data])
        
        current_accelerations = np.array([[df['acceleration_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                        for df in drones_data])
        
        # Update leader using optimization
        leader_idx, optimized_positions = optimizer.update_leader(
            current_positions, current_velocities, current_accelerations)
        
        # Update visualization for each drone
        for i, df in enumerate(drones_data):
            idx = min(int(progress * len(df)), len(df)-1)
            
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            time_data = df['time'].values[:idx+1]
            
            # Highlight leader
            if i == leader_idx:
                traj_points[i].set_markersize(12)
                traj_points[i].set_markeredgecolor('red')
            else:
                traj_points[i].set_markersize(8)
                traj_points[i].set_markeredgecolor(colors[i])
            
            # Update plots
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            height_lines[i].set_data(time_data, z_data)
            velocity_lines[i].set_data(time_data, df['velocity_total'].values[:idx+1])
            acceleration_lines[i].set_data(time_data, df['acceleration_total'].values[:idx+1])
        
        # Dynamic view angle
        ax_traj.view_init(elev=20 + 10*np.sin(frame/200), azim=frame/8)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    print("Creating animation...")
    # Fix: Add proper animation configuration
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    plt.tight_layout()
    print(f"Saving visualization to {output_path}...")
    # Fix: Add proper writer configuration
    writer = animation.PillowWriter(fps=30)
    anim.save(output_path, writer=writer)
    plt.close()
    print("Visualization completed!")

def main():
    # Fix: Add proper directory handling
    output_directory = "adjusted_positions"
    if not os.path.exists(output_directory):
        print(f"Creating directory: {output_directory}")
        os.makedirs(output_directory)
    
    try:
        print("Loading drone data...")
        drones_data = load_multiple_drone_data(output_directory)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        print("\nCreating visualization...")
        create_enhanced_visualization(drones_data)
        
        print("\nVisualization saved as 'drone_swarm_formation.gif'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import glob
from scipy.spatial.distance import cdist

def get_safe_limits_multiple(drones_data, column, padding=0.1):
    """Calculate safe min/max limits across multiple drone datasets with padding"""
    all_values = []
    for df in drones_data:
        all_values.extend(df[column].values)
    
    min_val = min(all_values)
    max_val = max(all_values)
    range_val = max_val - min_val
    
    return (min_val - range_val * padding,
            max_val + range_val * padding)

def load_drone_data(file_path):
    """Load and process individual drone flight data"""
    df = pd.read_csv(file_path)
    
    # Calculate total velocity and acceleration
    velocity_columns = ['velocity_x', 'velocity_y', 'velocity_z']
    acceleration_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']
    
    df['velocity_total'] = np.sqrt(df[velocity_columns].pow(2).sum(axis=1))
    df['acceleration_total'] = np.sqrt(df[acceleration_columns].pow(2).sum(axis=1))
    
    return df

def setup_plots(ax_traj, ax_height, ax_velocity, ax_acceleration,
                x_min, x_max, y_min, y_max, z_min, z_max,
                vel_min, vel_max, acc_min, acc_max):
    """Set up plot layouts and labels"""
    # 3D Trajectory plot
    ax_traj.set_xlim([x_min, x_max])
    ax_traj.set_ylim([y_min, y_max])
    ax_traj.set_zlim([z_min, z_max])
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_zlabel('Z Position (m)')
    ax_traj.set_title('Drone Swarm Trajectories')
    ax_traj.legend(loc='upper right')
    
    # Height plot
    ax_height.set_ylim([z_min, z_max])
    ax_height.set_xlabel('Time (s)')
    ax_height.set_ylabel('Height (m)')
    ax_height.set_title('Drone Heights')
    ax_height.legend(loc='upper right')
    ax_height.grid(True)
    
    # Velocity plot
    ax_velocity.set_ylim([vel_min, vel_max])
    ax_velocity.set_xlabel('Time (s)')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_title('Drone Velocities')
    ax_velocity.legend(loc='upper right')
    ax_velocity.grid(True)
    
    # Acceleration plot
    ax_acceleration.set_ylim([acc_min, acc_max])
    ax_acceleration.set_xlabel('Time (s)')
    ax_acceleration.set_ylabel('Acceleration (m/s²)')
    ax_acceleration.set_title('Drone Accelerations')
    ax_acceleration.legend(loc='upper right')
    ax_acceleration.grid(True)

class SwarmOptimizer:
    def __init__(self, num_drones):
        self.num_drones = num_drones
        self.previous_leader = None
        
    def update_leader(self, positions, velocities, accelerations):
        """Select leader based on position and dynamics"""
        # Calculate centroid
        centroid = np.mean(positions, axis=0)
        
        # Calculate distances to centroid
        distances = cdist([centroid], positions)[0]
        
        # Calculate velocity magnitudes
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Calculate acceleration magnitudes
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Normalize metrics
        norm_distances = distances / np.max(distances)
        norm_velocities = velocity_magnitudes / np.max(velocity_magnitudes)
        norm_accelerations = acceleration_magnitudes / np.max(acceleration_magnitudes)
        
        # Combined score (lower is better)
        scores = (0.5 * norm_distances + 
                 0.3 * norm_velocities + 
                 0.2 * norm_accelerations)
        
        # Select leader
        leader_idx = np.argmin(scores)
        
        # Calculate optimal positions (simplified)
        optimal_positions = positions.copy()
        for i in range(self.num_drones):
            if i != leader_idx:
                # Move slightly towards leader
                optimal_positions[i] = positions[i] + 0.1 * (positions[leader_idx] - positions[i])
        
        self.previous_leader = leader_idx
        return leader_idx, optimal_positions

def load_multiple_drone_data(directory):
    """Load and process flight data for multiple drones"""
    all_drones_data = []
    max_time = 0
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{'adjusted_positions/fixed 2'}' not found")
        
    csv_files = sorted(glob.glob(os.path.join('adjusted_positions/fixed 2', "adjusted_drone_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No drone data files found in '{'adjusted_positions/fixed 2'}'")
    
    print(f"Found {len(csv_files)} drone data files")
    
    for file_path in csv_files:
        print(f"Loading {os.path.basename(file_path)}")
        df = load_drone_data(file_path)
        max_time = max(max_time, df['time'].max())
        all_drones_data.append(df)
    
    # Normalize time scales
    for df in all_drones_data:
        df['time'] = df['time'] * (max_time / df['time'].max())
    
    return all_drones_data

def create_enhanced_visualization(drones_data, output_path='drone_swarm_formation.gif'):
    """Create animated visualization with dynamic leader selection"""
    if not drones_data or len(drones_data) == 0:
        raise ValueError("No drone data provided")
    
    print("Setting up visualization...")
    plt.style.use('dark_background')  # Enhanced visibility
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    optimizer = SwarmOptimizer(len(drones_data))
    ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
    ax_height = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_acceleration = fig.add_subplot(gs[2, 1])
    
    # Calculate global limits
    x_min, x_max = get_safe_limits_multiple(drones_data, 'position_x', 0.1)
    y_min, y_max = get_safe_limits_multiple(drones_data, 'position_y', 0.1)
    z_min, z_max = get_safe_limits_multiple(drones_data, 'position_z', 0.1)
    vel_min, vel_max = get_safe_limits_multiple(drones_data, 'velocity_total', 0.1)
    acc_min, acc_max = get_safe_limits_multiple(drones_data, 'acceleration_total', 0.1)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(drones_data)))
    
    traj_lines = []
    traj_points = []
    height_lines = []
    velocity_lines = []
    acceleration_lines = []
    
    for i, df in enumerate(drones_data):
        color = colors[i]
        drone_name = f'Drone {i+1}'
        
        line, = ax_traj.plot([], [], [], color=color, label=drone_name, linewidth=1.5)
        point, = ax_traj.plot([], [], [], color=color, marker='o', markersize=8)
        traj_lines.append(line)
        traj_points.append(point)
        
        height_line, = ax_height.plot([], [], color=color, label=drone_name, linewidth=1)
        velocity_line, = ax_velocity.plot([], [], color=color, label=drone_name, linewidth=1)
        acceleration_line, = ax_acceleration.plot([], [], color=color, label=drone_name, linewidth=1)
        
        height_lines.append(height_line)
        velocity_lines.append(velocity_line)
        acceleration_lines.append(acceleration_line)
    
    setup_plots(ax_traj, ax_height, ax_velocity, ax_acceleration,
               x_min, x_max, y_min, y_max, z_min, z_max,
               vel_min, vel_max, acc_min, acc_max)
    
    def init():
        """Initialize animation"""
        lines = traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
        for line in lines:
            line.set_data([], [])
            if line in traj_lines + traj_points:
                line.set_3d_properties([])
        return lines
    
    def update(frame):
        """Update animation frame"""
        progress = frame / 1000
        
        current_positions = np.array([[df['position_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                     df['position_z'].iloc[min(int(progress * len(df)), len(df)-1)]] 
                                    for df in drones_data])
        
        current_velocities = np.array([[df['velocity_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                      df['velocity_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                     for df in drones_data])
        
        current_accelerations = np.array([[df['acceleration_x'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_y'].iloc[min(int(progress * len(df)), len(df)-1)],
                                         df['acceleration_z'].iloc[min(int(progress * len(df)), len(df)-1)]]
                                        for df in drones_data])
        
        leader_idx, optimized_positions = optimizer.update_leader(
            current_positions, current_velocities, current_accelerations)
        
        for i, df in enumerate(drones_data):
            idx = min(int(progress * len(df)), len(df)-1)
            
            x_data = df['position_x'].values[:idx+1]
            y_data = df['position_y'].values[:idx+1]
            z_data = df['position_z'].values[:idx+1]
            time_data = df['time'].values[:idx+1]
            
            if i == leader_idx:
                traj_points[i].set_markersize(12)
                traj_points[i].set_markeredgecolor('red')
            else:
                traj_points[i].set_markersize(8)
                traj_points[i].set_markeredgecolor(colors[i])
            
            traj_lines[i].set_data(x_data, y_data)
            traj_lines[i].set_3d_properties(z_data)
            traj_points[i].set_data([x_data[-1]], [y_data[-1]])
            traj_points[i].set_3d_properties([z_data[-1]])
            
            height_lines[i].set_data(time_data, z_data)
            velocity_lines[i].set_data(time_data, df['velocity_total'].values[:idx+1])
            acceleration_lines[i].set_data(time_data, df['acceleration_total'].values[:idx+1])
        
        ax_traj.view_init(elev=20 + 10*np.sin(frame/200), azim=frame/8)
        
        return traj_lines + traj_points + height_lines + velocity_lines + acceleration_lines
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=1000, interval=20, blit=True)
    
    plt.tight_layout()
    print(f"Saving visualization to {output_path}...")
    writer = animation.PillowWriter(fps=30)
    anim.save(output_path, writer=writer)
    plt.close()
    print("Visualization completed!")

def main():
    output_directory = "adjusted_positions"
    if not os.path.exists(output_directory):
        print(f"Creating directory: {output_directory}")
        os.makedirs(output_directory)
    
    try:
        print("Loading drone data...")
        drones_data = load_multiple_drone_data(output_directory)
        
        print(f"\nLoaded data for {len(drones_data)} drones")
        print("\nCreating visualization...")
        create_enhanced_visualization(drones_data)
        
        print("\nVisualization saved as 'drone_swarm_formation.gif'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()