'''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
from math import radians, cos, sin, sqrt, atan2
import os
import glob

class DroneSwarmLeaderSelector:
    def __init__(self, cycle_time=1, n_drones=7):
        self.cycle_time = cycle_time
        self.n_drones = n_drones
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            max_depth=15,
            random_state=42
        )
        self.current_leader = None
        self.leadership_history = []
        self.drone_data_cache = {}
        
    def load_drone_files(self, data_directory):
        """Load all drone CSV files from a directory"""
        # Look for individual drone files
        drone_files = glob.glob(os.path.join(data_directory, "drone_*_data.csv"))
        
        if len(drone_files) != self.n_drones:
            print(f"Warning: Found {len(drone_files)} drone files, expected {self.n_drones}")
        
        drone_data = {}
        for file in drone_files:
            drone_id = os.path.basename(file).split('_')[1]  # Extract drone number
            drone_data[f'drone_{drone_id}'] = file
            
        return drone_data

    def convert_to_latlong(self, x, y, z, reference_lat=40.45804724, reference_lon=-79.78239570):
        """Convert local coordinates to latitude and longitude"""
        R = 6378137.0
        
        ref_lat = radians(reference_lat)
        ref_lon = radians(reference_lon)
        
        new_lat = reference_lat + (y / R) * (180 / np.pi)
        new_lon = reference_lon + (x / R) * (180 / np.pi) / cos(ref_lat)
        altitude = z
        
        return new_lat, new_lon, altitude

    def calculate_performance_metrics(self, drone_data):
        """Calculate key performance metrics from CSV data"""
        metrics = {}
        
        for drone_id, data in drone_data.items():
            try:
                # Energy Efficiency (0-1)
                power_consumption = data['battery_voltage'] * data['battery_current']
                energy_efficiency = 1 / (1 + power_consumption/100.0)
                
                # Stability Score (0-1)
                angular_stability = np.sqrt(
                    data['angular_x']**2 + 
                    data['angular_y']**2 + 
                    data['angular_z']**2
                )
                orientation_stability = np.sqrt(
                    data['orientation_x']**2 +
                    data['orientation_y']**2 +
                    data['orientation_z']**2 +
                    data['orientation_w']**2
                )
                stability = 1 / (1 + angular_stability + (1 - orientation_stability))
                
                # Position Control (0-1)
                velocity_control = np.sqrt(
                    data['velocity_x']**2 +
                    data['velocity_y']**2 +
                    data['velocity_z']**2
                )
                acceleration_control = np.sqrt(
                    data['linear_acceleration_x']**2 +
                    data['linear_acceleration_y']**2 +
                    data['linear_acceleration_z']**2
                )
                position_control = 1 / (1 + velocity_control + acceleration_control)
                
                # Wind Resistance (0-1)
                wind_effect = data['wind_speed'] * np.cos(np.deg2rad(data['wind_angle']))
                wind_resistance = 1 - (wind_effect / (data['wind_speed'] + 0.001))
                
                # Calculate relative position score
                lat, lon, alt = self.convert_to_latlong(
                    data['position_x'],
                    data['position_y'],
                    data['position_z']
                )
                position_score = 1 / (1 + abs(alt - data['position_z']))
                
                metrics[drone_id] = {
                    'energy_efficiency': energy_efficiency,
                    'stability': stability,
                    'position_control': position_control,
                    'wind_resistance': wind_resistance,
                    'position_score': position_score,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                }
            except Exception as e:
                print(f"Error processing metrics for {drone_id}: {str(e)}")
                continue
            
        return metrics

    def select_leader(self, drone_data):
        """Select leader using Random Forest classification"""
        metrics = self.calculate_performance_metrics(drone_data)
        
        if not metrics:
            raise ValueError("No valid metrics calculated for any drone")
        
        X = []
        drone_ids = []
        feature_names = ['energy_efficiency', 'stability', 'position_control', 
                        'wind_resistance', 'position_score']
        
        for drone_id, m in metrics.items():
            features = [m[feat] for feat in feature_names]
            X.append(features)
            drone_ids.append(drone_id)
        
        X = np.array(X)
        
        # Weighted scoring
        weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
        weighted_scores = np.dot(X, weights)
        y = np.zeros(len(X))
        y[weighted_scores.argmax()] = 1
        
        self.rf_model.fit(X, y)
        leader_probs = self.rf_model.predict_proba(X)[:, 1]
        new_leader = drone_ids[leader_probs.argmax()]
        
        timestamp = datetime.now()
        leadership_score = leader_probs.max()
        
        self.leadership_history.append({
            'timestamp': timestamp,
            'leader': new_leader,
            'score': leadership_score,
            'metrics': metrics[new_leader],
            'feature_importance': dict(zip(feature_names, 
                                        self.rf_model.feature_importances_))
        })
        
        self.current_leader = new_leader
        return new_leader, leadership_score, metrics

    def process_csv_data(self, csv_files):
        """Process multiple CSV files containing drone data"""
        all_drone_data = {}
        
        for drone_id, csv_file in csv_files.items():
            try:
                # Read only the latest data point if file has been updated
                file_stat = os.stat(csv_file)
                
                # Check if file has been modified since last read
                if (drone_id not in self.drone_data_cache or 
                    self.drone_data_cache[drone_id]['mtime'] != file_stat.st_mtime):
                    
                    df = pd.read_csv(csv_file)
                    latest_data = df.iloc[-1].to_dict()
                    
                    # Update cache
                    self.drone_data_cache[drone_id] = {
                        'mtime': file_stat.st_mtime,
                        'data': latest_data
                    }
                
                all_drone_data[drone_id] = self.drone_data_cache[drone_id]['data']
                
            except Exception as e:
                print(f"Error reading data for {drone_id}: {str(e)}")
                continue
            
        return all_drone_data

    def run_leadership_cycle(self, data_directory):
        """Run continuous leadership selection cycle using CSV data"""
        print("Starting Dynamic Leader Selection Cycle")
        print(f"Monitoring {self.n_drones} drones")
        print("---------------------------------------")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Load and process drone files
                csv_files = self.load_drone_files(data_directory)
                if not csv_files:
                    print("No drone files found. Waiting...")
                    time.sleep(self.cycle_time)
                    continue
                
                # Process CSV data
                drone_data = self.process_csv_data(csv_files)
                if not drone_data:
                    print("No valid drone data available. Waiting...")
                    time.sleep(self.cycle_time)
                    continue
                
                # Select leader
                try:
                    leader, score, metrics = self.select_leader(drone_data)
                    
                    # Print cycle results
                    print(f"\nCycle Time: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"Selected Leader: {leader}")
                    print(f"Leadership Score: {score:.4f}")
                    print("\nLeader Metrics:")
                    for metric, value in metrics[leader].items():
                        if metric in ['latitude', 'longitude', 'altitude']:
                            print(f"{metric}: {value:.6f}")
                        else:
                            print(f"{metric}: {value:.4f}")
                    
                    # Print feature importance
                    print("\nFeature Importance:")
                    for feature, importance in self.leadership_history[-1]['feature_importance'].items():
                        print(f"{feature}: {importance:.4f}")
                    
                except Exception as e:
                    print(f"Error in leader selection: {str(e)}")
                
                # Wait for next cycle
                elapsed = time.time() - cycle_start
                if elapsed < self.cycle_time:
                    time.sleep(self.cycle_time - elapsed)
                
        except KeyboardInterrupt:
            print("\nLeadership selection cycle stopped.")
            self.print_summary()

    def print_summary(self):
        """Print detailed summary of leadership history"""
        print("\nLeadership History Summary")
        print("--------------------------")
        
        if not self.leadership_history:
            print("No leadership data recorded.")
            return
        
        leader_stats = {}
        for entry in self.leadership_history:
            leader = entry['leader']
            if leader not in leader_stats:
                leader_stats[leader] = {
                    'count': 0,
                    'scores': [],
                    'metrics': {k: [] for k in entry['metrics'].keys()}
                }
            
            leader_stats[leader]['count'] += 1
            leader_stats[leader]['scores'].append(entry['score'])
            for metric, value in entry['metrics'].items():
                leader_stats[leader]['metrics'][metric].append(value)
        
        total_cycles = len(self.leadership_history)
        print(f"\nTotal Cycles: {total_cycles}")
        
        print("\nLeader Performance Summary:")
        for leader, stats in leader_stats.items():
            percentage = (stats['count'] / total_cycles) * 100
            avg_score = np.mean(stats['scores'])
            print(f"\n{leader}:")
            print(f"Selection Frequency: {stats['count']} times ({percentage:.1f}%)")
            print(f"Average Leadership Score: {avg_score:.4f}")
            print("Average Metrics:")
            for metric, values in stats['metrics'].items():
                if metric in ['latitude', 'longitude', 'altitude']:
                    print(f"  {metric}: {np.mean(values):.6f}")
                else:
                    print(f"  {metric}: {np.mean(values):.4f}")

# Example usage
if __name__ == "__main__":
    # Initialize the selector for 7 drones
    selector = DroneSwarmLeaderSelector(cycle_time=1, n_drones=7)
    
    # Specify the directory containing drone data files
    data_directory = "data files train"
    
    # Run the leadership cycle
    selector.run_leadership_cycle(data_directory)'''

'''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os
import glob

class DroneSwarmLeaderSelector:
    def __init__(self, cycle_time=1, n_drones=7):
        self.cycle_time = cycle_time
        self.n_drones = n_drones
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            max_depth=15,
            random_state=42
        )
        self.current_leader = None
        self.leadership_history = []
        self.parameter_history = {}  # Store parameter history between leader changes
        self.last_leader_change_time = None
    
    def process_csv_data(self, data_directory):
        """
        Process CSV files from the data directory
        Returns a dictionary of drone data
        """
        all_drone_data = {}
        
        # Look for CSV files in the directory
        try:
            csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
            
            if not csv_files:
                print(f"No CSV files found in {data_directory}")
                return {}
            
            for csv_file in csv_files:
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    
                    # Get the latest row of data
                    latest_data = df.iloc[-1].to_dict()
                    
                    # Extract drone ID from filename
                    drone_id = os.path.splitext(os.path.basename(csv_file))[0]
                    
                    # Add signal strength if not present (you can adjust this default)
                    if 'signal_strength' not in latest_data:
                        latest_data['signal_strength'] = 100.0  # Default value
                    
                    all_drone_data[drone_id] = latest_data
                    
                except Exception as e:
                    print(f"Error processing file {csv_file}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error accessing directory {data_directory}: {str(e)}")
            return {}
        
        return all_drone_data
        
    def compare_drone_parameters(self, drone_data):
        """
        Compare key parameters between drones and calculate relative scores
        Returns a dictionary of normalized parameter scores for each drone
        """
        parameter_scores = {}
        
        # Define parameter thresholds and weights
        thresholds = {
            'battery': {
                'min_voltage': 22.0,  # Minimum acceptable voltage
                'min_current': 0.05,  # Minimum current draw
                'weight': 0.25
            },
            'communication': {
                'min_signal': 70.0,  # Minimum signal strength
                'weight': 0.2
            },
            'stability': {
                'max_angular_velocity': 0.1,  # Maximum acceptable angular velocity
                'weight': 0.3
            },
            'position': {
                'max_position_error': 0.5,  # Maximum acceptable position error
                'weight': 0.25
            }
        }
        
        # Calculate scores for each drone
        for drone_id, data in drone_data.items():
            # Battery health score (0-1)
            battery_score = min(1.0, 
                              (data['battery_voltage'] - thresholds['battery']['min_voltage']) / 5.0) * \
                          min(1.0, data['battery_current'] / thresholds['battery']['min_current'])
            
            # Communication score (0-1)
            comm_score = data.get('signal_strength', 0) / 100.0
            
            # Stability score (0-1)
            angular_velocity = np.sqrt(
                data['angular_x']**2 + 
                data['angular_y']**2 + 
                data['angular_z']**2
            )
            stability_score = 1.0 / (1.0 + angular_velocity / thresholds['stability']['max_angular_velocity'])
            
            # Position accuracy score (0-1)
            position_error = np.sqrt(
                data['velocity_x']**2 +
                data['velocity_y']**2 +
                data['velocity_z']**2
            )
            position_score = 1.0 / (1.0 + position_error / thresholds['position']['max_position_error'])
            
            # Calculate weighted total score
            total_score = (
                battery_score * thresholds['battery']['weight'] +
                comm_score * thresholds['communication']['weight'] +
                stability_score * thresholds['stability']['weight'] +
                position_score * thresholds['position']['weight']
            )
            
            # Store all scores
            parameter_scores[drone_id] = {
                'battery_score': battery_score,
                'comm_score': comm_score,
                'stability_score': stability_score,
                'position_score': position_score,
                'total_score': total_score,
                'raw_parameters': {
                    'battery_voltage': data['battery_voltage'],
                    'battery_current': data['battery_current'],
                    'signal_strength': data.get('signal_strength', 0),
                    'angular_velocity': angular_velocity,
                    'position_error': position_error
                }
            }
            
        return parameter_scores

    def train_random_forest(self, parameter_scores):
        """
        Train Random Forest model on current parameter scores
        Returns leadership probabilities for each drone
        """
        # Prepare feature matrix
        feature_names = ['battery_score', 'comm_score', 'stability_score', 'position_score']
        X = []
        drone_ids = []
        
        for drone_id, scores in parameter_scores.items():
            features = [scores[feat] for feat in feature_names]
            X.append(features)
            drone_ids.append(drone_id)
        
        X = np.array(X)
        
        # Create target variable based on total scores
        total_scores = np.array([scores['total_score'] for scores in parameter_scores.values()])
        y = np.zeros(len(X))
        y[total_scores.argmax()] = 1
        
        # Train model and get probabilities
        self.rf_model.fit(X, y)
        probabilities = self.rf_model.predict_proba(X)[:, 1]
        
        # Combine with drone IDs
        return dict(zip(drone_ids, probabilities))

    def select_leader(self, drone_data):
        """
        Select leader based on parameter comparisons and Random Forest
        Returns selected leader and detailed comparison results
        """
        # Get parameter scores for all drones
        parameter_scores = self.compare_drone_parameters(drone_data)
        
        # Get Random Forest probabilities
        rf_probabilities = self.train_random_forest(parameter_scores)
        
        # Combine scores with RF probabilities
        final_scores = {}
        for drone_id in parameter_scores.keys():
            # Weighted combination of parameter scores and RF probability
            final_scores[drone_id] = {
                'parameter_score': parameter_scores[drone_id]['total_score'],
                'rf_probability': rf_probabilities[drone_id],
                'final_score': 0.6 * parameter_scores[drone_id]['total_score'] + 
                              0.4 * rf_probabilities[drone_id]  # Adjustable weights
            }
        
        # Select leader based on final scores
        new_leader = max(final_scores.items(), key=lambda x: x[1]['final_score'])[0]
        
        # Store comparison results
        comparison_results = {
            'parameter_scores': parameter_scores,
            'rf_probabilities': rf_probabilities,
            'final_scores': final_scores,
            'selected_leader': new_leader
        }
        
        return new_leader, comparison_results

    def run_leadership_cycle(self, data_directory):
        """Run continuous leadership selection with parameter tracking"""
        print("Starting Parameter-Based Leader Selection")
        print("----------------------------------------")
        
        try:
            while True:
                cycle_start = time.time()
                
                # Get current drone data
                drone_data = self.process_csv_data(data_directory)
                
                # Select leader and get comparison results
                new_leader, comparison_results = self.select_leader(drone_data)
                
                current_time = datetime.now()
                
                # If this is first selection or leader has changed
                if self.current_leader != new_leader:
                    if self.current_leader is not None:
                        # Print parameter history between last change and now
                        self.print_parameter_history(
                            self.last_leader_change_time,
                            current_time
                        )
                    
                    print(f"\n=== New Leader Selected at {current_time} ===")
                    print(f"Leader changed from {self.current_leader} to {new_leader}")
                    print("\nSelection Criteria:")
                    for drone_id, scores in comparison_results['final_scores'].items():
                        print(f"\nDrone {drone_id}:")
                        print(f"Parameter Score: {scores['parameter_score']:.4f}")
                        print(f"RF Probability: {scores['rf_probability']:.4f}")
                        print(f"Final Score: {scores['final_score']:.4f}")
                    
                    self.current_leader = new_leader
                    self.last_leader_change_time = current_time
                
                # Store parameters for this cycle
                self.store_cycle_parameters(drone_data, comparison_results)
                
                # Wait for next cycle
                elapsed = time.time() - cycle_start
                if elapsed < self.cycle_time:
                    time.sleep(self.cycle_time - elapsed)
                
        except KeyboardInterrupt:
            print("\nLeader selection stopped.")
            if self.last_leader_change_time:
                self.print_parameter_history(
                    self.last_leader_change_time,
                    datetime.now()
                )

    def store_cycle_parameters(self, drone_data, comparison_results):
        """Store parameters for current cycle"""
        current_time = datetime.now()
        self.parameter_history[current_time] = {
            'drone_data': drone_data,
            'comparison_results': comparison_results
        }

    def print_parameter_history(self, start_time, end_time):
        """Print parameter history between two timestamps"""
        print(f"\nParameter History ({start_time} to {end_time}):")
        print("----------------------------------------")
        
        relevant_history = {
            t: data for t, data in self.parameter_history.items()
            if start_time <= t <= end_time
        }
        
        for timestamp, data in sorted(relevant_history.items()):
            print(f"\nTime: {timestamp}")
            for drone_id, params in data['comparison_results']['parameter_scores'].items():
                print(f"\n{drone_id}:")
                for param, value in params['raw_parameters'].items():
                    print(f"  {param}: {value:.4f}")

# Example usage
if __name__ == "__main__":
    selector = DroneSwarmLeaderSelector(cycle_time=1, n_drones=7)
    selector.run_leadership_cycle("data files train")'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import time
import os
import glob

class SpiderWhaleOptimizer:
    def __init__(self, n_particles, n_dimensions, bounds):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            size=(n_particles, n_dimensions)
        )
        self.velocities = np.zeros((n_particles, n_dimensions))
        self.best_positions = self.positions.copy()
        self.best_scores = np.zeros(n_particles)
        self.global_best_position = None
        self.global_best_score = float('-inf')
        
    def update(self, fitness_func, iteration, max_iterations):
        # Spider Monkey phase
        a = 2 * (1 - iteration / max_iterations)  # Linearly decreased
        
        for i in range(self.n_particles):
            # Calculate fitness
            score = fitness_func(self.positions[i])
            
            # Update personal best
            if score > self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i].copy()
                
            # Update global best
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()
        
        # Whale Optimization phase
        for i in range(self.n_particles):
            r = np.random.random()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = np.random.random()
            
            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey
                    D = abs(C * self.global_best_position - self.positions[i])
                    self.positions[i] = self.global_best_position - A * D
                else:
                    # Search for prey
                    random_position = self.positions[np.random.randint(self.n_particles)]
                    D = abs(C * random_position - self.positions[i])
                    self.positions[i] = random_position - A * D
            else:
                # Spiral update
                D = abs(self.global_best_position - self.positions[i])
                self.positions[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + self.global_best_position
        
        # Bound the positions
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])
        
        return self.global_best_position, self.global_best_score

class DroneSwarmLeaderSelector:
    def __init__(self, cycle_time=1, n_drones=7):
        self.cycle_time = cycle_time
        self.n_drones = n_drones
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            max_depth=15,
            random_state=42
        )
        self.nn_model = None
        self.scaler = StandardScaler()
        self.current_leader = None
        self.leadership_history = []
        self.parameter_history = {}
        self.last_leader_change_time = None
        self.training_data = []
        self.optimizer = None
        
    def build_neural_network(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(8,)),  # 8 features
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self, data_directory):
        """Prepare training data from CSV files"""
        all_data = []
        labels = []
        
        csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Extract features
            features = df[[
                'battery_voltage', 'battery_current', 'signal_strength',
                'angular_x', 'angular_y', 'angular_z',
                'velocity_x', 'velocity_y', 'velocity_z'
            ]].values
            
            # Calculate leadership score based on parameters
            leadership_scores = self.calculate_leadership_scores(df)
            
            # Add to training data
            all_data.append(features)
            labels.extend([1 if score > 0.8 else 0 for score in leadership_scores])
            
        return np.vstack(all_data), np.array(labels)
    
    def calculate_leadership_scores(self, df):
        """Calculate leadership scores based on drone parameters"""
        scores = []
        
        for _, row in df.iterrows():
            # Battery health
            battery_score = (row['battery_voltage'] / 25.0) * (row['battery_current'] / 2.0)
            
            # Stability
            stability_score = 1.0 / (1.0 + np.sqrt(
                row['angular_x']**2 + 
                row['angular_y']**2 + 
                row['angular_z']**2
            ))
            
            # Position accuracy
            position_score = 1.0 / (1.0 + np.sqrt(
                row['velocity_x']**2 +
                row['velocity_y']**2 +
                row['velocity_z']**2
            ))
            
            # Combined score
            total_score = (
                0.4 * battery_score +
                0.3 * stability_score +
                0.3 * position_score
            )
            
            scores.append(total_score)
            
        return np.array(scores)
    
    def train_models(self, data_directory):
        """Train both Random Forest and Neural Network models"""
        # Prepare data
        X, y = self.prepare_training_data(data_directory)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_accuracy = self.rf_model.score(X_test_scaled, y_test)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # Train Neural Network
        self.nn_model = self.build_neural_network()
        history = self.nn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Initialize optimizer
        bounds = np.array([
            [0, 1],  # Weight bounds for each parameter
            [0, 1],
            [0, 1],
            [0, 1]
        ])
        self.optimizer = SpiderWhaleOptimizer(
            n_particles=20,
            n_dimensions=4,
            bounds=bounds
        )
    
    def select_leader(self, drone_data):
        """Select leader using combined models and optimization"""
        parameter_scores = self.compare_drone_parameters(drone_data)
        
        # Prepare features for prediction
        features = []
        drone_ids = []
        
        for drone_id, data in drone_data.items():
            feature_vector = [
                data['battery_voltage'],
                data['battery_current'],
                data.get('signal_strength', 100),
                data['angular_x'],
                data['angular_y'],
                data['angular_z'],
                data['velocity_x'],
                data['velocity_y']
            ]
            features.append(feature_vector)
            drone_ids.append(drone_id)
        
        features = np.array(features)
        scaled_features = self.scaler.transform(features)
        
        # Get predictions from both models
        rf_probs = self.rf_model.predict_proba(scaled_features)[:, 1]
        nn_probs = self.nn_model.predict(scaled_features).flatten()
        
        # Optimize weights using Spider-Whale
        def fitness_func(weights):
            combined_scores = (
                weights[0] * rf_probs +
                weights[1] * nn_probs +
                weights[2] * np.array([s['total_score'] for s in parameter_scores.values()]) +
                weights[3] * np.array([1.0 / (1.0 + s['raw_parameters']['position_error']) 
                                     for s in parameter_scores.values()])
            )
            return np.max(combined_scores)
        
        optimal_weights, _ = self.optimizer.update(
            fitness_func,
            len(self.leadership_history),
            1000  # max iterations
        )
        
        # Calculate final scores
        final_scores = {}
        for i, drone_id in enumerate(drone_ids):
            final_scores[drone_id] = {
                'rf_prob': rf_probs[i],
                'nn_prob': nn_probs[i],
                'parameter_score': parameter_scores[drone_id]['total_score'],
                'final_score': (
                    optimal_weights[0] * rf_probs[i] +
                    optimal_weights[1] * nn_probs[i] +
                    optimal_weights[2] * parameter_scores[drone_id]['total_score'] +
                    optimal_weights[3] * (1.0 / (1.0 + parameter_scores[drone_id]['raw_parameters']['position_error']))
                )
            }
        
        # Select leader based on final scores
        new_leader = max(final_scores.items(), key=lambda x: x[1]['final_score'])[0]
        
        comparison_results = {
            'parameter_scores': parameter_scores,
            'rf_probabilities': dict(zip(drone_ids, rf_probs)),
            'nn_probabilities': dict(zip(drone_ids, nn_probs)),
            'final_scores': final_scores,
            'selected_leader': new_leader,
            'optimal_weights': optimal_weights
        }
        
        return new_leader, comparison_results

    # ... (rest of the existing methods remain the same)

# Example usage
if __name__ == "__main__":
    selector = DroneSwarmLeaderSelector(cycle_time=1, n_drones=7)
    
    # Train models first
    print("Training models...")
    selector.train_models("data")
    
    # Run leadership selection
    print("\nStarting leadership selection...")
    selector.run_leadership_cycle("data")