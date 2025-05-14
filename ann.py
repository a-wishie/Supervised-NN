import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

class DroneDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DroneLeaderNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.drone_data = {}
        self.scaler = StandardScaler()
        
    def load_drone_data(self):
        """Load data from CSV file"""
        try:
            # Load single CSV file
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                # Assuming the CSV has a column to identify different drones
                # If not, we'll treat it as a single drone's data
                if 'drone_id' in df.columns:
                    for drone_id, group in df.groupby('drone_id'):
                        self.drone_data[str(drone_id)] = group
                        print(f"Loaded data for drone {drone_id} with {len(group)} records")
                else:
                    self.drone_data['drone1'] = df
                    print(f"Loaded single file with {len(df)} records")
            
            # Load multiple files from directory
            elif os.path.isdir(self.data_path):
                for file in os.listdir(self.data_path):
                    if file.endswith('.csv'):
                        drone_id = file.split('.')[0]
                        file_path = os.path.join(self.data_path, file)
                        self.drone_data[drone_id] = pd.read_csv(file_path)
                        print(f"Loaded {file} with {len(self.drone_data[drone_id])} records")
            
            else:
                raise ValueError(f"Invalid path: {self.data_path}")
                
            if not self.drone_data:
                raise ValueError("No data was loaded")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    # Rest of the class implementation remains the same...
        
    def prepare_training_data(self):
        """Prepare data for ANN training"""
        all_features = []
        all_labels = []
        
        for drone_id, df in self.drone_data.items():
            # Calculate additional features
            df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2 + df['velocity_z']**2)
            df['acceleration'] = np.sqrt(
                df['linear_acceleration_x']**2 + 
                df['linear_acceleration_y']**2 + 
                df['linear_acceleration_z']**2
            )
            df['angular_velocity'] = np.sqrt(
                df['angular_x']**2 + df['angular_y']**2 + df['angular_z']**2
            )
            
            # Select features for training
            features = df[[
                'battery_voltage', 'battery_current', 'wind_speed', 'wind_angle',
                'speed', 'acceleration', 'angular_velocity',
                'position_x', 'position_y', 'position_z'
            ]].values
            
            # Create synthetic labels based on performance metrics
            battery_health = df['battery_voltage'] * df['battery_current']
            stability = 1 / (1 + df['angular_velocity'])
            speed_efficiency = 1 - abs(df['speed'] - df['speed'].mean()) / df['speed'].max()
            wind_resilience = 1 - (df['wind_speed'] * np.cos(np.deg2rad(df['wind_angle']))) / df['wind_speed'].max()
            
            labels = (0.4 * battery_health / battery_health.max() +
                     0.3 * stability / stability.max() +
                     0.2 * speed_efficiency +
                     0.1 * wind_resilience)
            
            all_features.append(features)
            all_labels.append(labels)
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

def train_leader_selection_model(data_processor, epochs=50, batch_size=32, learning_rate=0.001):
    """Train the leader selection neural network"""
    # Prepare data
    X, y = data_processor.prepare_training_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = DroneDataset(X_train, y_train)
    val_dataset = DroneDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = DroneLeaderNet(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Create log file
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        batch_count = 0
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 50)
        
        # Training phase
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        predictions = []
        actual_values = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
                predictions.extend(outputs.numpy().flatten())
                actual_values.extend(batch_y.numpy().flatten())
        
        avg_train_loss = epoch_train_loss / batch_count
        avg_val_loss = val_loss / len(val_loader)
        
        # Log results
        log_message = (
            f"\nEpoch {epoch+1}/{epochs}\n"
            f"Average Training Loss: {avg_train_loss:.4f}\n"
            f"Validation Loss: {avg_val_loss:.4f}\n"
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
            "-" * 50
        )
        
        print(log_message)
        with open(log_filename, 'a') as f:
            f.write(log_message)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
    
    return model, data_processor.scaler

class SwarmSimulator:
    def __init__(self, trained_model, scaler, drone_data):
        self.model = trained_model
        self.scaler = scaler
        self.drone_data = drone_data
        self.current_indices = {drone_id: 0 for drone_id in drone_data.keys()}
    
    def get_next_state(self, drone_id):
        """Get next state from historical data"""
        df = self.drone_data[drone_id]
        idx = self.current_indices[drone_id]
        
        if idx >= len(df):
            return None
        
        state = df.iloc[idx]
        self.current_indices[drone_id] += 1
        return state
    
    def predict_leader_score(self, state):
        """Predict leadership score for a given state"""
        features = np.array([[
            state['battery_voltage'], state['battery_current'],
            state['wind_speed'], state['wind_angle'],
            state['speed'], state['acceleration'],
            state['angular_velocity'],
            state['position_x'], state['position_y'], state['position_z']
        ]])
        
        scaled_features = self.scaler.transform(features)
        with torch.no_grad():
            score = self.model(torch.FloatTensor(scaled_features))
        return score.item()

def main():
    # Load and process data
    data_processor = DataProcessor("flights.csv")
    data_processor.load_drone_data()
    
    # Train the model
    model, scaler = train_leader_selection_model(data_processor)
    
    # Initialize simulator
    simulator = SwarmSimulator(model, scaler, data_processor.drone_data)
    
    # Simulation loop
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}")
        print("-" * 50)
        
        # Get current states for all drones
        current_states = {}
        leader_scores = {}
        
        for drone_id in simulator.drone_data.keys():
            state = simulator.get_next_state(drone_id)
            if state is None:
                continue
                
            current_states[drone_id] = state
            leader_scores[drone_id] = simulator.predict_leader_score(state)
        
        if not current_states:
            break
        
        # Select leader
        current_leader = max(leader_scores.items(), key=lambda x: x[1])
        
        # Display status
        print(f"Current Leader: Drone {current_leader[0]}")
        print(f"Leader Score: {current_leader[1]:.4f}")
        print("\nDrone Status:")
        
        for drone_id, state in current_states.items():
            print(f"\nDrone {drone_id}:")
            print(f"Position: ({state['position_x']:.2f}, {state['position_y']:.2f}, {state['position_z']:.2f})")
            print(f"Battery: {state['battery_voltage']*state['battery_current']:.2f}W")
            print(f"Speed: {state['speed']:.2f} m/s")
            print(f"Leader Score: {leader_scores[drone_id]:.4f}")
        
        iteration += 1

if __name__ == "__main__":
    main()

'''
Drone drone1:
Position: (-79.78, 40.46, 289.08)
Battery: 438.92W
Speed: 3.98 m/s
Leader Score: 0.7480

Iteration 924
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7701

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.08)
Battery: 440.13W
Speed: 3.98 m/s
Leader Score: 0.7701

Iteration 925
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7624

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.11)
Battery: 449.21W
Speed: 4.00 m/s
Leader Score: 0.7624

Iteration 926
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7544

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.13)
Battery: 444.61W
Speed: 4.00 m/s
Leader Score: 0.7544

Iteration 927
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7710

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.13)
Battery: 447.35W
Speed: 4.00 m/s
Leader Score: 0.7710

Iteration 928
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7518

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.14)
Battery: 430.58W
Speed: 4.00 m/s
Leader Score: 0.7518

Iteration 929
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7404

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.15)
Battery: 432.64W
Speed: 4.00 m/s
Leader Score: 0.7404

Iteration 930
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7771

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.13)
Battery: 474.09W
Speed: 4.01 m/s
Leader Score: 0.7771

Iteration 931
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7755

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.14)
Battery: 443.40W
Speed: 4.00 m/s
Leader Score: 0.7755

Iteration 932
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7647

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.14)
Battery: 441.86W
Speed: 3.99 m/s
Leader Score: 0.7647

Iteration 933
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7484

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.13)
Battery: 441.96W
Speed: 3.99 m/s
Leader Score: 0.7484

Iteration 934
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7567

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.13)
Battery: 415.40W
Speed: 3.99 m/s
Leader Score: 0.7567

Iteration 935
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7373

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.12)
Battery: 398.02W
Speed: 3.99 m/s
Leader Score: 0.7373

Iteration 936
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7678

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.09)
Battery: 456.13W
Speed: 3.99 m/s
Leader Score: 0.7678

Iteration 937
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7681

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.07)
Battery: 419.70W
Speed: 4.00 m/s
Leader Score: 0.7681

Iteration 938
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7611

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.06)
Battery: 436.99W
Speed: 4.00 m/s
Leader Score: 0.7611

Iteration 939
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7704

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.03)
Battery: 438.97W
Speed: 4.02 m/s
Leader Score: 0.7704

Iteration 940
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7342

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.01)
Battery: 427.15W
Speed: 4.04 m/s
Leader Score: 0.7342

Iteration 941
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7570

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.97)
Battery: 456.85W
Speed: 4.02 m/s
Leader Score: 0.7570

Iteration 942
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7748

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.91)
Battery: 457.26W
Speed: 4.04 m/s
Leader Score: 0.7748

Iteration 943
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7425

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.89)
Battery: 428.21W
Speed: 4.04 m/s
Leader Score: 0.7425

Iteration 944
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7602

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.87)
Battery: 420.44W
Speed: 4.05 m/s
Leader Score: 0.7602

Iteration 945
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7620

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.81)
Battery: 439.82W
Speed: 4.06 m/s
Leader Score: 0.7620

Iteration 946
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7628

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.75)
Battery: 460.80W
Speed: 4.06 m/s
Leader Score: 0.7628

Iteration 947
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7426

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.73)
Battery: 427.84W
Speed: 4.07 m/s
Leader Score: 0.7426

Iteration 948
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7748

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.69)
Battery: 462.13W
Speed: 4.07 m/s
Leader Score: 0.7748

Iteration 949
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7770

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.64)
Battery: 469.78W
Speed: 4.07 m/s
Leader Score: 0.7770

Iteration 950
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7465

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.62)
Battery: 424.84W
Speed: 4.07 m/s
Leader Score: 0.7465

Iteration 951
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7748

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.57)
Battery: 471.17W
Speed: 4.07 m/s
Leader Score: 0.7748

Iteration 952
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7321

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.55)
Battery: 435.30W
Speed: 4.05 m/s
Leader Score: 0.7321

Iteration 953
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7636

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.51)
Battery: 411.05W
Speed: 4.03 m/s
Leader Score: 0.7636

Iteration 954
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7678

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.50)
Battery: 449.82W
Speed: 4.03 m/s
Leader Score: 0.7678

Iteration 955
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7311

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.46)
Battery: 400.18W
Speed: 4.01 m/s
Leader Score: 0.7311

Iteration 956
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7697

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.44)
Battery: 463.89W
Speed: 4.01 m/s
Leader Score: 0.7697

Iteration 957
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7490

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.41)
Battery: 380.44W
Speed: 4.01 m/s
Leader Score: 0.7490

Iteration 958
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7417

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.39)
Battery: 442.54W
Speed: 4.00 m/s
Leader Score: 0.7417

Iteration 959
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7326

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.38)
Battery: 419.65W
Speed: 4.00 m/s
Leader Score: 0.7326

Iteration 960
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7205

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.37)
Battery: 414.03W
Speed: 3.99 m/s
Leader Score: 0.7205

Iteration 961
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7655

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.36)
Battery: 431.34W
Speed: 3.99 m/s
Leader Score: 0.7655

Iteration 962
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7619

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.35)
Battery: 439.15W
Speed: 3.99 m/s
Leader Score: 0.7619

Iteration 963
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7658

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.35)
Battery: 441.01W
Speed: 3.99 m/s
Leader Score: 0.7658

Iteration 964
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7602

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.35)
Battery: 416.60W
Speed: 3.99 m/s
Leader Score: 0.7602

Iteration 965
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7582

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.34)
Battery: 404.82W
Speed: 4.00 m/s
Leader Score: 0.7582

Iteration 966
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7660

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.35)
Battery: 451.10W
Speed: 4.00 m/s
Leader Score: 0.7660

Iteration 967
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7508

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.31)
Battery: 422.22W
Speed: 4.03 m/s
Leader Score: 0.7508

Iteration 968
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7762

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.32)
Battery: 442.31W
Speed: 4.02 m/s
Leader Score: 0.7762

Iteration 969
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7348

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 424.94W
Speed: 4.02 m/s
Leader Score: 0.7348

Iteration 970
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7536

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.30)
Battery: 405.98W
Speed: 4.00 m/s
Leader Score: 0.7536

Iteration 971
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7540

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.30)
Battery: 420.60W
Speed: 4.00 m/s
Leader Score: 0.7540

Iteration 972
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7560

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 433.74W
Speed: 4.00 m/s
Leader Score: 0.7560

Iteration 973
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7830

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 488.75W
Speed: 4.00 m/s
Leader Score: 0.7830

Iteration 974
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7334

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 423.27W
Speed: 4.02 m/s
Leader Score: 0.7334

Iteration 975
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7617

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 457.17W
Speed: 4.03 m/s
Leader Score: 0.7617

Iteration 976
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7408

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.28)
Battery: 464.24W
Speed: 4.05 m/s
Leader Score: 0.7408

Iteration 977
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7547

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.29)
Battery: 468.70W
Speed: 4.06 m/s
Leader Score: 0.7547

Iteration 978
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7684

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.30)
Battery: 494.49W
Speed: 4.06 m/s
Leader Score: 0.7684

Iteration 979
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7259

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.31)
Battery: 438.23W
Speed: 4.05 m/s
Leader Score: 0.7259

Iteration 980
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7411

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.32)
Battery: 441.75W
Speed: 4.04 m/s
Leader Score: 0.7411

Iteration 981
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7720

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.33)
Battery: 448.89W
Speed: 4.03 m/s
Leader Score: 0.7720

Iteration 982
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7463

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.34)
Battery: 472.73W
Speed: 4.03 m/s
Leader Score: 0.7463

Iteration 983
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7901

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.37)
Battery: 461.14W
Speed: 4.04 m/s
Leader Score: 0.7901

Iteration 984
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7669

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.40)
Battery: 446.73W
Speed: 4.03 m/s
Leader Score: 0.7669

Iteration 985
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7892

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.42)
Battery: 499.81W
Speed: 4.02 m/s
Leader Score: 0.7892

Iteration 986
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7808

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.45)
Battery: 466.20W
Speed: 4.03 m/s
Leader Score: 0.7808

Iteration 987
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7532

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.49)
Battery: 434.42W
Speed: 4.04 m/s
Leader Score: 0.7532

Iteration 988
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7571

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.52)
Battery: 464.55W
Speed: 4.02 m/s
Leader Score: 0.7571

Iteration 989
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7779

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.59)
Battery: 459.76W
Speed: 4.01 m/s
Leader Score: 0.7779

Iteration 990
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7574

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.63)
Battery: 457.24W
Speed: 4.02 m/s
Leader Score: 0.7574

Iteration 991
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7634

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.68)
Battery: 481.32W
Speed: 4.02 m/s
Leader Score: 0.7634

Iteration 992
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7717

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.70)
Battery: 450.33W
Speed: 4.02 m/s
Leader Score: 0.7717

Iteration 993
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7733

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.78)
Battery: 436.25W
Speed: 4.02 m/s
Leader Score: 0.7733

Iteration 994
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7597

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.87)
Battery: 444.43W
Speed: 4.01 m/s
Leader Score: 0.7597

Iteration 995
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7730

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.89)
Battery: 456.94W
Speed: 4.02 m/s
Leader Score: 0.7730

Iteration 996
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7513

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 288.96)
Battery: 480.11W
Speed: 4.01 m/s
Leader Score: 0.7513

Iteration 997
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7692

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.01)
Battery: 452.75W
Speed: 4.02 m/s
Leader Score: 0.7692

Iteration 998
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7716

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.07)
Battery: 432.30W
Speed: 4.02 m/s
Leader Score: 0.7716

Iteration 999
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7623

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.15)
Battery: 462.48W
Speed: 4.03 m/s
Leader Score: 0.7623

Iteration 1000
--------------------------------------------------
Current Leader: Drone drone1
Leader Score: 0.7663

Drone Status:

Drone drone1:
Position: (-79.78, 40.46, 289.18)
Battery: 441.42W
Speed: 4.04 m/s
Leader Score: 0.7663
PS C:\Users\Lenovo\Downloads\finaldata>

'''