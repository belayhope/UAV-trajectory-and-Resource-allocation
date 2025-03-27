
# Environment Parameters
NUM_UAVS = 3
NUM_MIOTD = 50
SATELLITE_ALTITUDE = 50500  # meters
MAX_UAV_ALTITUDE = 100  # meters
UAV_GAIN = 5  # dBi
MIOTD_GAIN = 0  # dBi
FREQUENCY = 2e9  # Hz
SPEED_OF_LIGHT = 3e8  # m/s
BANDWIDTH = 5000000  # System bandwidth in Hz (500 MHz)
NOISE_POWER_DENSITY = -174  # dBm/Hz
NOISE_POWER = 10 ** ((NOISE_POWER_DENSITY - 30) / 10) * BANDWIDTH  # Convert to linear scale
CHANNEL_GAIN_REF = 1.42e-4  # Channel gain at reference distance of 1m
# UAV Environment Class
def compute_miotd_load_fairness_per_uav(data_rates_mbps):
    """Computes Jain's fairness index for MIoTD load distribution per UAV."""
    uav_loads = [np.sum(np.argmax(data_rates_mbps, axis=0) == uav_index) for uav_index in range(NUM_UAVS)]
    return compute_user_load_fairness(uav_loads)  # Use the existing fairness function

def compute_user_load_fairness(uav_loads):
    """Computes Jain's fairness index for user load distribution across UAVs."""
    # Convert uav_loads to a NumPy array to ensure element-wise operations
    uav_loads = np.array(uav_loads)
    numerator = (np.sum(uav_loads)) ** 2
    denominator = NUM_UAVS * np.sum(uav_loads ** 2) + 1e-9  # Avoid division by zero
    return numerator / denominator

def compute_bandwidth_fairness(bandwidth_allocation):
    """Computes Jain's fairness index for bandwidth allocation to MIoTDs."""
    numerator = (np.sum(bandwidth_allocation)) ** 2
    denominator = NUM_MIOTD * np.sum(bandwidth_allocation ** 2) + 1e-9  # Avoid division by zero
    return numerator / denominator

#MIoTD positions are now within 3km * 3km
class UAVEnvironment:
    def __init__(self):
        self.state_dim = NUM_UAVS * 3 + NUM_MIOTD * 3 + NUM_UAVS * NUM_MIOTD + NUM_MIOTD
        self.action_dim = NUM_UAVS * 2
        self.miotd_positions = np.random.uniform(0, 3000, (NUM_MIOTD, 3))  # Initialize MIoTD positions
        self.miotd_positions[:, 2] = 0  # Set altitude (z-coordinate) to 0
        self.miotd_positions = np.random.uniform(0, 3000, (NUM_MIOTD, 3))
        self.miotd_positions[:, 2] = 0  # Set altitude (z-coordinate) to 0
        # Initialize uav_positions before calling update_uav_positions
        self.uav_positions = np.zeros((NUM_UAVS, 3))  # Initialize UAV positions
        self.update_uav_positions()  # Initial UAV positioning using GMM
        self.bandwidth_allocation = np.zeros((NUM_UAVS, NUM_MIOTD))  # Initialize as 2D array
        # Initialize data_rates to avoid the AttributeError
        self.data_rates = np.zeros(NUM_MIOTD)  # Initialize data_rates to avoid the AttributeError
        self.data_rates_mbps = self.data_rates / 1e6  # Convert to Mbps and store as an attribute
        # Initialize uav_positions_history
        self.uav_loads_history = []

    def update_uav_positions(self):
        """Update UAV positions based on GMM clustering of MIoTDs and their data rates."""
        # Calculate data rates (same logic as in get_state)
        distances = np.linalg.norm(self.uav_positions[:, np.newaxis, :] - self.miotd_positions, axis=2)
        path_loss = CHANNEL_GAIN_REF / (distances ** 2 + 1e-9)
        signal_power = path_loss * (BANDWIDTH/50)
        snr = signal_power / NOISE_POWER
        data_rates = (BANDWIDTH/50) * np.log2(1 + snr)
        self.data_rates = np.max(data_rates, axis=0)
        self.data_rates_mbps = self.data_rates / 1e6  # Convert to Mbps

        # Combine locations and data rates for GMM input
        gmm_input = np.concatenate([self.miotd_positions, self.data_rates_mbps[:, np.newaxis]], axis=1)

        # Fit GMM with the combined data
        self.gmm = GaussianMixture(n_components=NUM_UAVS, random_state=42).fit(gmm_input)
        self.uav_positions = self.gmm.means_[:, :3]  # Extract UAV positions (first 3 columns)
        self.uav_positions[:, 2] = np.random.uniform(50, 100, NUM_UAVS)  # Set UAV altitudes

    def reset(self):
        self.miotd_positions = np.random.uniform(0, 3000, (NUM_MIOTD, 3))  # Initialize MIoTD positions
        self.miotd_positions[:, 2] = 0  # Set altitude (z-coordinate) to 0
        # Set initial UAV positions to GMM centroids
        self.update_uav_positions()  # Update UAV positions at the start of each episode
        # Reset data_rates in the reset method
        self.data_rates = np.zeros(NUM_MIOTD)
        # The following line was added to fix the AttributeError:
        self.data_rates_mbps = self.data_rates / 1e6  # Convert to Mbps
        self.uav_positions_history = []  # Reset the history for each episode
        return self.get_state()

    def get_state(self):
        distances = np.linalg.norm(self.uav_positions[:, np.newaxis, :] - self.miotd_positions, axis=2)
        path_loss = CHANNEL_GAIN_REF / (distances ** 2 + 1e-9)
        signal_power = path_loss * (BANDWIDTH/50)
        snr = signal_power / NOISE_POWER
        data_rates = (BANDWIDTH/50) * np.log2(1 + snr)
        data_rates_mbps = self.data_rates / 1e6  # Convert to Mbps #Now this line should work correctly.
        avg_data_rate_per_miotd = np.max(data_rates_mbps, axis=0)
        # Change: Updated state calculation for consistency
        state = np.concatenate([self.uav_positions.flatten(), self.miotd_positions.flatten(),
                                self.bandwidth_allocation.flatten(), # Flatten bandwidth_allocation
                                self.data_rates_mbps ])
        return state

    def get_miotd_data_rates(self):
        """Returns the data rates for each MIoTD."""
        return self.data_rates_mbps  # Return data_rates_mbps

    def step(self, actions):
        max_movement = 10  # Maximum movement per step
        min_bandwidth = 1  # Minimum bandwidth allocation per UAV
        max_bandwidth = BANDWIDTH / NUM_UAVS  # Maximum bandwidth allocation per UAV
        penalty_factor = -0.2
        # Update UAV positions dynamically based on conditions (e.g., every N steps)
        if step % 50 == 0:  # Adjust the interval as needed
            self.update_uav_positions()
        # Define movement options
        movements = {
            0: np.array([0, 0, 0]),  # No movement
            1: np.array([-1, 0, 0]),  # Left
            2: np.array([1, 0, 0]),   # Right
            3: np.array([0, 0, 1]),   # Up
            4: np.array([0, 0, -1]),  # Down
            5: np.array([0, 1, 0]),   # Forward
            6: np.array([0, -1, 0]),  # Backward
            7: np.array([-1, 0, 1]),  # Left-up
            8: np.array([1, 0, 1]),   # Right-up
            9: np.array([-1, 0, -1]), # Left-down
           10: np.array([1, 0, -1])  # Right-down
        }
        # Calculate data rates before movement
        distances = np.linalg.norm(self.uav_positions[:, np.newaxis, :] - self.miotd_positions, axis=2)
        path_loss = CHANNEL_GAIN_REF / (distances ** 2 + 1e-9)
        signal_power = path_loss * (BANDWIDTH/50)
        snr = signal_power / NOISE_POWER
        data_rates = (BANDWIDTH/50) * np.log2(1 + snr)
        self.data_rates = np.max(data_rates, axis=0)
        self.data_rates_mbps = self.data_rates / 1e6  # Convert to Mbps and store
        miotd_load_fairness_per_uav = compute_miotd_load_fairness_per_uav(self.data_rates_mbps)
        uav_loads = []
        for uav_index in range(NUM_UAVS):
            connected_miotds = np.where(np.atleast_1d(np.argmax(self.data_rates_mbps>=thrashold_R, axis=0)) == uav_index)[0]   # MIoTDs connected to this UAV
            uav_loads.append(len(connected_miotds))  # Number of connected MIoTDs
        self.uav_loads_history.append(uav_loads)  # Add to history
        # Apply movement based on actions and constraints
        for uav_index in range(NUM_UAVS):  # Iterate over UAVs
            action_index = uav_index * 2  # Calculate starting index for UAV's actions
            location_action = actions[action_index]  # Get location action (index 0 or 1)
            bandwidth_action = actions[action_index + 1]  # Get bandwidth action (index 2 or 3)

            # Discretize location action for movement options
            discretized_movement = int(np.clip(location_action, 0, len(movements) - 1))
            movement = movements.get(discretized_movement, [0, 0, 0])
            # Divide the movement into 1-meter steps
            num_steps = int(np.linalg.norm(movement))  # Calculate the number of 1-meter steps
            if num_steps > 0:
                step_size = movement / num_steps
                # Move the UAV in 1-meter steps
                for _ in range(num_steps):
                    self.uav_positions[uav_index] += step_size  # Apply the step  # Calculate the step size for each dimension
            # --- Bandwidth Allocation (Bandwidth Action) ---
            # Clip bandwidth action to ensure it's within bounds
            self.bandwidth_allocation[uav_index] = np.clip(bandwidth_action, min_bandwidth, max_bandwidth)
            # User Load Fairness and Bandwidth Fairness
            miotd_load_fairness_per_uav = compute_miotd_load_fairness_per_uav(self.data_rates_mbps)
            bandwidth_fairness = compute_bandwidth_fairness(self.bandwidth_allocation)

            # Calculate coverage
            connected_miotd_indices = np.where(env.data_rates_mbps >= thrashold_R)[0]  # Get indices of connected MIoTDs
            connected_miotd_count = len(connected_miotd_indices)  # Get the number of connected MIoTDs using len on the indices
            coverage = connected_miotd_count / NUM_MIOTD
            # Apply movement if constraints are satisfied
            if (coverage <= 0.7 and # Adjust coverage threshold as needed
               50 <= self.uav_positions[uav_index, 2] + movement[2] <= 100 and
               (miotd_load_fairness_per_uav + bandwidth_fairness) / 2 <= 0.7):  # Adjust fairness threshold as needed
               self.uav_positions[uav_index] += movement

        # Apply movement with constraints (e.g., maximum speed)
        max_speed = 10  # Adjust as needed
        movement = np.clip(movement, -max_speed, max_speed)
        self.uav_positions += movement
        avg_data_rate = np.mean(self.data_rates_mbps)
        w1 = 0.5  # Weight for coverage
        w2 = 0.3  # Weight for average data rate
        w3 = 0.2  # Weight for overall fairness
        penalty = penalty_factor * (np.sum(self.bandwidth_allocation > max_bandwidth) +
                               np.sum(self.uav_positions[:, 2] < 50) +
                               np.sum(self.uav_positions[:, 2] > 100))
        reward = (w1 * coverage + w2 * avg_data_rate + w3 * (0.5 * miotd_load_fairness_per_uav + 0.5 * bandwidth_fairness)-penalty)

        return self.get_state(), reward, False, {'miotd_load_fairness_per_uav': miotd_load_fairness_per_uav}
