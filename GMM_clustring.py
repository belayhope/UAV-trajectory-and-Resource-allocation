# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:26:35 2025
@author: Belayneh Abebe
"""
#GMM Clustring Model 
# reference : https://chatgpt.com/canvas/shared/67a5a1a7a9188191b4abbf6239a0d26a
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time

# Parameters
n_miotd = 50  # Number of Marine IoT devices
n_uavs = 3  # Number of UAVs
time_slots = 4  # Number of time slots

def generate_miotd_data():
    locations = np.random.rand(n_miotd, 2) * 100  # Assuming a 100x100 km area
    snr = np.random.rand(n_miotd, 1) * 30  # Signal-to-Noise Ratio (SNR) in dB
    bandwidth = np.full((n_miotd, 1), 5)  # 5 MHz bandwidth per device
    noise_power = 1e-9  # Noise power in Watts
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr / 10)
    
    # Calculate Data Rate using Shannon Capacity Formula: C = B * log2(1 + SNR)
    data_rate = bandwidth * np.log2(1 + snr_linear)  # in Mbps
    
    return np.hstack((locations, data_rate))

def jains_fairness_index(loads):
    return (np.sum(loads) ** 2) / (n_uavs * np.sum(loads ** 2) + 1e-6)  # Avoid division by zero

plt.figure(figsize=(10, 8))  # Create a single figure for all time slots
cluster_counts = np.zeros((time_slots, n_uavs))
cluster_loads = np.zeros((time_slots, n_uavs))
fairness_indices = np.zeros(time_slots)
average_data_rates = np.zeros(time_slots)

def normalize_data_rate(data_rate):
    return (data_rate - data_rate.min()) / (data_rate.max() - data_rate.min() + 1e-6)  # Normalize for transparency

for t in range(time_slots):
    print(f"Time Slot {t+1}")
    
    # Generate new MIOTd data for each time slot
    miotd_data = generate_miotd_data()
    
    # Apply Gaussian Mixture Model (GMM) for clustering
    gmm = GaussianMixture(n_components=n_uavs, random_state=42)
    miotd_clusters = gmm.fit_predict(miotd_data)
    
    # Count MIoTD in each cluster and compute total data rate per UAV
    for i in range(n_uavs):
        cluster_counts[t, i] = np.sum(miotd_clusters == i)
        cluster_loads[t, i] = np.sum(miotd_data[miotd_clusters == i, 2])
    
    # Calculate Jain's Fairness Index for this time slot
    fairness_indices[t] = jains_fairness_index(cluster_loads[t, :])
    
    # Calculate average MIoTD data rate for this time slot
    average_data_rates[t] = np.mean(miotd_data[:, 2])
    
    # Plot the clustering results in a subplot
    
    plt.subplot(2, 2, t+1)
    colors = ['r', 'g', 'b']
    for i in range(n_uavs):
        alpha_values = normalize_data_rate(miotd_data[miotd_clusters == i, 2])
        plt.scatter(miotd_data[miotd_clusters == i, 0], miotd_data[miotd_clusters == i, 1], 
                    c=colors[i], label=f'MIoTDs_ UAV {i+1}', alpha=alpha_values)
    
    # Plot UAV centroids
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='black', marker='X', s=100, label='UAV Centers')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Time Slot {t+1}')
    plt.legend()
    plt.grid()
    
    # Print UAV locations (centroids)
    print("UAV Cluster Centers:")
    print(gmm.means_)
    
    time.sleep(1)  # Simulate real-time updates

plt.tight_layout()
plt.show()

# Plot number of MIoTD in each cluster over time
plt.figure(figsize=(8, 6))
for i in range(n_uavs):
    plt.plot(range(1, time_slots + 1), cluster_counts[:, i], marker='o', label=f'UAV {i+1}')
plt.xlabel('Time Slot')
plt.ylabel('Number of MIoTD')
plt.title('Number of MIoTD in Each Cluster Over Time')
plt.legend()
plt.grid()
plt.show()

# Plot MIoTD data rate fairness across UAVs
plt.figure(figsize=(8, 6))
for i in range(n_uavs):
    plt.plot(range(1, time_slots + 1), cluster_loads[:, i], marker='s', label=f'UAV {i+1}')
plt.xlabel('Time Slot')
plt.ylabel('Total MIoTD Data Rate (Mbps)')
plt.title('MIoTD Data Rate Fairness Across UAVs Over Time')
plt.legend()
plt.grid()
plt.show()

# Plot Jain's Fairness Index over time
plt.figure(figsize=(8, 6))
plt.plot(range(1, time_slots + 1), fairness_indices, marker='d', linestyle='-', color='purple', label='Fairness Index')
plt.xlabel('Time Slot')
plt.ylabel("Jain's Fairness Index")
plt.title("Fairness of MIoTD Data Rate Distribution Over Time")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# Plot average MIoTD data rate per time slot
plt.figure(figsize=(8, 6))
plt.plot(range(1, time_slots + 1), average_data_rates, marker='x', linestyle='-', color='blue', label='Avg Data Rate')
plt.xlabel('Time Slot')
plt.ylabel('Average MIoTD Data Rate (Mbps)')
plt.title('Average MIoTD Data Rate Over Time')
plt.legend()
plt.grid()
plt.show()
