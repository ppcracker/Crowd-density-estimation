# 3. Colab for Graph Plotting

import matplotlib.pyplot as plt
import random
import numpy as np

# Example Data
time_intervals = np.arange(0, 50, 5)  # every 5 seconds
people_count = [random.randint(10, 100) for _ in range(len(time_intervals))]

# Time vs People Count
plt.figure(figsize=(8,5))
plt.plot(time_intervals, people_count, marker='o')
plt.title('Time vs People Count')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of People')
plt.grid(True)
plt.savefig('time_vs_people.png')
plt.show()

# Density Distribution
plt.figure(figsize=(8,5))
plt.hist(people_count, bins=[0,30,60,90,120], edgecolor='black')
plt.title('Density Distribution')
plt.xlabel('People Count Range')
plt.ylabel('Frequency')
plt.savefig('density_distribution.png')
plt.show()

# Download graphs
files.download("time_vs_people.png")
files.download("density_distribution.png")

# --- End of Graph Plotting Colab ---
