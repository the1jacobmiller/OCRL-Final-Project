import numpy as np
import matplotlib.pyplot as plt

merge = np.load('plot_data/flow_10.0.npy', allow_pickle=True)
leading = np.load('plot_data/flow_00.0.npy', allow_pickle=True)

acc_merge = merge[:,2].astype(np.float)
acc_leading = leading[:,2].astype(np.float)

pos_merge = merge[:,3].astype(np.float)
pos_leading = leading[:,3].astype(np.float)

vel_merge = merge[:,4].astype(np.float)
vel_leading = leading[:,4].astype(np.float)

plt.title("Accelerations")
plt.xlabel("Time Step")
plt.ylabel("Acceleration (m/s^2)")
plt.plot(acc_merge, color ="red", label = "merging vehicle")
plt.plot(acc_leading, color ="green", label = "leading inflow vehicle")
plt.legend(loc="upper right")
plt.show()

plt.title("Positions")
plt.xlabel("Time Step")
plt.ylabel("Positions (m)")
plt.plot(pos_merge, color ="red", label = "merging vehicle")
plt.plot(pos_leading, color ="green", label = "leading inflow vehicle")
plt.legend(loc="upper right")
plt.show()

plt.title("Velocities")
plt.xlabel("Time Step")
plt.ylabel("Velocities (m/s)")
plt.plot(vel_merge, color ="red", label = "merging vehicle")
plt.plot(vel_leading, color ="green", label = "leading inflow vehicle")
plt.legend(loc="upper right")
plt.show()
