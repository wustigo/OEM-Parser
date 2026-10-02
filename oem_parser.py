import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67 * 10**-11
M_earth = 5.972 * 10**24
R_earth = 6371000
A_heatshield = 19.87
v_rotation = 393  # Earth rotation at 32.3 deg latitude

def load_data(filename):
    times, x, y, z, vx, vy, vz, mass = [], [], [], [], [], [], [], []
    prev_mass = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]  # Skip the header
        for line in lines:
            parts = line.split()
            if len(parts) >= 8:
                current_mass_lb = float(parts[7])
                
                # Skip if mass suddenly drops
                if prev_mass is not None:
                    mass_drop = prev_mass - current_mass_lb
                    
                    # Ignore data points with > 50 lb mass drop
                    if mass_drop > 50:
                        continue
                
                # Store valid data and convert imperial units to SI units
                times.append(float(parts[0]))
                x.append(float(parts[1]) * 0.3048)
                y.append(float(parts[2]) * 0.3048)
                z.append(float(parts[3]) * 0.3048)
                vx.append(float(parts[4]) * 0.3048)
                vy.append(float(parts[5]) * 0.3048)
                vz.append(float(parts[6]) * 0.3048)
                mass.append(current_mass_lb * 0.453592)
                
                prev_mass = current_mass_lb
    
    return np.array(times), np.array(x), np.array(y), np.array(z), \
           np.array(vx), np.array(vy), np.array(vz), np.array(mass)

# Air density function
def air_density(altitude_m):
    h = altitude_m
    if h < 11000:
        T = 15.04 - 0.00649 * h
        P = 101.29 * ((T + 273.1) / 288.08) ** 5.256
    elif h < 25000:
        T = -56.46
        P = 22.65 * np.exp(1.73 - 0.000157 * h)
    else:
        T = -131.21 + 0.00299 * h
        P = 2.488 * ((T + 273.1) / 216.6) ** (-11.388)
    return max(P / (0.2869 * (T + 273.1)), 0)

# Pearson Correlation Function
def pearson_correlation(x, y):
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    return np.sum((x - mean_x) * (y - mean_y)) / (n * std_x * std_y)

# Loading data
t, x, y, z, vx, vy, vz, m = load_data('/home/rick/Downloads/artemis/all-artemis-ii-oem-files/2026.04.10 - Post-RTC3 to EI')

# Calculate derived quantities
r = np.sqrt(x**2 + y**2 + z**2)
altitude = r - R_earth
v_inertial = np.sqrt(vx**2 + vy**2 + vz**2)
v_rel = v_inertial - v_rotation

# Energies
KE = 0.5 * m * v_rel**2
U = -G * M_earth * m / r
E_mech = KE + U

# Dissipation rate
dissipation_rate = np.gradient(E_mech, t)

# Drag power
density = np.array([air_density(h) for h in altitude])
P_drag = 0.5 * 1.0 * A_heatshield * density * v_rel**3

# Filter out parachute events (extreme negative spikes)
valid = dissipation_rate > -2.5 * 10**9
correlation = pearson_correlation(np.abs(dissipation_rate[valid]), P_drag[valid])

print(f"\nPearson correlation: {correlation:.3f}")
print(f"Entry altitude: {altitude[0]/1000:.1f} km")
print(f"Splashdown altitude: {altitude[-1]/1000:.1f} km")
print(f"Entry velocity: {v_rel[0]/1000:.2f} km/s")
print(f"Total energy dissipated: {(E_mech[0] - E_mech[-1])/(10**11):.2f} × 10¹¹ J")

# Time in seconds
t_rel = (t - t[0])

# Plot 1: Mechanical Energy vs Time
plt.figure(figsize=(10, 6))
plt.plot(t_rel, E_mech/(10**11), 'b-', linewidth=2)
plt.xlabel('Time after Entry Interface (seconds)')
plt.ylabel('Mechanical Energy (×10¹¹ J)')
plt.title('Total Mechanical Energy vs Time')
plt.grid(True, alpha=0.3)
plt.savefig('graph1_energy_vs_time.png', dpi=150)


# Plot 2: Energy Dissipation Rate vs Time
plt.figure(figsize=(10, 6))
plt.plot(t_rel, -dissipation_rate/(10**9), 'r-', linewidth=2)
plt.xlabel('Time after Entry Interface (seconds)')
plt.ylabel('Dissipation Rate (×10⁹ W)')
plt.title('Mechanical Energy Dissipation Rate vs Time')
plt.grid(True, alpha=0.3)
plt.savefig('graph2_dissipation_vs_time.png', dpi=150)

# Plot 3: Drag Power vs Time
plt.figure(figsize=(10, 6))
plt.plot(t_rel, P_drag/(10**9), 'g-', linewidth=2)
plt.xlabel('Time after Entry Interface (seconds)')
plt.ylabel('Drag Power (×10⁹ W)')
plt.title('Drag Power vs Time')
plt.grid(True, alpha=0.3)
plt.savefig('graph3_drag_power_vs_time.png', dpi=150)


# Plot 4: Correlation Scatter Plot
plt.figure(figsize=(8, 8))
plt.scatter(P_drag[valid]/(10**9), np.abs(dissipation_rate[valid])/1e9, 
            alpha=0.5, s=20, c='purple')
plt.xlabel('Drag Power (×10⁹ W)')
plt.ylabel('Power Dissipated (×10⁹ W)')
plt.title(f'Dissipation Rate vs Drag Power\nPearson r = {correlation:.3f}')
plt.grid(True, alpha=0.3)
plt.savefig('graph4_correlation.png', dpi=150)