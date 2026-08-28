import os
import pandas as pd
import numpy as np

# -------------------------
# Paths
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_file = os.path.join(script_dir, "input", "appliances_usage_with_metadata.parquet")

# -------------------------
# Load data
# -------------------------
data = pd.read_parquet(parquet_file)

# -------------------------
# Devices in the household
# -------------------------
device_ids = [343, 226, 293, 284]  # washing_machine, boiler, vacuum, fridge
subset = data[data['id'].isin(device_ids)].copy()

# -------------------------
# Convert timestamp to date
# -------------------------
subset['date'] = subset['timestamp'].dt.date

# -------------------------
# Daily energy consumption per device
# -------------------------
daily_energy = subset.groupby(['id', 'date'])['energy_Wh'].sum().reset_index()

# -------------------------
# Average daily consumption per device
# -------------------------
avg_daily_energy = daily_energy.groupby('id')['energy_Wh'].mean()
print("Average daily energy consumption per device (Wh/day):")
print(avg_daily_energy)

# -------------------------
# Total household daily consumption
# -------------------------
total_daily = avg_daily_energy.sum()
print(f"\nTotal daily consumption: {total_daily:.0f} Wh/day")

# -------------------------
# Battery sizing for 2 days off-grid
# -------------------------
days_offgrid = 2
DoD = 0.8  # 80% Depth of Discharge

usable_capacity_Wh = total_daily * days_offgrid
nominal_capacity_Wh = usable_capacity_Wh / DoD

print(f"\nBattery capacity for 2 days off-grid:")
print(f"Usable: {usable_capacity_Wh/1000:.2f} kWh")
print(f"Nominal: {nominal_capacity_Wh/1000:.2f} kWh")

# -------------------------
# GHI calculation (monthly average)
# -------------------------
# DNI and DHI for 4 months in Tehran, Iran (in kWh/m²/day)
DNI = np.array([3.6754, 4.0382, 3.8249, 3.7286])
DHI = np.array([1.2497, 1.53, 2.0618, 2.7377])
# Zenith angles for each month in radians (example)
zenith_deg = np.array([30, 30, 30, 30])  # can be replaced with actual values
zenith_rad = np.deg2rad(zenith_deg)

# Calculate GHI
GHI = DHI + DNI * np.cos(zenith_rad)
print(f"\nGHI for 4 months (kWh/m²/day): {GHI}")

# -------------------------
# PV sizing
# -------------------------
pv_efficiency = 0.20  # 20%

# Average daily GHI
avg_daily_GHI = np.mean(GHI)  # kWh/m²/day

# Required PV area in m²
pv_area_m2 = total_daily / 1000 / (pv_efficiency * avg_daily_GHI)  # total_daily in Wh → kWh
print(f"\nRequired PV area (m²) for household: {pv_area_m2:.2f} m²")