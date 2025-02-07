"""
Wildfire Sensor Deployment Visualization

This script generates a visualization of target points (fire monitoring locations) 
and vantage points (potential drone positions) within a selected polygonal region.

Key Features:
- Loads a polygon (with or without obstacles) from a GeoJSON file.
- Generates target points inside the polygon using random sampling.
- Assigns fire risk weights to target points (color-coded visualization).
- Generates candidate vantage points for monitoring.
- Displays the polygon, target points, and vantage points with a legend.

Dependencies:
- numpy
- geopandas
- matplotlib
- shapely
- networkx

Usage:
Ensure `polygons.geojson` is present in the working directory.
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import networkx as nx

# Load polygon (example GeoJSON file)
polygon = gpd.read_file("./polygons.geojson").geometry[2]

# Generate random target points inside the polygon
num_targets = 50
minx, miny, maxx, maxy = polygon.bounds
targets = [Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)) 
           for _ in range(num_targets) if polygon.contains(Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)))]

# Assign fire risk weights randomly (0 = low risk, 1 = high risk)
fire_risk = np.random.uniform(0, 1, len(targets))

# Generate random vantage points
num_vantages = 10
vantages = [Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)) for _ in range(num_vantages)]

# Plot visualization
fig, ax = plt.subplots(figsize=(8, 6))
gpd.GeoSeries(polygon).plot(ax=ax, color="lightgrey", edgecolor="black", label="Polygon Boundary")

# Plot targets (color-coded by fire risk)
scatter = ax.scatter([t.x for t in targets], [t.y for t in targets], 
                      c=fire_risk, cmap="coolwarm", s=50, edgecolors="black", label="Target Points")

# Plot vantage points
ax.scatter([v.x for v in vantages], [v.y for v in vantages], 
           color="black", marker="^", s=80, label="Vantage Points")

# Add colorbar for fire risk levels
cbar = plt.colorbar(scatter, ax=ax, label="Fire Risk Level")
cbar.set_label("Fire Risk Level (0 = Low, 1 = High)")

# Add legend
ax.legend(loc="upper right")

plt.title("Target and Vantage Points with Fire Risk")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
