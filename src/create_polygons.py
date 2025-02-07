import geopandas as gpd
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# Define Convex Polygon (Hexagon)
convex_polygon = Polygon([
    (0, 0), (2, 0), (3, 1), (2, 2), (0, 2), (-1, 1), (0, 0)
])

# Define Concave Polygon (Notch)
concave_polygon = Polygon([
    (0, 0), (4, 0), (4, 2), (2, 1.5), (2, 3), (0, 3), (0, 0)
])

# Define Polygon with a Hole (Obstacle inside)
outer = Polygon([
    (0, 0), (5, 0), (5, 5), (0, 5), (0, 0)
])
hole = Polygon([
    (2, 2), (3, 2), (3, 3), (2, 3), (2, 2)
])
polygon_with_hole = Polygon(outer.exterior.coords, holes=[hole.exterior.coords])

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    {'geometry': [convex_polygon, concave_polygon, polygon_with_hole]},
    index=["convex", "concave", "with_hole"]
)

# Save as GeoJSON
gdf.to_file("polygons.geojson", driver="GeoJSON")

# Print GeoJSON Content
print(json.dumps(json.loads(gdf.to_json()), indent=2))

# Plot polygons
fig, ax = plt.subplots(figsize=(6, 6))
gdf.plot(ax=ax, edgecolor='black', alpha=0.5, cmap="Set1")
plt.title("Generated Polygons")
plt.show()
