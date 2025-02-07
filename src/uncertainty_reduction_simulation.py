import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ----------------------- Configuration -----------------------
GRID_SIZE = (20, 20)
NUM_STEPS = 20
NEIGHBORHOOD = 3       # Radius for uncertainty reduction (used by all algorithms)
REDUCTION_FACTOR = 0.7  # Multiplicative reduction factor
GREEDY_RADIUS = 10     # Radius within which the greedy algorithm chooses its next cell

# ----------------------- Global Variables -----------------------
selected_algorithm = "Greedy"  # Default algorithm selection
step_index = 0                 # Current time step
cbar = None                    # Global colorbar handle (will be created once)

# Randomize starting position (for all algorithms)
np.random.seed()  # Or use a fixed seed, e.g. np.random.seed(42)
initial_position = tuple(np.random.randint(0, GRID_SIZE[i]) for i in range(2))

# Initialize uncertainty fields for each algorithm
uncertainty_greedy = np.ones(GRID_SIZE, dtype=float)
uncertainty_entropy = np.ones(GRID_SIZE, dtype=float)
uncertainty_gp = np.ones(GRID_SIZE, dtype=float)

# Storage for results: list of tuples (selected_point, uncertainty_field)
greedy_results = []
entropy_results = []
gp_results = []

# For GP-based selection: training points (start with the same initial position)
gp_selected_points = [initial_position]

# ----------------------- Helper Functions -----------------------
def reduce_neighborhood(uncertainty, point, neighborhood=NEIGHBORHOOD, factor=REDUCTION_FACTOR):
    """
    Reduces uncertainty in a square neighborhood around the given point.
    """
    x, y = point
    updated = uncertainty.copy()
    x_min = max(0, x - neighborhood)
    x_max = min(updated.shape[0], x + neighborhood + 1)
    y_min = max(0, y - neighborhood)
    y_max = min(updated.shape[1], y + neighborhood + 1)
    updated[x_min:x_max, y_min:y_max] *= factor
    return updated

# ----------------------- Selection Functions -----------------------
def greedy_selection(uncertainty, current_position, radius=GREEDY_RADIUS):
    """
    Greedy selection (modified):
      - Restricts candidate cells to those within a radius (default=10) of the current position.
      - Among those, finds the cells with the maximum uncertainty and then randomly chooses one.
      - Sets the uncertainty at the chosen cell to 0.
    
    Parameters:
        uncertainty (np.ndarray): The current uncertainty field.
        current_position (tuple): (row, col) of the current position.
        radius (int): The search radius for candidate cells.
    
    Returns:
        tuple: (selected_point, updated_uncertainty)
    """
    x0, y0 = current_position
    # Get all indices (neighbors) within the specified Euclidean radius.
    neighbors = [
        (i, j)
        for i in range(uncertainty.shape[0])
        for j in range(uncertainty.shape[1])
        if np.sqrt((i - x0) ** 2 + (j - y0) ** 2) <= radius
    ]
    
    # Get the uncertainty values for the candidate cells.
    neighbor_uncertainties = [uncertainty[i, j] for (i, j) in neighbors]
    max_val = max(neighbor_uncertainties)
    # Identify candidates that achieve the maximum uncertainty.
    best_candidates = [pt for pt in neighbors if uncertainty[pt[0], pt[1]] == max_val]
    
    # Randomly choose one among the best candidates.
    selected_index = best_candidates[np.random.choice(len(best_candidates))]
    updated = uncertainty.copy()
    updated[selected_index] = 0.0  # Set the chosen cell's uncertainty to 0.
    return selected_index, updated

def entropy_based_selection(uncertainty):
    """
    Entropy-based selection: choose a point with high entropy and reduce uncertainty around it.
    """
    updated = uncertainty.copy()
    # Compute an entropy-like measure (adding a tiny value to avoid log(0))
    entropy_field = -updated * np.log(updated + 1e-9)
    max_entropy = np.max(entropy_field)
    candidates = np.argwhere(entropy_field >= max_entropy * 0.9)
    if candidates.size > 0:
        chosen_index = tuple(candidates[np.random.choice(len(candidates))])
    else:
        chosen_index = initial_position  # Fallback if no candidate is found
    updated = reduce_neighborhood(updated, chosen_index)
    return chosen_index, updated

def gp_based_selection(uncertainty, selected_points):
    """
    GP-based selection: use a Gaussian Process to model uncertainty and sample a point
    based on the predicted standard deviation. Uncertainty is then reduced around that point.
    """
    updated = uncertainty.copy()
    # Prepare grid points (each cell coordinate)
    grid_points = np.array([(i, j) for i in range(updated.shape[0])
                                   for j in range(updated.shape[1])])
    kernel = C(1.0) * RBF(length_scale=3.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    if selected_points:
        X_train = np.array(selected_points)
        y_train = np.array([uncertainty[x, y] for x, y in selected_points])
        gp.fit(X_train, y_train)
        # Predict standard deviation as a measure of uncertainty
        _, std_pred = gp.predict(grid_points, return_std=True)
        std_field = std_pred.reshape(updated.shape)
    else:
        std_field = updated.copy()
    total_std = std_field.sum()
    if total_std > 0:
        prob_distribution = std_field / total_std
        flat_index = np.random.choice(np.arange(std_field.size), p=prob_distribution.flatten())
    else:
        flat_index = np.random.choice(np.arange(std_field.size))
    selected_index = np.unravel_index(flat_index, std_field.shape)
    updated = reduce_neighborhood(updated, selected_index)
    return selected_index, updated, std_field

# ----------------------- Precompute Results -----------------------
# For all three algorithms, force the same starting position.
# Set the initial cell's uncertainty to 0 and record the starting point.

# Greedy algorithm:
uncertainty_greedy[initial_position] = 0.0
greedy_results.append((initial_position, uncertainty_greedy.copy()))
current_position = initial_position
for _ in range(NUM_STEPS - 1):
    g_point, uncertainty_greedy = greedy_selection(uncertainty_greedy, current_position, radius=GREEDY_RADIUS)
    greedy_results.append((g_point, uncertainty_greedy.copy()))
    current_position = g_point  # Update current position for the next step

# Entropy-based algorithm:
uncertainty_entropy[initial_position] = 0.0
entropy_results.append((initial_position, uncertainty_entropy.copy()))
for _ in range(NUM_STEPS - 1):
    e_point, uncertainty_entropy = entropy_based_selection(uncertainty_entropy)
    entropy_results.append((e_point, uncertainty_entropy.copy()))

# GP-based algorithm:
uncertainty_gp[initial_position] = 0.0
gp_results.append((initial_position, uncertainty_gp.copy()))
for _ in range(NUM_STEPS - 1):
    gp_point, uncertainty_gp, std_field = gp_based_selection(uncertainty_gp, gp_selected_points)
    gp_selected_points.append(gp_point)
    gp_results.append((gp_point, uncertainty_gp.copy()))
    # Note: The GP predicted std field (std_field) is computed but not displayed.

# ----------------------- Plotting -----------------------
# Create the main axes for the uncertainty map.
fig, ax = plt.subplots(figsize=(8, 6))
# Adjust the subplot so that there is room on the right for the colorbar.
plt.subplots_adjust(bottom=0.3, right=0.85)
# Create a dedicated axes for the colorbar inside the figure.
cb_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])

def update_plot():
    """Update the plot for the current algorithm and step index."""
    global cbar
    ax.clear()
    
    if selected_algorithm == "Greedy":
        point, uncertainty_map = greedy_results[step_index]
    elif selected_algorithm == "Entropy-Based":
        point, uncertainty_map = entropy_results[step_index]
    elif selected_algorithm == "Gaussian Process":
        point, uncertainty_map = gp_results[step_index]
    else:
        point, uncertainty_map = greedy_results[step_index]
    
    # Display the uncertainty map using a colormap where 0 is white and 1 is black.
    # The fixed range is enforced by vmin=0 and vmax=1.
    im = ax.imshow(uncertainty_map, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    # Mark the selected point with a filled red circle.
    ax.scatter(point[1], point[0], color="red", marker="o", s=100)
    ax.set_title(f"{selected_algorithm} Selection - Step {step_index + 1}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Create or update the colorbar in the dedicated axes.
    if cbar is None:
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.set_label("Uncertainty")
    else:
        cbar.update_normal(im)
        cbar.set_label("Uncertainty")
    
    fig.canvas.draw_idle()

def next_step(event):
    global step_index
    if step_index < NUM_STEPS - 1:
        step_index += 1
        update_plot()

def prev_step(event):
    global step_index
    if step_index > 0:
        step_index -= 1
        update_plot()

def change_algorithm(label):
    global selected_algorithm
    selected_algorithm = label
    update_plot()

# Set up buttons and radio controls.
ax_prev = plt.axes([0.1, 0.05, 0.2, 0.075])
ax_next = plt.axes([0.7, 0.05, 0.2, 0.075])
btn_prev = Button(ax_prev, "Previous")
btn_next = Button(ax_next, "Next")
btn_prev.on_clicked(prev_step)
btn_next.on_clicked(next_step)

ax_radio = plt.axes([0.3, 0.05, 0.3, 0.15])
radio = RadioButtons(ax_radio, ["Greedy", "Entropy-Based", "Gaussian Process"])
radio.on_clicked(change_algorithm)

update_plot()
plt.show()
