# 🔍 Uncertainty-Based Wildfire Sensor Selection

## 📌 Project Overview
This project explores different **sensor selection strategies** for wildfire monitoring, using:
- **Greedy selection** 📍
- **Entropy-based selection** 📊
- **Gaussian Process (GP)-based selection** 📈

These algorithms **iteratively reduce uncertainty** in a wildfire-prone environment by selecting optimal observation points.

## 🎯 Objective
The goal is to **simulate and compare different strategies** for selecting sensor locations to maximize information gain. The environment consists of a **20×20 uncertainty grid**, where each step reduces uncertainty in selected regions.

---

## 🏗️ Algorithm Descriptions

### 1️⃣ Greedy Selection 🏆
- Selects the **cell with the highest uncertainty**.
- Only considers candidates within a **radius of 10** from the current position.
- Directly **sets the selected cell’s uncertainty to 0**, without affecting nearby cells.

### 2️⃣ Entropy-Based Selection 🔥
- Computes an **entropy-like measure**:
  - Higher entropy means higher uncertainty.
- Selects a high-entropy point and **reduces uncertainty in a local neighborhood**.
- Prioritizes areas with **diverse uncertainty values**.

### 3️⃣ Gaussian Process-Based Selection 📡
- Uses a **Gaussian Process (GP) model** to estimate uncertainty.
- Selects points where the **predicted variance is highest**.
- Updates the GP model iteratively as new data points are observed.
- More computationally intensive but provides **statistical confidence estimates**.

---

## 🗺️ How It Works
1. A **random starting position** is selected.
2. At each step:
   - The selected algorithm **chooses a new point** based on the selection strategy.
   - The **uncertainty field is updated** to reflect the new observation.
3. The simulation runs for **20 steps**.

📊 **Live Visualization:**
- A **grid heatmap** shows uncertainty evolution.
- **A scatter point (X) marks the selected observation**.
- The GP-based approach also displays a **variance field**.

---

## 🚀 Installation & Setup
### 📦 Dependencies
This project uses:
- **Python 3.8+**
- **NumPy** → Array manipulation
- **Matplotlib** → Visualization
- **Scikit-Learn** → Gaussian Process Regression
