# CARLA Autonomous Driving Simulation

A Python-based autonomous driving project built on the [CARLA simulator](https://carla.org/) (v0.9.14). The notebook demonstrates progressively complex driving scenarios — from spawning a single vehicle to a full self-driving agent with real-time ML steering, object detection, and traffic light awareness.

---

## Features

- **Vehicle & sensor spawning** — Spawn cars and attach RGB cameras with configurable resolution and mount offsets.
- **Route planning** — Use `GlobalRoutePlanner` to trace and visualise routes between two map locations.
- **Speed controller** — Simple throttle controller to maintain a target speed (default 20 km/h).
- **Autonomous agent** — Multi-mode driving system with:
  - Hough-transform-based lane steering
  - ML steering via a trained CNN (`DrivingNet`)
  - YOLO object detection with bounding-box overlays
  - Traffic light monitoring and threat classification (`CLEAR` / `RED_LIGHT` / `OBJ_DANGER`)
  - Stuck detection and recovery
- **Data collection & training** — Record `(image, steering_angle)` pairs while driving and retrain the model in the background.
- **NPC traffic** — Spawn up to 50 vehicles and 20 pedestrians controlled by CARLA's Traffic Manager and AI walker controllers.
- **Pygame dashboard** — Live HUD showing speed, steering, throttle, drive mode, ML confidence, and sensor feeds (front + rear cameras, heatmap, detection panel).
- **Model evaluation** — Confusion matrix and class-wise accuracy plots using scikit-learn and seaborn.

---

## Requirements

| Dependency | Purpose |
|---|---|
| CARLA 0.9.14 | Simulator & Python API |
| Python 3.8+ | Runtime |
| `numpy` | Array/image manipulation |
| `opencv-python` (`cv2`) | Camera feed display & image processing |
| `networkx` | Route graph (used internally by GlobalRoutePlanner) |
| `pygame` | Dashboard rendering & keyboard input |
| `torch` / `torchvision` | DrivingNet training & inference |
| `ultralytics` | YOLOv8 object detection |
| `scikit-learn` | Model evaluation metrics |
| `matplotlib` / `seaborn` | Evaluation plots |

Install Python dependencies:

```bash
pip install numpy opencv-python networkx pygame torch torchvision ultralytics scikit-learn matplotlib seaborn
```

---

## Setup

1. **Install CARLA 0.9.14** and start the simulator server:
   ```bash
   ./CarlaUE4.sh   # Linux
   # or CarlaUE4.exe on Windows
   ```

2. **Add the CARLA PythonAPI to your path.** The notebook uses:
   ```python
   import sys
   sys.path.append('C:\\CARLA_0.9.14\\WindowsNoEditor\\PythonAPI\\carla')
   ```
   Adjust this path to match your installation.

3. Open `CARLA.ipynb` in Jupyter and run cells in order.

---

## Notebook Structure

### 1. Basic Spawning
Connect to the CARLA server, retrieve spawn points, and place a firetruck on the map with autopilot enabled.

### 2. Map & Route Planning
Fetch the road topology, construct a `GlobalRoutePlanner` graph, and draw a route between two coordinates directly in the simulator viewport.

### 3. Camera Sensor
Attach a 640×360 RGB camera to a Mini Cooper S. A callback populates a shared dictionary with the latest frame for display via OpenCV.

### 4. Speed Control Loop
Drive straight while maintaining a target speed. The `maintain_speed()` function returns throttle values based on the gap between current and desired speed.

### 5. Full Autonomous Agent (`main()`)
The complete pipeline run in synchronous mode:

```
world.tick() → YOLO detection → traffic light check
    → drive_mode branch (MANUAL / AUTO)
        AUTO: agent.step() → Hough steer → optional ML blend
    → draw overlays → pygame dashboard render
```

**Keyboard controls during simulation:**

| Key | Action |
|---|---|
| `M` | Toggle Manual / Auto mode |
| `L` | Toggle ML steering on/off |
| `C` | Toggle data collection |
| `T` | Trigger async model training |
| `Q` | Quit & clean up |

### 6. DrivingNet Architecture
A compact end-to-end CNN (inspired by NVIDIA's PilotNet) for steering regression:

```
Input (3×H×W)
→ Conv2d(3→24, 5×5, stride=2) → ELU
→ Conv2d(24→36, 5×5, stride=2) → ELU
→ Conv2d(36→48, 5×5, stride=2) → ELU
→ Conv2d(48→64, 3×3) → ELU
→ Conv2d(64→64, 3×3) → ELU
→ Flatten → Linear(1152→100) → Dropout
→ Linear(100→50) → Dropout
→ Linear(50→10) → Linear(10→1) → Tanh × 0.5
Output: steering angle ∈ [−0.5, 0.5]
```

### 7. Model Evaluation
Loads `driving_model.pt` and `driving_data.pkl`, converts continuous steering angles to `LEFT / STRAIGHT / RIGHT` classes, and plots a confusion matrix and per-class accuracy bar chart.

---

## File Outputs

| File | Description |
|---|---|
| `driving_model.pt` | Saved DrivingNet weights |
| `driving_data.pkl` | Collected `(image, steer)` training pairs |

---

## Configuration

Key constants at the top of the main cell:

```python
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN       = "Town10HD_Opt"   # map to load
PREFERRED_SPEED  = 20         # km/h
SPEED_THRESHOLD  = 2          # tolerance band
MAX_STEER        = 0.5
HM_PANEL_W, HM_PANEL_H = ...  # heatmap panel size
```

---

## Notes

- The simulator must be running **before** executing any cell that calls `carla.Client(...)`.
- Synchronous mode (`cfg.synchronous_mode = True`) is used in the full agent loop — every frame is stepped explicitly with `world.tick()`. Do not mix with asynchronous calls.
- The cleanup block in the `finally` clause destroys all spawned actors and restores original world settings. If the kernel is killed abruptly, restart the CARLA server to clear orphaned actors.
- YOLO and PyTorch are optional; the agent degrades gracefully if they are not installed (`TORCH_OK`, `YOLO_OK` flags).
