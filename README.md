# BCS_secy_submission_Keshav_Agarwal
# Goblet of Fire
# A = Death Eater
# B= Harry Potter
# C= The Cup
# 🧠 Maze RL Simulation – B Escapes A and Reaches C

In this project, I implemented a Reinforcement Learning (Q-learning) agent `B` that learns to escape a chaser `A` and reach the goal `C` in the given grid-based maze. I have rendered the environment using `pygame`, and training progress is visualized using `matplotlib`.

---

## ✅ Basic Approach

- **Environment**: The maze is already pre-defined in a text file (`V2.txt`) where:
  - `'X'` represents walls
  - `' '` (space) represents free paths

- **My Agents**:
  - `C`: Static target. ( The Cup ) Its position is constant during each episode.
  - `B`: Learns to reach `C` using **Q-learning** and avoids `A`.
  - `A`: Always chases `B` using **Breadth-First Search (BFS)**.
    
- **I have drawn the maze and then initialized random positions for A,B,C**

- **Movement**: All agents can move in the directions `up`, `down`, `left`, `right`—only if there is no wall.

- ** Defining Q-Learning for B**:
  - **States**: Grid positions
  - **Actions**: 4 directions
  - **Rewards**:
    - `+100` for reaching C
    - `-100` for getting caught by A
    - `+10` for moving closer to C
    - `-1 to -10` for poor moves (walls, backtracking, etc.)
  -**also using noise**

- **My logic for training B**:
  - B explores randomly at first (epsilon-greedy), then exploits learned policy to reach C
  - A starts far from B to give it a fair chance
  - Successful runs are counted to compute success rate

-**Then I have plotted graphs of success rates using Matplot**
---

## ⚙️ Assumptions Made

- The maze file `V2.txt` contains only `'X'` for walls and spaces for paths.
- B and C are spawned randomly anywhere in the maze (but C's position stays fixed during the episode).
- A is always spawned kinda farther from B so that B gains some time to learn.
- BFS is used for A to keep its behavior consistent and predictable.

-** Important to type `python goblet_of_fire.py train` in the terminal to see the plots and the success rate.

---

## ▶️ How to Run

###  1. For installing required libraries

```bash
pip install pygame matplotlib
```
### ✅ 2. For training the Agent ( important )

```bash
python your_script.py train

train_agent(1000)
```
### ✅ 3. To Run the Simulation (After Training)

```bash
python your_script.py
```
