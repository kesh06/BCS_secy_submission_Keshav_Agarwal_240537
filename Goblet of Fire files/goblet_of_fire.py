import pygame
import random
import pickle
from collections import deque
import sys
import matplotlib.pyplot as plt

# imp constants
TILE_SIZE = 40
WALL_COLOR = (0, 0, 0)
PATH_COLOR = (255, 255, 255)
A_COLOR = (255, 0, 0)
B_COLOR = (0, 0, 255)
C_COLOR = (0, 255, 0)
CHECKPOINT_COLOR = (255, 255, 0)

ACTIONS = ['up', 'down', 'left', 'right']
Q_table = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# defining maze functions
def load_maze(filename):
    with open(filename, 'r') as f:
        return [list(line.strip()) for line in f]

def get_open_positions(maze):
    positions = []
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] != 'X':
                positions.append((x, y))
    return positions

def is_valid_move(pos, maze):
    x, y = pos
    if 0 <= y < len(maze) and 0 <= x < len(maze[0]):
        return maze[y][x] != 'X'
    return False

# drwing of maze
def draw_maze(screen, maze, A_pos, B_pos, C_pos, checkpoints):
    font = pygame.font.SysFont("Arial", 12)
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if maze[y][x] == 'X':
                color = WALL_COLOR
            else:
                color = PATH_COLOR
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

            state = (x, y)
            if state in Q_table and maze[y][x] != 'X':
                best_action = max(Q_table[state], key=Q_table[state].get)
                text = font.render(best_action[0].upper(), True, (0, 0, 0))
                screen.blit(text, (x * TILE_SIZE + 12, y * TILE_SIZE + 12))

    for cp in checkpoints:
        rect = pygame.Rect(cp[0] * TILE_SIZE, cp[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, CHECKPOINT_COLOR, rect)

    pygame.draw.rect(screen, A_COLOR, pygame.Rect(A_pos[0] * TILE_SIZE, A_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    pygame.draw.rect(screen, B_COLOR, pygame.Rect(B_pos[0] * TILE_SIZE, B_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    pygame.draw.rect(screen, C_COLOR, pygame.Rect(C_pos[0] * TILE_SIZE, C_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE))

# using BFS for A for chasign B
def bfs(maze, start, goal):
    queue = deque([(start, [])])
    visited = set()

    while queue:
        current, path = queue.popleft()
        if current == goal:
            if path:
                return path[0]
            else:
                return current
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(0,-1), (0,1), (-1,0), (1,0)]:
            nx = current[0] + dx
            ny = current[1] + dy

            if is_valid_move((nx, ny), maze):
                queue.append(((nx, ny), path + [(nx, ny)]))

    return start

# using Q-learning for B
def init_state(state):
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in ACTIONS}

def choose_action(state, epsilon):
    init_state(state)
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(Q_table[state], key=Q_table[state].get)

def get_next_position(pos, action):
    x, y = pos
    if action == 'up':
        return (x, y - 1)
    elif action == 'down':
        return (x, y + 1)
    elif action == 'left':
        return (x - 1, y)
    elif action == 'right':
        return (x + 1, y)
    return pos

def get_reward(new_pos, maze, C_pos, A_pos, checkpoints):
    if not is_valid_move(new_pos, maze):
        return -10
    if new_pos == C_pos:
        return 100
    if new_pos == A_pos:
        return -100
    if new_pos in checkpoints:
        return 20
    return -1

def update_Q_table(state, action, reward, next_state):
    init_state(state)
    init_state(next_state)
    old_value = Q_table[state][action]
    future_max = max(Q_table[next_state].values())
    noise = random.uniform(-0.05, 0.05)
    Q_table[state][action] = old_value + alpha * (reward + gamma * future_max - old_value + noise)

def save_q_table():
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

def load_q_table():
    global Q_table
    try:
        with open("q_table.pkl", "rb") as f:
            Q_table = pickle.load(f)
    except:
        Q_table = {}

# loop for trainign
def train_agent(episodes, maze_file="V2.txt"):
    maze = load_maze(maze_file)
    open_tiles = get_open_positions(maze)
    success_count = 0
    rewards = []
    success_rates = []

    for ep in range(episodes):
        C_pos = random.choice(open_tiles)
        B_pos = random.choice([p for p in open_tiles if abs(p[0] - C_pos[0]) < 6])
        A_pos = max(open_tiles, key=lambda p: abs(p[0] - B_pos[0]) + abs(p[1] - B_pos[1]))
        checkpoints = random.sample(open_tiles, min(3, len(open_tiles)))

        total_reward = 0
        step = 0
        max_steps = 250
        last_B_pos = B_pos

        while step < max_steps:
            epsilon_now = max(0.01, epsilon * (0.999 ** step))
            A_pos = bfs(maze, A_pos, B_pos)

            state = B_pos
            action = choose_action(state, epsilon_now)
            new_pos = get_next_position(B_pos, action)

            if not is_valid_move(new_pos, maze):
                reward = -10
            else:
                reward = get_reward(new_pos, maze, C_pos, A_pos, checkpoints)

                old_dist = abs(B_pos[0] - C_pos[0]) + abs(B_pos[1] - C_pos[1])
                new_dist = abs(new_pos[0] - C_pos[0]) + abs(new_pos[1] - C_pos[1])

                if new_dist < old_dist:
                    reward += 10
                elif new_dist > old_dist:
                    reward -= 1

                if new_pos == last_B_pos:
                    reward -= 1

                update_Q_table(state, action, reward, new_pos)
                last_B_pos = B_pos
                B_pos = new_pos

            total_reward += reward
            step += 1

            if B_pos == C_pos:
                success_count += 1
                break
            elif A_pos == B_pos:
                break

        rewards.append(total_reward)
        success_rates.append(success_count / (ep + 1))

    save_q_table()
    print(f"Training done. Success rate: {success_count}/{episodes} ({success_count/episodes*100:.2f}%)")

    # graph for rewards and sucess rate
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Total Reward per Episode')
    plt.title("B's Total Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(success_rates, label='Success Rate')
    plt.title("B's Success Rate Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    maze = load_maze("V2.txt")
    open_tiles = get_open_positions(maze)
    C_pos = random.choice(open_tiles)
    A_pos = random.choice(open_tiles)
    B_pos = random.choice([p for p in open_tiles if abs(p[0] - C_pos[0]) < 6])
    checkpoints = random.sample(open_tiles, min(3, len(open_tiles)))

    pygame.init()
    screen = pygame.display.set_mode((len(maze[0]) * TILE_SIZE, len(maze) * TILE_SIZE))
    pygame.display.set_caption("Maze Game")
    clock = pygame.time.Clock()

    load_q_table()
    running = True
    B_path = [B_pos]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q_table()
                running = False

        A_pos = bfs(maze, A_pos, B_pos)
        state = B_pos
        action = choose_action(state, 0)
        new_pos = get_next_position(B_pos, action)

        if is_valid_move(new_pos, maze):
            B_pos = new_pos

        B_path.append(B_pos)

        if B_pos == C_pos:
            print("B reached C!")
            print("B Path:", B_path)
            running = False
        elif A_pos == B_pos:
            print("A caught B!")
            print("B Path:", B_path)
            running = False

        screen.fill((0, 0, 0))
        draw_maze(screen, maze, A_pos, B_pos, C_pos, checkpoints)
        pygame.display.flip()
        clock.tick(5)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_agent(10000)
    else:
        main()
