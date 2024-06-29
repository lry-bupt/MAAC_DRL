import numpy as np
import heapq
from scipy.io import loadmat
import math

x_max=99
y_max=99
m = loadmat("C://Users/Administrator.DESKTOP-NLH290A/Desktop/2302_code/FIGURE_3/mapdata_0717.mat") 
#correct_action=0
MARK= m["MARK_new"]

def generate_directions(num_directions):
    # 生成均匀分布的方向向量
    directions = []
    angle_step = 360 / num_directions
    for i in range(num_directions):
        angle = math.radians(i * angle_step)
        directions.append((math.cos(angle), math.sin(angle)))
    return directions

def is_valid(x, y, MARK, x_max, y_max):
    # Check if (x, y) is within bounds and not an obstacle
    return 0 <= int(x) < x_max and 0 <= int(y) < y_max and MARK[int(x), int(y)] != 2

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, MARK, x_max, y_max):
    # A* search to find the shortest path from start to goal
    neighbors = [ (0, -1), (-1, 0),(1, 0), (0, 1)]
    # neighbors = generate_directions(8)
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data.reverse()
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1

            if not is_valid(neighbor[0], neighbor[1], MARK, x_max, y_max):
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def save_path_to_txt(path, filename):
    with open(filename, 'w') as f:
        for x, y in path:
            f.write(f"{x},{y}\n")

# Define start, goal, and MARK matrix
x1, y1 = 40, 40  # Starting point
x2, y2 = 90, 90  # Goal point
# x_max, y_max = 10, 10  # Grid size

# Example MARK matrix with obstacles
# MARK = np.zeros((x_max, y_max))
# MARK[4, 4] = 2
# MARK[4, 5] = 2
# MARK[4, 6] = 2
# MARK[5, 4] = 2
# MARK[6, 4] = 2

start = (x1, y1)
goal = (x2, y2)

# Find path
path = a_star_search(start, goal, MARK, x_max, y_max)

# Save path to txt file
if path:
    save_path_to_txt(path, 'robot_path_2_v2.txt')
    print("Path found and saved to robot_path.txt")
else:
    print("No path found")
