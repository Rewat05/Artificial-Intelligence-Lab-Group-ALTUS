from collections import deque
import heapq as hq
import random

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h  # estimated distance to goal
        self.f = g + h  # evaluation function

    def __lt__(self, other):
        return self.f < other.f

def get_successors(node):
    successors = []
    index = node.state.index(0)
    quotient = index // 3
    remainder = index % 3

    moves = []
    if quotient > 0: moves.append(-3)  # move up
    if quotient < 2: moves.append(3)   # move down
    if remainder > 0: moves.append(-1) # move left
    if remainder < 2: moves.append(1)  # move right

    for move in moves:
        new_index = index + move
        if 0 <= new_index < 9:
            new_state = list(node.state)
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            successor = Node(new_state, node, node.g + 1)
            successors.append(successor)

    return successors

def heuristic(state, goal_state):
    return sum(s != g for s, g in zip(state, goal_state))

def search_agent(start_state, goal_state):
    start_node = Node(start_state)
    start_node.h = heuristic(start_state, goal_state)
    
    frontier = []
    hq.heappush(frontier, (start_node.f, start_node))
    
    visited = set()
    nodes_explored = 0

    while frontier:
        _, node = hq.heappop(frontier)
        
        if tuple(node.state) in visited:
            continue
        
        visited.add(tuple(node.state))
        nodes_explored += 1

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored', nodes_explored)
            return path[::-1]

        for successor in get_successors(node):
            successor.h = heuristic(successor.state, goal_state)
            successor.f = successor.g + successor.h
            hq.heappush(frontier, (successor.f, successor))

    print('Total nodes explored', nodes_explored)
    return None

def generate_goal_state(start_state, num_moves):
    current_state = start_state[:]
    for _ in range(num_moves):
        successors = get_successors(Node(current_state))
        if successors:
            current_state = random.choice(successors).state
    return current_state

start_state = [1, 0, 2, 4, 3, 6, 5, 8, 7]
goal_state = generate_goal_state(start_state, 20)

print("Start state:", start_state)
print("Goal state:", goal_state)

solution = search_agent(start_state, goal_state)

if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")