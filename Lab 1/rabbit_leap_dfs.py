from collections import deque

def is_goal_state(state):
    return state == "EEE_WWW"

def get_successors(state):
    successors = []
    empty_index = state.index('_')
    possible_moves = [
        (empty_index - 1, empty_index),  # Move left
        (empty_index + 1, empty_index),  # Move right
        (empty_index - 2, empty_index),  # Jump left
        (empty_index + 2, empty_index)   # Jump right
    ]
    
    for move in possible_moves:
        if 0 <= move[0] < len(state):
            new_state = list(state)
            new_state[empty_index], new_state[move[0]] = new_state[move[0]], new_state[empty_index]
            successors.append(''.join(new_state))
    
    return successors

def dfs(start_state):
    stack = [(start_state, [])]
    visited = set()
    while stack:
        (state, path) = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if is_goal_state(state):
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    return None
start_state = "WWW_EEE"
goal_state = "EEE_WWW"

solution = dfs(start_state)
if solution:
    print("Optimal solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")