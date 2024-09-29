import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import copy
from collections import deque

def read_matrix_from_file(file_name):
    data_list = []

    with open(file_name, "r") as file:
        content = file.readlines()

    for line in content[5:]:
        line = line.strip()
        if line:
            try:
                data_list.append(int(line))
            except ValueError:
                print(f"Invalid line skipped: {line}")

    matrix_array = np.array(data_list)
    if matrix_array.size != 512 * 512:
        raise ValueError(f"Incorrect matrix size: {matrix_array.size}, expected 262144.")

    return matrix_array.reshape((512, 512))

def generate_patches(img):
    patch_dim = 128
    patches_map = {}
    grid = []
    patch_idx = 0

    total_patches = img.shape[0] // patch_dim

    for row in range(total_patches):
        grid_row = []
        for col in range(total_patches):
            grid_row.append(patch_idx)
            sub_patch = img[row * patch_dim : (row + 1) * patch_dim, col * patch_dim : (col + 1) * patch_dim]
            patches_map[patch_idx] = sub_patch
            patch_idx += 1
        grid.append(grid_row)

    return patches_map, grid

def rebuild_image(patches, grid_structure):
    patch_h, patch_w = patches[0].shape[:2]
    total_rows, total_cols = len(grid_structure), len(grid_structure[0])

    reconstructed = np.zeros((total_rows * patch_h, total_cols * patch_w), dtype=np.uint8)

    for row_idx, row in enumerate(grid_structure):
        for col_idx, patch_id in enumerate(row):
            reconstructed[
                row_idx * patch_h : (row_idx + 1) * patch_h,
                col_idx * patch_w : (col_idx + 1) * patch_w,
            ] = patches[patch_id]

    return reconstructed

def patch_match_score(arr1, arr2):
    return np.sum(np.abs(np.array(arr1) - np.array(arr2)))

def find_best_patch(patch_list, parent_patch, patch_data, direction):
    best_score = float('inf')
    best_patch_id = -1
    parent = np.array(parent_patch)

    for patch_id in patch_list:
        score = 0
        candidate = np.array(patch_data[patch_id])

        if direction == (0, 1):
            score = patch_match_score(parent[:, -1], candidate[:, 0])
        elif direction == (0, -1):
            score = patch_match_score(parent[:, 0], candidate[:, -1])
        elif direction == (1, 0):
            score = patch_match_score(parent[-1, :], candidate[0, :])
        elif direction == (-1, 0):
            score = patch_match_score(parent[0, :], candidate[-1, :])

        if score < best_score:
            best_score = score
            best_patch_id = patch_id

    return best_patch_id

def fill_grid_bfs(grid_layout, patch_data, available_patches):
    direction_set = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    processing_queue = deque([(0, 0)])
    visited_nodes = {(0, 0)}

    while processing_queue:
        x, y = processing_queue.popleft()

        for dx, dy in direction_set:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in visited_nodes:
                best_candidate = find_best_patch(available_patches, patch_data[grid_layout[x][y]], patch_data, (dx, dy))
                grid_layout[nx][ny] = best_candidate
                available_patches.remove(best_candidate)
                processing_queue.append((nx, ny))
                visited_nodes.add((nx, ny))

def display_image(img_data):
    plt.imshow(img_data, cmap='gray')
    plt.title('Reconstructed Image')
    plt.colorbar()
    plt.show()

def compatibility(grid, patches):
    score = 0
    rows, cols = len(grid), len(grid[0])

    for i in range(rows):
        for j in range(cols):
            neighbors = find_neighbors(i, j, grid)

            for ni, nj in neighbors:
                if nj == j + 1:
                    score += patch_match_score(patches[grid[ni][nj]][:, 0], patches[grid[i][j]][:, -1])
                if nj == j - 1:
                    score += patch_match_score(patches[grid[ni][nj]][:, -1], patches[grid[i][j]][:, 0])
                if ni == i + 1:
                    score += patch_match_score(patches[grid[ni][nj]][0, :], patches[grid[i][j]][-1, :])
                if ni == i - 1:
                    score += patch_match_score(patches[grid[ni][nj]][-1, :], patches[grid[i][j]][0, :])

    return np.sqrt(score)

def find_neighbors(i, j, grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])

    if i - 1 >= 0:
        neighbors.append((i - 1, j))
    if i + 1 < rows:
        neighbors.append((i + 1, j))
    if j - 1 >= 0:
        neighbors.append((i, j - 1))
    if j + 1 < cols:
        neighbors.append((i, j + 1))

    return neighbors

def compute_gradients(image_matrix, threshold_value=100):
    sobel_x = cv2.Sobel(image_matrix, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_matrix, cv2.CV_64F, 0, 1, ksize=3)

    gradient_sum = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    gradient_sum[gradient_sum < threshold_value] = 0
    return np.sqrt(np.sum(np.abs(gradient_sum)))

def optimize_grid(init_grid, patch_set, init_value):
    grid_copy = copy.deepcopy(init_grid)
    best_grid = copy.deepcopy(grid_copy)
    current_score = compatibility(grid_copy, patch_set)
    best_score = current_score

    temperature = 10
    min_temp = 1
    cooling_rate = 0.995

    while temperature > min_temp:
        x1, y1, x2, y2 = random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)
        grid_copy[x1][y1], grid_copy[x2][y2] = grid_copy[x2][y2], grid_copy[x1][y1]

        new_score = compatibility(grid_copy, patch_set)

        if new_score < current_score or random.random() < math.exp((current_score - new_score) / temperature):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_grid = copy.deepcopy(grid_copy)
        else:
            grid_copy[x1][y1], grid_copy[x2][y2] = grid_copy[x2][y2], grid_copy[x1][y1]

        temperature *= cooling_rate

    return best_grid, best_score

if __name__ == "__main__":
    matrix_data = read_matrix_from_file("scrambled.mat")
    matrix_data = matrix_data.T

    patch_data, initial_grid = generate_patches(matrix_data)

    final_result_grid = None
    best_found_score = float('inf')

    for init in range(16):
        grid_layout = [[-1 for _ in range(4)] for _ in range(4)]
        grid_layout[0][0] = init
        remaining_patches = list(range(16))
        remaining_patches.remove(init)

        fill_grid_bfs(grid_layout, patch_data, remaining_patches)

        candidate_grid = copy.deepcopy(grid_layout)
        candidate_image = rebuild_image(patch_data, candidate_grid)

        score = compute_gradients(candidate_image)
        optimized_grid, opt_score = optimize_grid(candidate_grid, patch_data, score)

        if opt_score < best_found_score:
            final_result_grid = optimized_grid
            best_found_score = opt_score

    final_image = rebuild_image(patch_data, final_result_grid)
    display_image(final_image)