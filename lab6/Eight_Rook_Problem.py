import numpy as np
import matplotlib.pyplot as plt

# We generate initial state randomly.


def generate_initial_state(initial_state):
  unique=[]
  i=0
  while(i<8):
    pos = np.random.randint(low=0, high=64)
    if pos not in unique:
      unique.append(pos)
      i+=1
    else:
      continue


  for x in unique:
    initial_state[x//8][x%8] = 1


# Calculate the energy for state.
def energy(x):
  E1=0
  E2=0
  for j in range(8):
    sum=0
    for i in range(8):
      sum+=x[i][j]
    sum-=1
    E1+=sum**2


  for i in range(8):
    sum=0
    for j in range(8):
      sum+=x[i][j]
    sum-=1
    E2+=sum**2
  return E1+E2


# We try to minimize the E1+E2 by flipping the position where rook is present in current state and another position where rook is not present.
def flip(curr_energy, x):
    while True:
        pos1 = np.random.randint(low=0, high=64)
        pos2 = np.random.randint(low=0, high=64)
        if x[pos1 // 8][pos1 % 8] != 1 and x[pos2 // 8][pos2 % 8] != 0:
            continue
        if x[pos1 // 8][pos1 % 8] == 1 and x[pos2 // 8][pos2 % 8] == 0:
            x[pos1 // 8][pos1 % 8] = 0
            x[pos2 // 8][pos2 % 8] = 1
            new_energy = energy(x)
            if curr_energy > new_energy:  # Check if new energy is lower
                curr_energy = new_energy  # Update current energy
            else:  # Revert the changes if energy doesn't decrease
                x[pos1 // 8][pos1 % 8] = 1
                x[pos2 // 8][pos2 % 8] = 0
            break
    return curr_energy
initial_state = np.zeros((8,8))
generate_initial_state(initial_state)


plt.imshow(initial_state, cmap='binary', interpolation='nearest')

curr_energy = energy(initial_state)
print("Energy of initial state: ",curr_energy)


iterations = 1000
x = initial_state
for i in range(iterations):
    curr_energy = flip(curr_energy, x)
    # print("Energy in " ,i, "iteration: ",curr_energy)
plt.imshow(initial_state, cmap='binary', interpolation='nearest')