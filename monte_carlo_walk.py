#==============================================================================
# Monte carlo simulation of random walk in discrete space
# This way taxi cabs are moving across Manhattan.
#
# In this simulation two sets of random walkers are trying to move
# across the grid with Manhattan distance metric.
# One group is moving down right, another up right.
# What emerges is quite interesting: If you modulate the random walk, the
# wave mechanics emerges along with interference pattaern in the middle,
# where two waves of walkers meet.
#
# But wait, there is more! If you try to measure a walker before all paths
# are completed, the interferece pattern gets destroyed!
# This 70 something line code reproduces the Quantum Measurement Problem!
# Change MEASURE to True to observe it.
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt

MEASURE = False

def monte_carlo_walk(num_walkers, steps_per_walker, grid_size=100):
    """
    Performs a Monte Carlo simulation of random walkers on a grid.

    Args:
        num_walkers (int): The total number of walkers to simulate.
        steps_per_walker (int): The number of steps each walker takes.
        grid_size (int): The dimension of the square grid (e.g., 100 for 100x100).
    """
    current_weight = 10.0 # Start with a positive weight/amplitude
    modulation_period = 5 # Example: Flip the sign every n steps

    # Initialize a 100x100 grid of zeros to count visits
    # Grid coordinates range from -50 to 49 for x and y
    visits = np.zeros((grid_size, grid_size), dtype=int)
    min_coord = 0 
    max_coord = grid_size  - 1

    for i in range(num_walkers):
        # Alternate between the two starting positions
        if i % 2 == 0:
            x, y = min_coord, min_coord
        else:
            x, y = min_coord, max_coord

        for step in range(steps_per_walker):
            step += 1
            # Check if within bounds before recording visit and moving
            if min_coord <= x <= max_coord and min_coord <= y <= max_coord:
                visits[y, x] += current_weight # Index as [y, x] for correct orientation

            if (step) % modulation_period == 0:
                current_weight *= -1
            # Take a random step:
            # Determine movement based on starting set
            if i % 2 == 0:
                # Set 1: increase x or y
                if np.random.rand() < 0.5:
                    x += 1
                else:
                    y += 1
            else:
                # Set 2: increase x or decrease y
                if np.random.rand() < 0.5:
                    x += 1
                else:
                    y -= 1
                    
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Optimization
            # This is something quite interesting:
            # If MEASURE is set to True, the interference pattern created by 
            # this simulation gets completely destroyed !!!
            # While simply trying to optimize this code, I stumbled upon
            # the famous "Measurment problem" from Quantum Mechanics
            # If you end a single walk based on the knowlwdge of where it is,
            # it is not wave anymore. It becomes a random particle.
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if(MEASURE and x > max_coord):
                break;

    # Visualization using matplotlib.imshow()
    plt.figure(figsize=(8, 8))
    # 'origin="lower"' places (0,0) at the bottom left
    # 'extent' sets the real-world coordinates for the axes
    plt.imshow(visits, cmap='viridis', origin='lower', extent=[min_coord, max_coord + 1, min_coord, max_coord + 1])
    plt.colorbar(label='Number of Visits')
    plt.title(f'Monte Carlo Simulation of {num_walkers} Random Walkers')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# Example Usage:
# Simulating 10,000 walkers, each taking 200 steps
monte_carlo_walk(num_walkers=1000, steps_per_walker=500, grid_size=200)
