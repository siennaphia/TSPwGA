# Genetic Algorithm for Travelling Salesman Problem

This repository contains a Python implementation of a Genetic Algorithm (GA) solution for the Travelling Salesman Problem (TSP). The TSP is a classic optimization problem that seeks to find the shortest possible route for a salesperson who needs to visit a list of cities once and return to the origin city.

## Code Structure and Functions

The code is organized into several components: the `City` class, the `Tour` class, helper functions, and the `GeneticAlgorithm` class.

### City Class

The `City` class represents a city in the TSP. It has a name and coordinates (x, y).

- `__init__(self, name, x, y)`: Initializes the city object with a name and its coordinates.
- `distance(self, city)`: Computes and returns the Euclidean distance between the current city and another given city.
- `__repr__(self)`: Provides a string representation of the city object when printed.

### Tour Class

The `Tour` class represents a possible solution to the TSP. It consists of a sequence of cities representing a specific route.

- `__init__(self, cities)`: Initializes the tour object with a list of cities. The tour starts and ends at the same city.
- `total_distance(self)`: Calculates and returns the total distance of the tour.
- `__repr__(self)`: Provides a string representation of the tour object when printed.

### Helper Functions

These helper functions assist in managing the operations within the genetic algorithm.

- `createRoute(cityList)`: Generates a random route using all cities in the given list.
- `initialPopulation(popSize, cityList)`: Creates an initial population of tours.
- `tournament_selection(population, tournament_size)`: Performs tournament selection to select parents for the crossover operation.
- `crossover(parent1, parent2)`: Performs a crossover operation between two parent tours to generate a child tour.
- `mutate(tour, mutation_rate)`: Applies mutation to a given tour based on the provided mutation rate.

### GeneticAlgorithm Class

The `GeneticAlgorithm` class encompasses the entire GA process.

- `__init__(self, pop_size, mutation_rate, tournament_size, improvement_threshold, max_stagnant_generations)`: Initializes the parameters for the genetic algorithm.
- `genetic_algorithm(self, cities)`: The main function to run the genetic algorithm.

## Stagnant Generations

The implementation includes a concept called "stagnant generations" to prevent the algorithm from getting stuck at a local minimum. Local minima occur when the solution cannot improve by making small, incremental changes. In such cases, the algorithm may repeatedly generate populations where no tour has a shorter distance than the best one found so far. However, larger changes might lead to better solutions.

To address this, the algorithm tracks the number of generations without improvement using the `max_stagnant_generations` parameter. If this count exceeds the threshold, the algorithm assumes that it has likely reached a local minimum. This prevents the algorithm from wasting computational resources by generating further generations that won't significantly improve the solution.

By incorporating stagnant generations, the algorithm strikes a balance between searching for better solutions and avoiding unnecessary iterations. If the obtained solution is unsatisfactory, the entire algorithm can be rerun with different parameters or initial conditions.

## What I Learned

- **Genetic Algorithm**: I learned how to implement a Genetic Algorithm in Python from scratch. This includes understanding the various stages involved in a Genetic Algorithm such as selection, crossover, and mutation.

- **Traveling Salesman Problem**: I gained a practical understanding of how the TSP can be approached using evolutionary algorithms like GA. This project helped me grasp the constraints and

 objectives of TSP better.

- **Tournament Selection**: I understood the concept of tournament selection in GAs, which involves running "tournaments" among a few individuals chosen at random from the population and selecting the best among them.

- **Crossover and Mutation**: These genetic operators are crucial in GAs. I learned how to implement and use them effectively to generate better offspring.

- **Python Programming**: This project also helped me improve my Python coding skills, particularly in working with classes, list comprehensions, and built-in functions like `min()`, `random.sample()`, etc.

- **Project Collaboration**: As this project was done in collaboration with my teammates, it provided me valuable experience in working as part of a team, distributing tasks, integrating individual components, and troubleshooting together.

## Usage

To run the TSP solver, execute the Python script:

```bash
python version2.py
```

In `version2.py`, you will be prompted to enter your preferred parameters for the Genetic Algorithm:

- Population size
- Mutation rate
- Tournament size

These parameters influence the performance and efficiency of the Genetic Algorithm.

## Dependencies

The script requires the following Python libraries:

- `numpy`
- `random`
