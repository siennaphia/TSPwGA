# Team members: Susanne Santos Erenst, Sienna Gaita-Monjaraz, Maximo Mejia
# CAP4630 - Intro to AI
# Project 2 - TSP using GA
# Due: 06/21/2023
#https://colab.research.google.com/drive/1yzKkKLjPIuQtf5F7QStqbU86HzKAtB6Z?usp=sharin

import numpy as np
import random
#import operator

# Class to represent cities with a name and x, y coordinates for positioning
class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    # Compute the distance between the current city and a given city
    def distance(self, city):
        return ((self.x - city.x)**2 + (self.y - city.y)**2)**0.5

    def __repr__(self):
        return self.name

# Class for a tour made up of multiple cities
class Tour:
    def __init__(self, cities):
        # The tour starts and ends at the same city
        self.cities = cities + [cities[0]]

    # Compute the total distance of the tour
    def total_distance(self):
        return sum(self.cities[i].distance(self.cities[i + 1]) for i in range(len(self.cities) - 1))

    def __repr__(self):
        return ' -> '.join(city.name for city in self.cities)


# Function to create a random route using all cities
def createRoute(cityList):
    route = random.sample(cityList,len(cityList))
    return route

# Function to initialize a population of tours
def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(Tour(createRoute(cityList)))
    return population

# Function to perform tournament selection
def tournament_selection(population, tournament_size):
    # Select a random subset of individuals for the tournament
    tournament = random.sample(population, tournament_size)
    # Select the best individual from the tournament
    best_individual = min(tournament, key=lambda tour: tour.total_distance())
    return best_individual

def crossover(parent1, parent2):
    subset_start, subset_end = sorted(random.sample(range(1, len(parent1.cities) - 1), 2))
    child_cities = [parent1.cities[0]] + parent1.cities[subset_start:subset_end]
    for city in parent2.cities:
        if city not in child_cities and city != parent1.cities[0]:
            child_cities.append(city)
    return Tour(child_cities)  # Remove the duplicate city at the end

# Mutates tour by swapping two cities
def mutate(tour, mutation_rate):
    if random.random() < mutation_rate:
        tour_cities = tour.cities[1:-1]  # exclude first and last city
        i, j = random.sample(range(len(tour_cities)), 2)
        tour_cities[i], tour_cities[j] = tour_cities[j], tour_cities[i]
        # rebuild the tour with the mutated cities
        tour.cities = [tour.cities[0]] + tour_cities + [tour.cities[0]]

# Define a class for the genetic algorithm
class GeneticAlgorithm:
    def __init__(self, pop_size=100, mutation_rate=0.01, tournament_size=20, improvement_threshold=0.01, max_stagnant_generations=100):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.improvement_threshold = improvement_threshold
        self.max_stagnant_generations = max_stagnant_generations

    # Main function to run the genetic algorithm
    def genetic_algorithm(self, cities):
        # Create an initial population
        population = initialPopulation(self.pop_size, cities)
        #Epoch counter
        e = 1
        # Track the number of generations with no improvement
        stagnant_generations = 0
        # Track the best tour found so far
        best_tour = min(population, key=lambda tour: tour.total_distance())
        # Tracking the fitness of the best tour found so far
        last_best_fitness = best_tour.total_distance()

        #Running the genetic algorithm until the max number of stagnant generations is reached which is set in the function call
        while stagnant_generations < self.max_stagnant_generations:
            new_population = []
            for _ in range(self.pop_size):
                #Picking two parents using tournament selection
                parent1 = tournament_selection(population, self.tournament_size)
                parent2 = tournament_selection(population, self.tournament_size)

                #Omg genetic algorithm calls!!
                child = crossover(parent1, parent2)
                mutate(child, self.mutation_rate)
                new_population.append(child)

            # Replace the old population with the new population
            population = new_population
            # Determine the fitness of the best individual in the new population
            current_best_tour = min(population, key=lambda tour: tour.total_distance())
            current_best_fitness = current_best_tour.total_distance()

            # Check if  improvement in the best fitness
            if last_best_fitness - current_best_fitness > self.improvement_threshold:
                # If improved update the best tour and reset the stagnant generations count
                best_tour = current_best_tour
                last_best_fitness = current_best_fitness
                stagnant_generations = 0
            else:
                # If there hasn't been an improvement, increment the stagnant generations count
                stagnant_generations += 1
            print("Epoch " + str(e) + "| " + "Minimum Total Distance: " + str(current_best_tour.total_distance()))
            e += 1
        # Return the best tour found
        return best_tour

#list of city names
city_names = [
    "Wintefell", "King's Landing", "Braavos", "Pentos", "Riverrun", "Dorne", "Highgarden",
    "Lannisport", "The Vale", "The Eyrie", "Mareen", "Volantis", "Qarth", "Ashaai", "Pyke",
    "Oldtown", "Storm's End", "Gulltown", "White Harbor", "Qohor", "Lys", "Lorath",
    "Stormlands", "Essos", "Tyrosh"
]
#list of cities with random coordinates
city_list = [City(name, random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0)) for name in city_names]


print("----------------------------------------------------------------")
print("Parameters for the genetic algorithm")
print("----------------------------------------------------------------")

# Print the names of the cities in the best tour and its total distance
print("Number of cities in the tour:", len(city_names))

# Prompt for population size
pop_size = int(input("Enter the integer population size (default = 100): "))

# Prompt for mutation rate
mutation_rate = float(input("Enter the float mutation rate (default = 0.01): "))

# Prompt for tournament size
tournament_size = int(input("Enter the proportion of new children in each generation (default = 20): "))

# Print the number of stagnant generations
print("Number of stagnant generations: 100")
print("\n")

# Initialize the genetic algorithm with the desired parameters
ga = GeneticAlgorithm(pop_size, mutation_rate, tournament_size, improvement_threshold=0.01, max_stagnant_generations=100)
# Run the genetic algorithm and get the best tour
best_tour = ga.genetic_algorithm(city_list)

print("\nBest Tour:")
i = 1
for city in best_tour.cities:
    print(str(i) + ". " + city.name)
    i += 1
print("Total distance:", best_tour.total_distance())
