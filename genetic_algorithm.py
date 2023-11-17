from flask import Flask, render_template, request, jsonify
import numpy as np

# Função de fitness
def fitness(x, y):
    return 15 + x * np.cos(2 * np.pi * x) + y * np.cos(14 * np.pi * y)

# Inicialização da população
def initialize_population(pop_size, chromosome_length):
    return np.random.rand(pop_size, chromosome_length)

# Avaliação da população
def evaluate_population(population):
    return np.array([fitness(x, y) for x, y in population])

# Seleção por roleta
def roulette_selection(fitness_values):
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(len(fitness_values), size=len(fitness_values), p=probabilities)
    return selected_indices

# Seleção por torneio
def tournament_selection(fitness_values, tournament_size):
    selected_indices = []
    for _ in range(len(fitness_values)):
        competitors = np.random.choice(len(fitness_values), size=tournament_size)
        winner = np.argmax(fitness_values[competitors])
        selected_indices.append(competitors[winner])
    return selected_indices

# Cruzamento de um ponto
def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Cruzamento de dois pontos
def two_point_crossover(parent1, parent2):
    crossover_points = np.sort(np.random.choice(len(parent1), size=2, replace=False))
    child1 = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]))
    child2 = np.concatenate((parent2[:crossover_points[0]], parent1[crossover_points[0]:crossover_points[1]], parent2[crossover_points[1]:]))
    return child1, child2

# Mutação
def mutate(child, mutation_rate):
    mutation_mask = (np.random.rand(len(child)) < mutation_rate).astype(int)
    mutation_values = np.random.rand(len(child))
    child = child + mutation_mask * mutation_values
    return child

# Elitismo
def elitism(population, fitness_values, elite_size):
    elite_indices = np.argsort(fitness_values)[-elite_size:]
    elite_population = population[elite_indices]
    return elite_population

# Parâmetros do algoritmo genético
pop_size = 50 # Tamanho da populacao
chromosome_length = 2 #Tamanho do cromossomo
mutation_rate = 0.1 # Probabilidade de mutação
crossover_rate = 0.8 # Probabilidade de cruzamento
num_generations = 100 # Quantidade de gerações
tournament_size = 5 # Tamanho do torneio
elite_size = 1 #Quantidade de membros da elite

def run_genetic_algorithm(pop_size, chromosome_length, mutation_rate, crossover_rate, num_generations, tournament_size, elite_size):

    # Inicialização da população
    population = initialize_population(pop_size, chromosome_length)

    for generation in range(num_generations):
        # Avaliação da população
        fitness_values = evaluate_population(population)

        # Seleção
        selected_indices = tournament_selection(fitness_values, tournament_size)

        # Cruzamento
        for i in range(0, len(selected_indices), 2):
            if np.random.rand() < crossover_rate:
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[i + 1]]
                child1, child2 = one_point_crossover(parent1, parent2)
                population[selected_indices[i]] = child1
                population[selected_indices[i + 1]] = child2

        # Mutação
        for i in range(len(population)):
            population[i] = mutate(population[i], mutation_rate)

        # Elitismo
        elite_population = elitism(population, fitness_values, elite_size)

        # Substituição da população
        population[:-elite_size] = population[selected_indices[:-elite_size]]
        population[-elite_size:] = elite_population

    # Resultado final
    best_solution_index = np.argmax(fitness_values)
    best_solution = population[best_solution_index]
    best_fitness = fitness_values[best_solution_index]

    print("Melhor solução:", best_solution)
    print("Melhor valor de fitness:", best_fitness)
    return {'bestSolutin': best_solution.tolist(), 'bestFitness': best_fitness}