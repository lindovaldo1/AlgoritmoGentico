import random
import math

# Definindo as variáveis e limites
x1_lower_bound, x1_upper_bound = -3.1, 12.1
x2_lower_bound, x2_upper_bound = 4.1, 5.8


# Função de aptidão a ser maximizada
def fitness_function(x, y):
    return 15 + x * math.cos(2 * math.pi * x) + y * math.cos(14 * math.pi * y)


# Função para gerar indivíduos aleatórios
def generate_individual():
    x1 = random.uniform(x1_lower_bound, x1_upper_bound)
    x2 = random.uniform(x2_lower_bound, x2_upper_bound)
    return x1, x2


# Função de crossover (recombinação)
def crossover(parent1, parent2, probability_crossover, crossover_type):
    # Verifica se o crossover deve ocorrer
    if random.random() < probability_crossover:
        if crossover_type == "one_point":
            return one_point_crossover(parent1, parent2)
        elif crossover_type == "two_point":
            return two_point_crossover(parent1, parent2)
        else:
            raise ValueError("Tipo de crossover inválido. Escolha 'one_point' ou 'two_point'.")
    else:
        # Sem crossover, retornamos os pais sem alterações
        return parent1, parent2

def one_point_crossover(parent1, parent2):
    crossover_point = random.uniform(0, 1)
    child1 = (crossover_point * parent1[0] + (1 - crossover_point) * parent2[0],
              crossover_point * parent1[1] + (1 - crossover_point) * parent2[1])
    child2 = (crossover_point * parent2[0] + (1 - crossover_point) * parent1[0],
              crossover_point * parent2[1] + (1 - crossover_point) * parent1[1])
    return child1, child2

def two_point_crossover(parent1, parent2):
    crossover_point1 = random.uniform(0, 1)
    crossover_point2 = random.uniform(0, 1)

    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1

    child1 = (
        crossover_point1 * parent1[0] + (crossover_point2 - crossover_point1) * parent2[0] + (1 - crossover_point2) * parent1[0],
        crossover_point1 * parent1[1] + (crossover_point2 - crossover_point1) * parent2[1] + (1 - crossover_point2) * parent1[1]
    )

    child2 = (
        crossover_point1 * parent2[0] + (crossover_point2 - crossover_point1) * parent1[0] + (1 - crossover_point2) * parent2[0],
        crossover_point1 * parent2[1] + (crossover_point2 - crossover_point1) * parent1[1] + (1 - crossover_point2) * parent2[1]
    )

    return child1, child2

# Função de mutação
def mutate(individual, mutation_rate):
    mutated_x1 = individual[0] + random.uniform(-mutation_rate, mutation_rate)
    mutated_x2 = individual[1] + random.uniform(-mutation_rate, mutation_rate)

    # Garantindo que os novos valores estejam dentro dos limites
    mutated_x1 = max(min(mutated_x1, x1_upper_bound), x1_lower_bound)
    mutated_x2 = max(min(mutated_x2, x2_upper_bound), x2_lower_bound)

    return mutated_x1, mutated_x2


# Seleção de pais com base no método escolhido
def select_parents(population, fitness_scores, method, tournament_size):
    if method == "roulette":
        # Seleção por roleta
        selected_parents = random.choices(population, weights=fitness_scores, k=len(population))
    elif method == "tournament":
        # Seleção por torneio
        selected_parents = []
        for _ in range(len(population)):
            tournament_candidates = random.sample(list(enumerate(fitness_scores)), tournament_size)
            tournament_winner = max(tournament_candidates, key=lambda x: x[1])[0]
            selected_parents.append(population[tournament_winner])
    else:
        raise ValueError("Método de seleção inválido. Escolha 'roulette' ou 'tournament'.")

    return selected_parents

# Algoritmo genético principal
def genetic_algorithm(population_size, generations, selection_method, tournament_size, mutation_rate, probability_crossover, elitism_rate, crossover_type):

    # Inicialização da população
    population = [generate_individual() for _ in range(population_size)]


    for generation in range(generations):
        # Avaliação da aptidão de cada indivíduo na população
        fitness_scores = [fitness_function(x, y) for x, y in population]

        # Seleção de pais com base na aptidão
        selected_parents = select_parents(population, fitness_scores, selection_method, tournament_size)

        # Aplica elitismo: mantém os melhores indivíduos sem crossover ou mutação
        elite_size = int(elitism_rate * population_size)
        elite = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_size]
        elite = [ind for ind, _ in elite]

        # Crossover e mutação para gerar o restante da população
        new_population = elite.copy()  # Inicia com os melhores indivíduos (elitismo)

        for i in range(0, population_size, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2, probability_crossover, crossover_type)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    # Retornar o melhor indivíduo da última geração
    best_individual = max(population, key=lambda ind: fitness_function(ind[0], ind[1]))
    return best_individual, fitness_function(best_individual[0], best_individual[1])


# Parâmetros do algoritmo genético
# chromossome_size
population_size = 100
probability_crossover = 0.8
mutation_rate = 0.1
generations = 50
selection_method = "roulette"  # Alternativas: "roulette" ou "tournament"
tournament_size = 5  # Usado apenas se o método de seleção for "tournament"
elitism_rate = 0.2
crossover_type = "two_point"  #one_point ou "two_point"

# Executar o algoritmo genético
best_solution, best_fitness = genetic_algorithm(population_size, generations,
                                                selection_method, tournament_size,
                                                mutation_rate, probability_crossover,
                                                elitism_rate, crossover_type)

# Exibir os resultados
print("Melhor solução:", best_solution)
print("Melhor valor de aptidão:", best_fitness)
