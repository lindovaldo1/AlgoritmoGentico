# app.py

from flask import Flask, render_template, request, jsonify
from genetic_algorithm import run_genetic_algorithm

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_genetic_algorithm', methods=['POST'])
def run_genetic_algorithm_route():
    # pop_size = int(request.form.get['popSize'])
    # chromosome_length = int(request.form['chromosomeLength'])
    # mutation_rate = float(request.form['mutationRate'])
    # crossover_rate = float(request.form['crossoverRate'])
    # num_generations = int(request.form['numGenerations'])
    # tournament_size = int(request.form['tournamentSize'])
    # elite_size = int(request.form['eliteSize'])

    # result = run_genetic_algorithm(pop_size, chromosome_length, mutation_rate, crossover_rate, num_generations, tournament_size, elite_size)

    print(request.form['popSize'])
    return jsonify("result")

if __name__ == '__main__':
    app.run(debug=True)
