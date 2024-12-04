# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

import pygad

from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from graphics_both import GraphicsBoth
from attempts.scott_dick_controller import ScottDickController

from genetic_controller import GeneticTeamController

begin = time.time()

### Kessler Stuff ###

my_test_scenario = Scenario(
    name="Test Scenario",
    ## set the asteroids location and whatnot for repeatability
    asteroid_states=[
        {"position": (200, 400), "angle": 0, "speed": 30, "size": 4},
        #  {'position': (400, 600), 'angle': 30, 'speed': 30, 'size': 4},
        #  {'position': (600, 400), 'angle': 0, 'speed': 30, 'size': 4},
        #  {'position': (200, 200), 'angle': 0, 'speed': 60, 'size': 4},
        #  {'position': (100, 700), 'angle': 0, 'speed': 20, 'size': 4},
        #  {'position': (600, 600), 'angle': 70, 'speed': 70, 'size': 4},
        #  {'position': (900, 700), 'angle': 0, 'speed': 30, 'size': 4},
        #  {'position': (900, 800), 'angle': 20, 'speed': 90, 'size': 4},
        #  {'position': (600, 800), 'angle': 0, 'speed': 30, 'size': 4},
        #  {'position': (600, 800), 'angle': 35, 'speed': 60, 'size': 4},
    ],
    ship_states=[
        {
            "position": (400, 400),
            "angle": 90,
            "lives": 3,
            "team": 1,
            "mines_remaining": 3,
        },
        {
            "position": (500, 500),
            "angle": 90,
            "lives": 3,
            "team": 2,
            "mines_remaining": 3,
        },
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)

game_settings = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
    "graphics_obj": None,
    "frequency": 30,
}


### Genetic Stuff ###
"""
these are the limits that we have for each fuzzy membership varaible.
"""
gene_space = [
    # speed
    {"low": -170, "high": -130},  # RR zmf upper value
    {"low": -140, "high": -110},  # R median
    {"low": 40, "high": 60},  # R standard dev
    {"low": -20, "high": 20},  # S median
    {"low": 45, "high": 70},  # S standard dev
    {"low": 110, "high": 135},  # F median
    {"low": 40, "high": 50},  # F standard dev
    {"low": 130, "high": 170},  # FF smf lower
    # distance
    {"low": 135, "high": 160},  # C zmf upper value
    {"low": 210, "high": 240},  # CM median
    {"low": 90, "high": 110},  # CM standard dev
    {"low": 450, "high": 550},  # M median
    {"low": 140, "high": 150},  # M standard dev
    {"low": 750, "high": 800},  # MF median
    {"low": 90, "high": 110},  # MF standard dev
    {"low": 825, "high": 875},  # F smf lower value
    # thrust
    {"low": -275, "high": -225},  # RR zmf upper value
    {"low": -275, "high": -225},  # R  median
    {"low": 90, "high": 110},  # Rstandard dev
    {"low": -20, "high": 20},  # S median
    {"low": 110, "high": 135},  # S standard dev
    {"low": 225, "high": 275},  # F median
    {"low": 90, "high": 110},  # F standard dev
    {"low": 235, "high": 250},  # FF smf lower value
]


def generate_params(solution):
    params = {
        "speed_RR": solution[0],
        "speed_R": solution[1],
        "speed_R_sig": solution[2],
        "speed_S": solution[3],
        "speed_S_sig": solution[4],
        "speed_F": solution[5],
        "speed_F_sig": solution[6],
        "speed_FF": solution[7],
        "distance_C": solution[8],
        "distance_CM": solution[9],
        "distance_CM_sig": solution[10],
        "distance_M": solution[11],
        "distance_M_sig": solution[12],
        "distance_MF": solution[13],
        "distance_MF_sig": solution[14],
        "distance_F": solution[15],
        "thrust_RR": solution[16],
        "thrust_R": solution[17],
        "thrust_R_sig": solution[18],
        "thrust_S": solution[19],
        "thrust_S_sig": solution[20],
        "thrust_F": solution[21],
        "thrust_F_sig": solution[22],
        "thrust_FF": solution[23],
    }
    return params


def fitness_function(ga_instance, solution, solution_idx):
    game = TrainerEnvironment(settings=game_settings)
    # game = KesslerGame(settings=game_settings)

    params = generate_params(solution)
    score, perf_data = game.run(
        scenario=my_test_scenario,
        controllers=[GeneticTeamController(params), ScottDickController()],
    )

    asteroids_hit = score.teams[0].asteroids_hit
    accuracy = score.teams[0].accuracy
    deaths = score.teams[0].deaths
    time = score.sim_time

    fitness = asteroids_hit + (accuracy * 100) - (deaths * 75)
    return fitness


def on_gen(ga_instance):
    print("\nGeneration : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    for idx, solution in enumerate(ga_instance.population):
        formatted_solution = [int(gene) for gene in solution]
        print(f"Solution {idx}: {formatted_solution}")


ga_instance = pygad.GA(
    num_generations=5,
    num_parents_mating=5,
    sol_per_pop=5,
    fitness_func=fitness_function,
    num_genes=len(gene_space),
    mutation_percent_genes=10,
    gene_space=gene_space,
    random_mutation_min_val=-30.0,
    random_mutation_max_val=30.0,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    # parallel_processing=['thread', 4], # didnt see a difference in times on my machine between running parallel processes on threads?
    parallel_processing=["process", 6],
    on_generation=on_gen,
)

ga_instance.run()

print("\n---------------------------------------------------------------")
solution, solution_fitness, solution_idx = ga_instance.best_solution()
formatted_solution = [int(gene) for gene in solution]
print("Best solution Generation", ga_instance.best_solution_generation)
print("Best solution:", formatted_solution)
print("Best solution fitness:", solution_fitness)

end = time.time()
print(f"TOTAL GA TIME - {int(end - begin)} seconds")

## Final Run with the Best Solution ##
print("\n---------------------------------------------------------------")
print("Running with Best Solution:")
game = KesslerGame(settings=game_settings)
params = generate_params(solution)
score, perf_data = game.run(
    scenario=my_test_scenario,
    controllers=[GeneticTeamController(params), ScottDickController()],
)

print(score.stop_reason)
print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
print("Deaths: " + str([team.deaths for team in score.teams]))
print("Accuracy: " + str([team.accuracy for team in score.teams]))
print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))

ga_instance.plot_fitness()
