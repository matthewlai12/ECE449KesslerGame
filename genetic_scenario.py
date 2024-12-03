# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

import pygad

from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from graphics_both import GraphicsBoth

from genetic_controller import GeneticTeamController

begin = time.time() 

### Kessler Stuff ###

my_test_scenario = Scenario(name='Test Scenario',
                            ## set the asteroids location and whatnot for repeatability
                            asteroid_states=[{'position': (200, 400), 'angle': 0, 'speed': 30, 'size': 4},
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
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}


### Genetic Stuff ###

# the values we submitted for the competition
# function_inputs = [[-150, -125, 50, 0, 60, 125, 50, 150, 
#                    150, 225, 100, 500, 150, 775, 100, 850,
#                    -250, -250, 100, 0, 125, 250, 100, 250]]

'''
these are the limits that we have for each fuzzy membership varaible.
'''
gene_space = [
    # speed
    {'low': -170, 'high': -130},    #RR zmf upper value
    {'low': -140, 'high': -110},    # R median
    {'low': 40, 'high': 60},        # R standard dev
    {'low': -20, 'high': 20},       # S median
    {'low': 45, 'high': 70},        # S standard dev
    {'low': 110, 'high': 135},      # F median
    {'low': 40, 'high': 50},        # F standard dev
    {'low': 130, 'high': 170},      # FF smf lower 

    # distance
    {'low': 135, 'high': 160},      # C zmf upper value
    {'low': 210, 'high': 240},      # CM median
    {'low': 90, 'high': 110},       # CM standard dev
    {'low': 450, 'high': 550},      # M median
    {'low': 140, 'high': 150},      # M standard dev
    {'low': 750, 'high': 800},      # MF median
    {'low': 90, 'high': 110},       # MF standard dev
    {'low': 825, 'high': 875},      # F smf lower value

    # thrust
    {'low': -275, 'high': -225},    # RR zmf upper value
    {'low': -275, 'high': -225},    # R  median
    {'low': 90, 'high': 110},       # Rstandard dev
    {'low': -20, 'high': 20},       # S median
    {'low': 110, 'high': 135},      # S standard dev
    {'low': 225, 'high': 275},      # F median
    {'low': 90, 'high': 110},       # F standard dev
    {'low': 235, 'high': 250},       # FF smf lower value
]

'''
pygad defaults fitness function to maximization.
'''
def fitness_function(ga_instance, solution, solution_idx):
    params = {
        'speed_RR': solution[0],
        'speed_R': solution[1],
        'speed_R_sig': solution[2],
        'speed_S': solution[3],
        'speed_S_sig': solution[4],
        'speed_F': solution[5],
        'speed_F_sig': solution[6],
        'speed_FF': solution[7],
        'distance_C': solution[8],
        'distance_CM': solution[9],
        'distance_CM_sig': solution[10],
        'distance_M': solution[11],
        'distance_M_sig': solution[12],
        'distance_MF': solution[13],
        'distance_MF_sig': solution[14],
        'distance_F': solution[15],
        'thrust_RR': solution[16],
        'thrust_R': solution[17],
        'thrust_R_sig': solution[18],
        'thrust_S': solution[19],
        'thrust_S_sig': solution[20],
        'thrust_F': solution[21],
        'thrust_F_sig': solution[22],
        'thrust_FF': solution[23]
    }
    
    #game = TrainerEnvironment(settings=game_settings)
    game = KesslerGame(settings=game_settings)
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[GeneticTeamController(params)])

    asteroids_hit = score.teams[0].asteroids_hit
    accuracy = score.teams[0].accuracy
    deaths = score.teams[0].deaths
    time = score.sim_time

    # threw some weightings in, can adjust to optimize better. 
    # fitness = asteroids_hit + (accuracy * 100) - (deaths * 200) 
    fitness = 1/time
        
    return fitness

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


ga_instance = pygad.GA(
    num_generations=5,
    num_parents_mating=2,
    sol_per_pop=3,
    fitness_func=fitness_function,
    num_genes=len(gene_space),
    mutation_percent_genes=10,
    gene_space= gene_space,
    random_mutation_min_val=-15.0,
    random_mutation_max_val=15.0,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    #parallel_processing=['thread', 4], # didnt see a difference in times on my machine between running parallel processes on threads?
    parallel_processing=['process', 6],
    on_generation=on_gen,
    )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Best solution fitness:", solution_fitness)

end = time.time() 
print(f"TOTAL TIME - {end - begin}")
ga_instance.plot_fitness()
