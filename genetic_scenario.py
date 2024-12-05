# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

import pygad

from genetic_algo import GeneticAlgorithm
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from graphics_both import GraphicsBoth
from attempts.scott_dick_controller import ScottDickController

from genetic_controller import GeneticTeamController

my_test_scenario = Scenario(
    name="Test Scenario",
    # asteroid_states=[
        # {"position": (200, 400), "angle": 0, "speed": 30, "size": 4},
        #  {'position': (400, 600), 'angle': 30, 'speed': 30, 'size': 4},
        #  {'position': (600, 400), 'angle': 0, 'speed': 30, 'size': 4},
        #  {'position': (200, 200), 'angle': 0, 'speed': 60, 'size': 4},
    # ],
    num_asteroids= 10,
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

#game = KesslerGame(settings=game_settings)
game = TrainerEnvironment(settings=game_settings)

begin = time.time()
ga = GeneticAlgorithm(game, my_test_scenario, GeneticTeamController, ScottDickController)
best_solution = ga.train()
end = time.time()
print(f"TOTAL GA TIME - {int(end - begin)} seconds")

print("\n---------------------------------------------------------------")
print("Running with Best Solution:")
game = KesslerGame(settings=game_settings)

score, perf_data = game.run(
    scenario=my_test_scenario,
    controllers=[GeneticTeamController(best_solution), ScottDickController()],
)

print(score.stop_reason)
print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
print("Deaths: " + str([team.deaths for team in score.teams]))
print("Accuracy: " + str([team.accuracy for team in score.teams]))
print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))
