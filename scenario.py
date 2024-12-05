# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from graphics_both import GraphicsBoth

from EGABML_team_controller import EGABMLTeamController

my_test_scenario = Scenario(
    name="Test Scenario",
    num_asteroids= 10,
    ship_states=[
        {
            "position": (400, 400),
            "angle": 90,
            "lives": 3,
            "team": 1,
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

game = TrainerEnvironment(settings=game_settings)
game = KesslerGame(settings=game_settings)

score, perf_data = game.run(
    scenario=my_test_scenario,
    controllers=[EGABMLTeamController()],
)

print(score.stop_reason)
print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
print("Deaths: " + str([team.deaths for team in score.teams]))
print("Accuracy: " + str([team.accuracy for team in score.teams]))
print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))
