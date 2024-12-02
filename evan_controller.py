# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt


class EvanController(KesslerController):
    def __init__(self):
        self.eval_frames = 0 #What is this?

        self.iterator = 120 #For timing mines
        self.number_asteroids = 0 #For tracking number of asteroids remaining


        #################################################################################################################
        #DEFINE CONTROLS FOR AUTO-TARGETING MODE
        #################################################################################################################

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))  
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)



    #################################################################################################################
    #SHIP ACTIONS
    #################################################################################################################

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:

        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        
        #Two asteroids found each time.
        #Most deadly asteroid is asteroid which will strike ship soonest, but also weighted on size.
        #Closeset asteroid closest asteroid outside distance of 70 (too close asteroids hard to hit)
        #Asteroid to shoot is passed to SD controller. If any deadly asteroids exist, they assigned. If none, then closest.
        most_deadly_asteroid = None  
        closest_asteroid = None
        asteroid_to_shoot = None

        #For labelling if an asteroid is dangerous or not.
        asteroid_is_dangerous = False
        
        #Each asteroid has its danger level rated.
        asteroid_danger_level = 0
        #Ship danger level is sum of asteroid danger levels.
        ship_danger_level = 0
        #Number of deadly asteroids.
        deadly_asteroids = 0

        #Variables for computing vector math to determine which asteroids are on a trajectory
        #for our ship. These are considered "dangerous asteroids" or "deadly asteroids"
        intercept1 = [0,0]
        intercept2 = [0,0]
        asteroidvec = [0,0]
        tangentvec = [0,0]
        tangentveclen = 0
        paddingdistance = 1
        interceptvec1 = [0,0]
        interceptvec2 = [0,0]
        crossprod1 = 0
        crossprod2 = 0
        
        self.number_asteroids = 0

        for a in game_state["asteroids"]:
            self.number_asteroids += 1
            
            ############################################################################################################
            #Part 1: Finding the next-most dangerous asteroid and the ship's danger level.
            ############################################################################################################
                 
            #Assume asteroid is safe until proven guilty
            asteroid_is_dangerous = False
            asteroid_danger_level = 0
            
            #Get vector from asteroid to ship
            asteroidvec = tuple(map(lambda i,j: i-j, ship_state["position"], a["position"]))
            
            #Get perpendicular of asteroid vector and normalize
            tangentvec[0] = -asteroidvec[1]
            tangentvec[1] = asteroidvec[0]
            tangentveclen = sqrt(tangentvec[0] ** 2 + tangentvec[1] ** 2).real

            #Get padding distance, current distance, and approximate time to striking the ship
            #padding distance how lose an asteroid is allowed to get to the ship. The absolute minimum
            #Is the ship's radius plus the asteroid's radius plus one so a strike doesn't occur.
            #I increased asteroid's radius by 20% so big asteroids need to keep further away because
            #they create a spread of asteroids when they explode.
            aster_speed = sqrt(a["velocity"][0] ** 2 + a["velocity"][1] ** 2).real
            paddingdistance = a["radius"] * 1.2 + ship_state["radius"] + 1
            curr_dist = tangentveclen
            strike_time = (curr_dist - paddingdistance) / (aster_speed)

            #Create tangential vector
            tangentvec[0] = tangentvec[0] / tangentveclen
            tangentvec[1] = tangentvec[1] / tangentveclen

            #Make intercept bound 1
            intercept1[0] = ship_state["position"][0] + tangentvec[0] * paddingdistance
            intercept1[1] = ship_state["position"][1] + tangentvec[1] * paddingdistance

            #Make intercept bound 2
            intercept2[0] = ship_state["position"][0] - tangentvec[0] * paddingdistance
            intercept2[1] = ship_state["position"][1] - tangentvec[1] * paddingdistance

            #Make vectors from intercepts to the asteroid
            interceptvec2 = tuple(map(lambda i,j: i-j, intercept1, a["position"]))
            interceptvec1 = tuple(map(lambda i,j: i-j, intercept2, a["position"]))

            #Check if asteroid velocity within intercept boundaries (take cross products)
            crossprod1 = interceptvec1[0] * a["velocity"][1] - interceptvec1[1] * a["velocity"][0]
            crossprod2 = interceptvec2[0] * a["velocity"][1] - interceptvec2[1] * a["velocity"][0]
            between_angles = (crossprod1 * crossprod2) < 0

            #Check if asteroid velocity heading towards or away from ship (take dot product)
            correct_direction = (a["velocity"][0] * asteroidvec[0] + a["velocity"][1] * asteroidvec[1]) > 0

            #Determine if asteroid is bound for the ship (asteroid is dangerous)
            if (between_angles == True) and (correct_direction == True):
                asteroid_is_dangerous = True
                deadly_asteroids += 1
                #Big asteroids dangerous, close asteroids VERY dangerous. Danger level linear with asteroid size,
                #and squared with strike time
                #This weighting function would make a big difference in controller operation.
                asteroid_danger_level = (a["size"] / (strike_time ** 2)) * 100

                #ship_danger_level mostly stays less than 1 when all dangerous asteroids far away.
                #If it gets above 100, you're in danger
                #Shoots into the 10,000's when you're about to die.
                ship_danger_level += asteroid_danger_level
            

            ############################################################################################################
            #Part 2: Finding the closest asteroid and most deadly asteroid
            ############################################################################################################            

            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum.
                closest_asteroid = dict(aster = a, dist = curr_dist)

            if closest_asteroid["dist"] > curr_dist and curr_dist > 70:
                # New minimum found
                # Only aim for the closest asteroid that isn't super close
                # The super-close ones are hard to hit.
                # Should add something to aim for super close ones if no other asteroids left.
                closest_asteroid["aster"] = a
                closest_asteroid["dist"] = curr_dist

            if asteroid_is_dangerous == True:
                #Initialize closest deadly asteroid
                if most_deadly_asteroid is None :
                    # Does not yet exist, so initialize first asteroid as the most deadly
                    most_deadly_asteroid = dict(aster = a, dist = curr_dist, time = strike_time, dang = asteroid_danger_level)

                if most_deadly_asteroid["dang"] < asteroid_danger_level:
                    #New minimum found. Initialize
                    #Asteroid time and danger level not used yet, but added in case we want them
                    #to be used in the controller.
                    most_deadly_asteroid["aster"] = a
                    most_deadly_asteroid["dist"] = curr_dist
                    most_deadly_asteroid["time"] = strike_time
                    most_deadly_asteroid["dang"] = asteroid_danger_level


        ############################################################################################################
        #Part 3: Selector for targeting system. Shoot most deadly asteroid, followed by closest.
        ############################################################################################################
        
        #Declare the targeted asteroid
        asteroid_to_shoot = dict(aster = None, dist = 0, time = None, dang = 0)

        #Assign most deadly asteroid as targeted
        if most_deadly_asteroid is not None:
            #For debugging:
            #print("Danger Level:")
            #print(ship_danger_level)
            asteroid_to_shoot["aster"] = most_deadly_asteroid["aster"]
            asteroid_to_shoot["dist"] = most_deadly_asteroid["dist"]
        #If no deadly asteroids, shoot closest for speed.
        else:
            #Debugging:
            #print("SAFE!")
            asteroid_to_shoot["aster"] = closest_asteroid["aster"]
            asteroid_to_shoot["dist"] = closest_asteroid["dist"]


        ############################################################################################################
        #Part 4: Scott Dick Shooting Controller
        ############################################################################################################
        
        asteroid_ship_x = ship_pos_x - asteroid_to_shoot["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - asteroid_to_shoot["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(asteroid_to_shoot["aster"]["velocity"][1], asteroid_to_shoot["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(asteroid_to_shoot["aster"]["velocity"][0]**2 + asteroid_to_shoot["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * asteroid_to_shoot["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (asteroid_to_shoot["dist"]**2))
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * asteroid_to_shoot["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * asteroid_to_shoot["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = asteroid_to_shoot["aster"]["position"][0] + asteroid_to_shoot["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = asteroid_to_shoot["aster"]["position"][1] + asteroid_to_shoot["aster"]["velocity"][1] * (bullet_t+1/30)

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False


        ############################################################################################################
        #Part 5: Special Cases Programming
        ############################################################################################################

        #Haven't figured these out yet. Set static for now.
        thrust = 0
        drop_mine = False

        #Firing special cases:
        #If more than 30 asteroids remaining or if danger level high high, never stop shooting.
        #This is to counter when SD controller pauses shooting during turning and ship in danger.
        self.iterator += 1
        if ship_danger_level > 30:
            fire = True
        elif self.number_asteroids > 30:
            fire = True

        #Mine special cases:
        #If in danger, drop mine to try to clear area. Losing one life often worth the space.
        #Iterator forces ship to wait until previous mine detonated until dropping another.
        if ship_danger_level > 200 and self.iterator > 120 and ship_state["lives_remaining"] > 1:
            drop_mine = True
            self.iterator = 0
        
        #If about to die, drop suicide mine to get grave kills.
        #I found this was great at sending mines at the SD controller, killing it, and getting a bunch
        #of points in the process.
        if ship_danger_level > 5000 and self.iterator > 100:
            drop_mine = True
        
        self.eval_frames +=1
        
        #DEBUG
        #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Red Leader"