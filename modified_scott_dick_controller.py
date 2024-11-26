# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt


class ModifiedScottDickController(KesslerController):
    
    def __init__(self):
        self.eval_frames = 0 #What is this?

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
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
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


####################################################################################################
## Self Avoidance Stuff##
####################################################################################################
        
        distance_low_a = 0
        distance_low_b = 0
        distance_low_c = 100
        distance_med_a = 75
        distance_med_b = 250
        distance_med_c = 350
        distance_high_a = 345
        distance_high_b = 600
        distance_high_c = 600
        
        distance = ctrl.Antecedent(np.arange(distance_low_a, distance_high_c + 1, 1), 'distance')

        distance['close'] = fuzz.trimf(distance.universe, [distance_low_a, distance_low_b, distance_low_c])
        distance['medium'] = fuzz.trimf(distance.universe, [distance_med_a, distance_med_b, distance_med_c])
        distance['far'] = fuzz.trimf(distance.universe, [distance_high_a, distance_high_b, distance_high_c])

        angle_front_a = 0
        angle_front_b = 0
        angle_front_c = 35
        angle_front_glance_a = 25
        angle_front_glance_b = 45
        angle_front_glance_c = 75
        angle_glance_a = 65
        angle_glance_b = 90
        angle_glance_c = 125
        angle_back_glance_a = 120
        angle_back_glance_b = 135
        angle_back_glance_c = 150
        angle_back_a = 145
        angle_back_b = 180
        angle_back_c = 180

        angle = ctrl.Antecedent(np.arange(0, 181, 1), 'angle')

        angle['front'] = fuzz.trimf(angle.universe, [angle_front_a, angle_front_b, angle_front_c])
        angle['front_glancing'] = fuzz.trimf(angle.universe, [angle_front_glance_a, angle_front_glance_b, angle_front_glance_c])
        angle['glancing'] = fuzz.trimf(angle.universe, [angle_glance_a, angle_glance_b, angle_glance_c])
        angle['back_glancing'] = fuzz.trimf(angle.universe, [angle_back_glance_a, angle_back_glance_b, angle_back_glance_c])
        angle['behind'] = fuzz.trimf(angle.universe, [angle_back_a, angle_back_b, angle_back_c])

        thrust_back_high_a = -300
        thrust_back_high_b = -250
        thrust_back_high_c = -150
        thrust_back_medium_a = -175
        thrust_back_medium_b = -100
        thrust_back_medium_c = -75
        thrust_back_low_a = -100
        thrust_back_low_b = -75
        thrust_back_low_c = -25
        thrust_zero_a = -50
        thrust_zero_b = 0
        thrust_zero_c = 50
        thrust_forwards_low_a = 25
        thrust_forwards_low_b = 75
        thrust_forwards_low_c = 100
        thrust_forwards_medium_a = 75
        thrust_forwards_medium_b = 100
        thrust_forwards_medium_c = 175
        thrust_forwards_high_a = 150
        thrust_forwards_high_b = 250
        thrust_forwards_high_c = 325

        thrust = ctrl.Consequent(np.arange(-300, 325, 1), 'thrust')

        thrust['back_high'] = fuzz.trimf(thrust.universe, [thrust_back_high_a, thrust_back_high_b, thrust_back_high_c])
        thrust['back_medium'] = fuzz.trimf(thrust.universe, [thrust_back_medium_a, thrust_back_medium_b, thrust_back_medium_c])
        thrust['back_low'] = fuzz.trimf(thrust.universe, [thrust_back_low_a, thrust_back_low_b, thrust_back_low_c])
        thrust['zero'] = fuzz.trimf(thrust.universe, [thrust_zero_a, thrust_zero_b, thrust_zero_c])
        thrust['forwards_low'] = fuzz.trimf(thrust.universe, [thrust_forwards_low_a, thrust_forwards_low_b, thrust_forwards_low_c])
        thrust['forwards_medium'] = fuzz.trimf(thrust.universe, [thrust_forwards_medium_a, thrust_forwards_medium_b, thrust_forwards_medium_c])
        thrust['forwards_high'] = fuzz.trimf(thrust.universe, [thrust_forwards_high_a, thrust_forwards_high_b, thrust_forwards_high_c])
        
        mrule1 = ctrl.Rule(distance['close'] & angle['front'], thrust['back_high'])
        mrule2a = ctrl.Rule(distance['close'] & angle['front_glancing'], thrust['back_high'])
        mrule2 = ctrl.Rule(distance['close'] & angle['glancing'], (thrust['back_high']))
        mrule2b = ctrl.Rule(distance['close'] & angle['back_glancing'], (thrust['forwards_high']))
        mrule3 = ctrl.Rule(distance['close'] & angle['behind'], (thrust['forwards_high']))
        mrule4 = ctrl.Rule(distance['medium'] & angle['front'], (thrust['back_medium']))
        mrule5a = ctrl.Rule(distance['medium'] & angle['front_glancing'], (thrust['back_low']))
        mrule5 = ctrl.Rule(distance['medium'] & angle['glancing'], (thrust['back_low']))
        mrule5b = ctrl.Rule(distance['medium'] & angle['back_glancing'], (thrust['forwards_low']))
        mrule6 = ctrl.Rule(distance['medium'] & angle['behind'], (thrust['forwards_medium']))
        mrule7 = ctrl.Rule(distance['far'] & angle['front'], (thrust['forwards_medium']))
        mrule8a = ctrl.Rule(distance['far'] & angle['front_glancing'], (thrust['forwards_medium']))
        mrule8 = ctrl.Rule(distance['far'] & angle['glancing'], (thrust['back_medium']))
        mrule8b = ctrl.Rule(distance['far'] & angle['back_glancing'], (thrust['forwards_low']))
        mrule9 = ctrl.Rule(distance['far'] & angle['behind'], (thrust['zero']))

        self.avoidance_control = ctrl.ControlSystem()
        self.avoidance_control.addrule(mrule1)
        self.avoidance_control.addrule(mrule2)
        self.avoidance_control.addrule(mrule3)
        self.avoidance_control.addrule(mrule4)
        self.avoidance_control.addrule(mrule5)
        self.avoidance_control.addrule(mrule6)
        self.avoidance_control.addrule(mrule7)
        self.avoidance_control.addrule(mrule8)
        self.avoidance_control.addrule(mrule9)
        self.avoidance_control.addrule(mrule2a)
        self.avoidance_control.addrule(mrule2b)
        self.avoidance_control.addrule(mrule5a)
        self.avoidance_control.addrule(mrule5b)
        self.avoidance_control.addrule(mrule8a)
        self.avoidance_control.addrule(mrule8b)

    '''
    Try to map out if an asteroid is within a certain distance from the ship. If it is closer than a certain threshold, then determine if the asteroid's
    radius will come within the ship's radius. If it is calculated to, then determine the relative angle of where the asteroid is in comparison to the ship.
    Returns angle in degrees.

    I tried using this angle as an antecedant (the angle of the asteroid that should hit us), but if a collision would be detected and another asteroid was 
    close by, it would steer itself into the other asteroid.
    '''
    def isCollision(self, asteroid, ship, current_dist):
        DISTANCE_THRESHOLD = 150
        if current_dist < DISTANCE_THRESHOLD:
            asteroid_pos_x, asteroid_pos_y = asteroid["position"]
            asteroid_vel_x, asteroid_vel_y = asteroid["velocity"]

            ship_pos_x, ship_pos_y = ship["position"]
            ship_vel_x, ship_vel_y = ship["velocity"]

            relative_vel_x = asteroid_vel_x - ship_vel_x
            relative_vel_y = asteroid_vel_y - ship_vel_y

            mag_relative_vel = math.sqrt(relative_vel_x**2 + relative_vel_y**2)

            min_distance_between = abs((asteroid_pos_x-ship_pos_x) * relative_vel_y - (asteroid_pos_y - ship_pos_y) * relative_vel_x) / mag_relative_vel

            if min_distance_between <= (ship["radius"] + asteroid["radius"]):
                r_x = ship_pos_x - asteroid_pos_x
                r_y = ship_pos_y - asteroid_pos_y

                r_magnitude = math.sqrt(r_x**2 + r_y**2)
                r_x /= r_magnitude
                r_y /= r_magnitude

                v_rel_x = relative_vel_x
                v_rel_y = relative_vel_y

                dot_product = v_rel_x * r_x + v_rel_y * r_y
                direction_cosine = dot_product / mag_relative_vel

                angle = math.acos(direction_cosine)
                angle_deg = math.degrees(angle)

                return (angle_deg, True)
        return (0, False)


    '''
    Calculate the angle between the direction of the relative velocity of an asteroid and the relative position of the asteroid from the ship. 
    Used to determine if the asteroid is moving towards/away at an angle. Returns the angle that the asteroid is moving relative to the ship.
    I think doctor Dick did this, but i was to far into it to realize what i was doing...
    '''
    def angleBetweenShipAndAsteroid(self, asteroid, ship):
        asteroid_pos = np.array(asteroid["position"])
        asteroid_vel = np.array(asteroid["velocity"])
        ship_pos = np.array(ship["position"])
        ship_vel = np.array(ship["velocity"])

        relative_vel = asteroid_vel - ship_vel

        relative_pos = ship_pos - asteroid_pos

        # Normalize to be [-1,1]
        relative_vel_unit = relative_vel / np.linalg.norm(relative_vel)
        relative_pos_unit = relative_pos / np.linalg.norm(relative_pos)

        # Compute the angle 
        dot_product = np.dot(relative_vel_unit, relative_pos_unit)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        return angle_deg
        

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        collision_angle = 0
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                angle, collision =  self.isCollision(a, ship_state, curr_dist)
                if collision:
                    collision_angle = angle
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
                    angle =  self.angleBetweenShipAndAsteroid(a, ship_state)
                    # angle, collision =  self.isCollision(a, ship_state, curr_dist)
                    # if collision:
                    #     collision_angle = angle

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
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
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
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
        
        #Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        fire = False
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
        
        thrust_calc = ctrl.ControlSystemSimulation(self.avoidance_control,flush_after_run=1)
        thrust_calc.input['angle'] = abs(collision_angle)
        thrust_calc.input['distance'] = curr_dist
        
        thrust_calc.compute()
        thrust = thrust_calc.output['thrust']
       
        drop_mine = False
        
        self.eval_frames +=1
        
        #DEBUG
        #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        print(thrust, angle)
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "ScottDick Controller"
