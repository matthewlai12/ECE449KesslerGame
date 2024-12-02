from pickle import FALSE

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

class ConstantDistance(KesslerController):
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

# Trying to keep a constant distance between the closest asteroid and the ship with this############
####################################################################################################
## Self Avoidance Stuff##
####################################################################################################
        '''
        try and keep at a constant distance
        '''
        target_distance = 120

        close_a = 0
        close_b = 0
        close_c = 80
        optimal_a = 60
        optimal_b = 120
        optimal_c = 170
        far_a = 140
        far_b = 600
        far_c = 600

        distance = ctrl.Antecedent(np.arange(close_a, far_c + 1, 1), 'distance')

        distance['close'] = fuzz.trimf(distance.universe, [close_a, close_b, close_c])
        distance['optimal'] = fuzz.trimf(distance.universe, [optimal_a, optimal_b, optimal_c])
        distance['far'] = fuzz.trimf(distance.universe, [far_a, far_b, far_c])

        '''
        Control the speed. Once get over a certain speed it seems to lose it's mind.
        '''
        max_speed = 150

        fast_back_a = -240
        fast_back_b = -240
        fast_back_c = -160
        good_back_a = -180
        good_back_b = -145
        good_back_c = -90
        slow_a = -110
        slow_b = 0
        slow_c = 110
        good_front_a = 90
        good_front_b = 145
        good_front_c = 180
        fast_front_a = 160
        fast_front_b = 240
        fast_front_c = 240

        speed = ctrl.Antecedent(np.arange(fast_back_a, fast_front_c + 1, 1), 'speed')

        speed["fast_back"] = fuzz.trimf(speed.universe, [fast_back_a, fast_back_b, fast_back_c])
        speed["good_back"] = fuzz.trimf(speed.universe, [good_back_a, good_back_b, good_back_c])
        speed["slow"] = fuzz.trimf(speed.universe, [slow_a, slow_b, slow_c])
        speed["good_forwards"] = fuzz.trimf(speed.universe, [good_front_a, good_front_b, good_front_c])
        speed["fast_forwards"] = fuzz.trimf(speed.universe, [fast_front_a, fast_front_b, fast_front_c])

        '''
        Is the asteroid approaching us?
        '''
        approaching = ctrl.Antecedent(np.arange(-1, 1, 0.1), 'approaching')
        approaching['N'] = fuzz.trimf(approaching.universe, [-1,-1,-0.1])
        approaching['M'] = fuzz.trimf(approaching.universe, [-0.25,0,0.25])
        approaching['Y'] = fuzz.trimf(approaching.universe, [0.1,1,1]) 


        '''
        Thrust consequent
        '''
        thrust_back_high_a = -325
        thrust_back_high_b = -250
        thrust_back_high_c = -150
        thrust_back_medium_a = -175
        thrust_back_medium_b = -145
        thrust_back_medium_c = -100
        thrust_back_low_a = -125
        thrust_back_low_b = -90
        thrust_back_low_c = -60
        thrust_zero_a = -50
        thrust_zero_b = 0
        thrust_zero_c = 50
        thrust_forwards_low_a = 60
        thrust_forwards_low_b = 90
        thrust_forwards_low_c = 125
        thrust_forwards_medium_a = 100
        thrust_forwards_medium_b = 145
        thrust_forwards_medium_c = 175
        thrust_forwards_high_a = 150
        thrust_forwards_high_b = 250
        thrust_forwards_high_c = 325

        thrust = ctrl.Consequent(np.arange(thrust_back_high_a, thrust_forwards_high_c, 1), 'thrust')

        thrust['back_high'] = fuzz.trimf(thrust.universe, [thrust_back_high_a, thrust_back_high_b, thrust_back_high_c])
        thrust['back_medium'] = fuzz.trimf(thrust.universe, [thrust_back_medium_a, thrust_back_medium_b, thrust_back_medium_c])
        thrust['back_low'] = fuzz.trimf(thrust.universe, [thrust_back_low_a, thrust_back_low_b, thrust_back_low_c])
        thrust['zero'] = fuzz.trimf(thrust.universe, [thrust_zero_a, thrust_zero_b, thrust_zero_c])
        thrust['forwards_low'] = fuzz.trimf(thrust.universe, [thrust_forwards_low_a, thrust_forwards_low_b, thrust_forwards_low_c])
        thrust['forwards_medium'] = fuzz.trimf(thrust.universe, [thrust_forwards_medium_a, thrust_forwards_medium_b, thrust_forwards_medium_c])
        thrust['forwards_high'] = fuzz.trimf(thrust.universe, [thrust_forwards_high_a, thrust_forwards_high_b, thrust_forwards_high_c])
        
        mrule0 = ctrl.Rule(distance['close'] &  speed["fast_back"] & approaching['N'], thrust['forwards_high'])
        mrule1 = ctrl.Rule(distance['optimal'] &  speed["good_back"] & approaching['N'], thrust['zero'])
        mrule2 = ctrl.Rule(distance['far'] &  speed["slow"] & approaching['N'], thrust['forwards_high'])
        mrule3 = ctrl.Rule(distance['far'] &  speed["good_forwards"] & approaching['M'], thrust['forwards_high'])
        mrule4 = ctrl.Rule(distance['optimal'] &  speed["fast_forwards"] & approaching['M'], thrust['back_high'])
        mrule5 = ctrl.Rule(distance['close'] &  speed["fast_forwards"] & approaching['Y'], thrust['back_high'])
        mrule6 = ctrl.Rule(distance['close'] &  speed["good_forwards"] & approaching['N'], thrust['back_medium'])
        mrule7 = ctrl.Rule(distance['optimal'] &  speed["slow"] & approaching['Y'], thrust['zero'])
        mrule8 = ctrl.Rule(distance['far'] &  speed["fast_back"] & approaching['Y'], thrust['forwards_high'])
        mrule9 = ctrl.Rule(distance['close'] &  speed["good_back"] & approaching['M'], thrust['forwards_low'])
        mrule10 = ctrl.Rule(distance['far'] &  speed["good_back"] & approaching['Y'], thrust['forwards_high'])
        mrule11 = ctrl.Rule(distance['optimal'] &  speed["fast_back"] & approaching['M'], thrust['forwards_medium'])
        mrule12 = ctrl.Rule(distance['optimal'] &  speed["good_forwards"] & approaching['Y'], thrust['back_low'])
        mrule13 = ctrl.Rule(distance['far'] &  speed["fast_forwards"] & approaching['N'], thrust['forwards_high'])
        mrule14 = ctrl.Rule(distance['close'] &  speed["slow"] & approaching['M'], thrust['back_low'])
        mrule15 = ctrl.Rule(distance['close'] & speed['fast_back'] & approaching['Y'], thrust['forwards_high'])
        mrule16 = ctrl.Rule(distance['close'] & speed['slow'] & approaching['M'], thrust['back_low'])
        mrule17 = ctrl.Rule(distance['close'] & speed['good_back'] & approaching['Y'], thrust['forwards_medium'])
        mrule18 = ctrl.Rule(distance['optimal'] & speed['fast_back'] & approaching['Y'], thrust['forwards_low'])
        mrule19 = ctrl.Rule(distance['optimal'] & speed['good_forwards'] & approaching['N'], thrust['zero'])
        mrule20 = ctrl.Rule(distance['optimal'] & speed['slow'] & approaching['M'], thrust['zero'])
        mrule21 = ctrl.Rule(distance['far'] & speed['fast_forwards'] & approaching['M'], thrust['back_low'])
        mrule22 = ctrl.Rule(distance['far'] & speed['good_forwards'] & approaching['Y'], thrust['zero'])
        mrule23 = ctrl.Rule(distance['far'] & speed['slow'] & approaching['M'], thrust['forwards_medium'])
        mrule23 = ctrl.Rule(distance['close'] & speed['slow'] & approaching['N'], thrust['forwards_low'])
        mrule25 = ctrl.Rule(distance['close'] & speed['good_forwards'] & approaching['Y'], thrust['back_medium'])
        mrule26 = ctrl.Rule(distance['optimal'] & speed['good_back'] & approaching['M'], thrust['zero'])
        mrule27 = ctrl.Rule(distance['optimal'] & speed['good_back'] & approaching['M'], thrust['zero'])
        mrule28 = ctrl.Rule(distance['optimal'] & speed['fast_forwards'] & approaching['N'], thrust['zero'])
        mrule29 = ctrl.Rule(distance['far'] & speed['good_forwards'] & approaching['M'], thrust['zero'])
        mrule30 = ctrl.Rule(distance['far'] & speed['fast_back'] & approaching['N'], thrust['forwards_high'])

        self.avoidance_control = ctrl.ControlSystem()
        self.avoidance_control.addrule(mrule0)
        self.avoidance_control.addrule(mrule1)
        self.avoidance_control.addrule(mrule2)
        self.avoidance_control.addrule(mrule3)
        self.avoidance_control.addrule(mrule4)
        self.avoidance_control.addrule(mrule5)
        self.avoidance_control.addrule(mrule6)
        self.avoidance_control.addrule(mrule7)
        self.avoidance_control.addrule(mrule8)
        self.avoidance_control.addrule(mrule9)
        self.avoidance_control.addrule(mrule10)
        self.avoidance_control.addrule(mrule11)
        self.avoidance_control.addrule(mrule12)
        self.avoidance_control.addrule(mrule13)
        self.avoidance_control.addrule(mrule14)
        self.avoidance_control.addrule(mrule15)
        self.avoidance_control.addrule(mrule16)
        self.avoidance_control.addrule(mrule17)
        self.avoidance_control.addrule(mrule18)
        self.avoidance_control.addrule(mrule19)
        self.avoidance_control.addrule(mrule20)
        self.avoidance_control.addrule(mrule21)
        self.avoidance_control.addrule(mrule22)
        self.avoidance_control.addrule(mrule23)
        self.avoidance_control.addrule(mrule25)
        self.avoidance_control.addrule(mrule26)
        self.avoidance_control.addrule(mrule27)
        self.avoidance_control.addrule(mrule28)
        self.avoidance_control.addrule(mrule29)
        self.avoidance_control.addrule(mrule30)




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


    def isApproaching(self, a, ship_state):
        rel_velocity_x = a["velocity"][0] - ship_state["velocity"][0]
        rel_velocity_y = a["velocity"][1] - ship_state["velocity"][1]

        dir_vector_x = a["position"][0] - ship_state["velocity"][0]
        dir_vector_y = a["position"][1] - ship_state["velocity"][0]
        
        dot_product = (rel_velocity_x * dir_vector_x) + (rel_velocity_y * dir_vector_y)
        
        # > 0 means moving away
        if dot_product > 0:
            return 1
        # < 0 means moving towards
        elif dot_product < 0:
            return -1
        else:
            return 0

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

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                angle =  self.angleBetweenShipAndAsteroid(a, ship_state)
                approaching = self.isApproaching(a, ship_state)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
                    angle =  self.angleBetweenShipAndAsteroid(a, ship_state)
                    approaching = self.isApproaching(a, ship_state)

            # if closest_asteroid is None :
            #     # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
            #     closest_asteroid = dict(aster = a, dist = death)
            #     angle =  self.angleBetweenShipAndAsteroid(a, ship_state)
                
            # else:    
            #     # closest_asteroid exists, and is thus initialized. 
            #     if closest_asteroid["dist"] > death:
            #         # New minimum found
            #         closest_asteroid["aster"] = a
            #         closest_asteroid["dist"] = death
            #         angle =  self.angleBetweenShipAndAsteroid(a, ship_state)

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
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
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
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        thrust_calc = ctrl.ControlSystemSimulation(self.avoidance_control,flush_after_run=1)
        #thrust_calc.input['angle'] = abs(angle)
        thrust_calc.input['distance'] = curr_dist
        thrust_calc.input['speed'] = ship_state["speed"]
        thrust_calc.input['approaching'] = approaching
        
        thrust_calc.compute()
        thrust = thrust_calc.output['thrust']
               

        drop_mine = False
        
        self.eval_frames +=1
        
        #DEBUG
        #print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Closest Angle Controller"
