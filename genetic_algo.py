import pygad

class GeneticAlgorithm:

    def __init__(self, game, scenario, genetic_controller, baseline_controller):
        self.genetic_controller = genetic_controller
        self.baseline_controller = baseline_controller
        self.game = game
        self.scenario = scenario
        self.ga_instance = None
        self.gene_space = [
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
            # enemy dist
            {"low": 130, "high": 160},  # C zmf upper value
            # danger level
            {"low": 4990, "high": 5050},  # L zmf upper value
        ]

    def generate_params(self, solution):
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
            "enemy_dist_C": solution[24],
            "danger_L": solution[25]
        }
        return params
    
    def fitness_function(self, ga_instance, solution, solution_idx):
        params = self.generate_params(solution)
        genetic_controller_instance = self.genetic_controller(params)
        score, perf_data = self.game.run(
            scenario=self.scenario,
            controllers=[genetic_controller_instance, self.baseline_controller()],
        )

        asteroids_hit = score.teams[0].asteroids_hit
        accuracy = score.teams[0].accuracy
        deaths = score.teams[0].deaths
        time = score.sim_time

        fitness = asteroids_hit + (accuracy * 100) - (deaths * 30)
        return fitness
    
    def on_gen(self, ga_instance):
        print("\nGeneration : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])
        for idx, solution in enumerate(ga_instance.population):
            formatted_solution = [int(gene) for gene in solution]
            print(f"Solution {idx}: {formatted_solution}")

    def print_best_solution(self, ga_instance):
        print("\n---------------------------------------------------------------")
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        formatted_solution = [int(gene) for gene in solution]
        print("Best solution Generation", ga_instance.best_solution_generation)
        print("Best solution:", formatted_solution)
        print("Best solution fitness:", solution_fitness)
    
    def train(self):      
        ga_instance = pygad.GA(
            num_generations=3,
            num_parents_mating=4,
            sol_per_pop=8,
            fitness_func=self.fitness_function,
            num_genes=len(self.gene_space),
            mutation_percent_genes=40,
            gene_space=self.gene_space,
            random_mutation_min_val=-30.0,
            random_mutation_max_val=30.0,
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            # parallel_processing=['thread', 4], # didnt see a difference in times on my machine between running parallel processes on threads?
            parallel_processing=["process", 6],
            on_generation=self.on_gen,
        )

        ga_instance.run()
        self.ga_instance = ga_instance
        self.print_best_solution(ga_instance)
        a = self.generate_params(ga_instance.best_solution()[0])
        return a
        #return ga_instance.best_solution()
        
        
    def plot(self):
        if self.ga_instance != None:
            self.ga_instance.plot_fitness
        else:
            pass
