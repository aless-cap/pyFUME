class GeneticAlgorithm:
    def __init__(self, dataX, dataY, population_size, variable_names, generations, crossover_prob, mutation_prob, selection_method, performance_metric, verbose):
        self.dataX = dataX
        self.dataY = dataY
        self.variable_names = variable_names
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method
        self.verbose = verbose
        self.num_features = len(dataX.columns)
        self.population = self.initialize_population()

    def initialize_population(self):
        # Each individual is a binary vector of length num_features
        return np.random.choice([0, 1], (self.population_size, self.num_features))

    def calculate_fitness(self, individual):
        # Extract features based on the individual's representation
        selected_features = self.dataX.columns[individual.astype(bool)]
        X = self.dataframe[selected_features]

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, dataY, test_size=0.3, random_state=42)

        # Train Random Forest and calculate fitness (negative MSE)
        regressor = RandomForestRegressor(n_jobs=-1)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        #implement different types of performance metrics?
        mse = mean_squared_error(y_test, y_pred)
        return -mse  # Negative MSE because we want to maximize fitness

    def select_parents(self,fitness_scores):
        if self.selection_method == "roulette":
          # Roulette wheel selection
          random_indices = [int(0) if x >= self.population_size else int(x) 
                      for x in np.floor(np.random.exponential(self.population_size//5, self.population_size))]
          parents = self.population[(-fitness_scores).argsort()[random_indices]]
          return parents
        elif self.selection_method == "tournament":
          # Tournament selection
          tournament_size = 5
          parents = []
          for _ in range(self.population_size):
              tournament = np.random.randint(0, self.population_size, tournament_size)
              best_in_tournament = np.argmax([fitness_scores[i] for i in tournament])
              parents.append(self.population[tournament[best_in_tournament]])
          return parents
        elif self.selection_method == "best":
          # Best half
          parents1 = self.population[(-fitness_scores).argsort()[:self.population_size//2]]
          parents2 = self.population[(-fitness_scores).argsort()[:(self.population_size-(self.population_size//2))]]
          parents = np.concatenate((parents1, parents2), axis = 0)
          return parents

    def crossover(self, parent1, parent2):
        # One-point crossover
        crossover_point = np.random.randint(1, self.num_features)
        offspring = np.empty(self.num_features)
        offspring[0:crossover_point] = parent1[0:crossover_point]
        offspring[crossover_point:] = parent2[crossover_point:]
        return offspring

    def mutate(self, individual):
        # Mutation - flipping bits with a certain probability
        for i in range(self.num_features):
            if np.random.rand() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        # Main loop of the genetic algorithm
        start_time = time.time()
        if self.verbose: print("Starting Genetic Algorithm...")
        best_fitness = None

        # DataFrame to save population at each generation
        population_history = pd.DataFrame()

        for generation in range(self.generations):
            gen_start = time.time()

            # Evaluate fitness
            fitness_scores = np.array([self.calculate_fitness(ind) for ind in self.population])
            if self.verbose: print(f"Generation {generation+1}/{self.generations}: Fitness evaluation completed.")

            # Optionally save the population of this generation
            if save_population:
                gen_population_df = pd.DataFrame(self.population)
                gen_population_df['Fitness'] = fitness_scores
                gen_population_df['Generation'] = generation + 1
                population_history = population_history.append(gen_population_df, ignore_index=True)
            
            # Selection
            parents = self.select_parents(fitness_scores)
            if self.verbose: print(f"Generation {generation+1}/{self.generations}: Parent selection completed.")

            # Crossover and Mutation
            next_population = []
            for _ in range(self.population_size):
                if np.random.rand() < self.crossover_prob:
                    # Select parent indices from the parents array
                    parent_indices = np.random.choice(self.population_size, 2, replace=False)
                    # Extract the actual parent individuals using the selected indices
                    parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
                    offspring = self.crossover(parent1, parent2)
                else:
                    # If no crossover, just copy an individual from the parents
                    offspring = parents[np.random.choice(self.population_size)]

                if np.random.rand() < self.mutation_prob:
                    offspring = self.mutate(offspring)
                
                next_population.append(offspring)

            self.population = np.array(next_population)
            gen_end = time.time()
            if self.verbose: print(f"Generation {generation+1}/{self.generations} completed in {gen_end - gen_start:.2f} seconds.")

            gen_best_fitness = np.max(fitness_scores)
            if best_fitness is None or best_fitness < gen_best_fitness:
                best_fitness = gen_best_fitness
                best_individual = self.population[np.argmax(fitness_scores)]
              
        # Return best individual
        total_time = time.time() - start_time
        if self.verbose: print(f"Genetic Algorithm completed in {total_time:.2f} seconds.")

        selected_feature_indices = [i for i, value in enumerate(best_individual) if value == 1]

        selected_feature_names = [self.variable_names[i] for i in selected_feature_indices]
        if self.verbose: print('The following features were selected:',  selected_feature_names)
        
        return selected_feature_indices, selected_feature_names