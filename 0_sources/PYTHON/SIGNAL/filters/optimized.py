from scipy.signal import savgol_filter, lfilter, butter
from .kalman import kalman
import pygad
import numpy as np


def savgol_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'savgol')
    return line

def lfilter_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'lfilter')
    return line

def kalman_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'kalman')
    return line

def optimize(prediction,target,filter,
                num_generations = 50,
                num_parents_mating = 4,
                sol_per_pop = 8,
                parent_selection_type = "sss",
                keep_parents = 1,
                crossover_type = "single_point",
                mutation_type = "random"):

    """ Function to optimize a filter parameters to fit the target as much as possible

            :param prediction (np.ndarray): the prediction which will be filtered
            :param target     (np.ndarray): the real behaviour of the signal
            :param filter     (str):        the type of filter

            :optional num_generations = 50             #
            :optional num_parents_mating = 4           #
            :optional sol_per_pop = 8                  #
            :optional parent_selection_type = "sss"    #        Visit: https://pygad.readthedocs.io
            :optional keep_parents = 1                 #
            :optional crossover_type = "single_point"  #
            :optional mutation_type = "random"         #
            :optional mutation_percent_genes = 10      #

            :return line (np.ndarray): the prediction once filtered
            :return params     (dict): a dict containing parameters of the filter
    """

    def filter_func(solution):

        # Get filtered function
        if filter == 'savgol':

            # Mode constrains and conversion
            if solution[4] >= 5 or solution[4]<0:
                return 9e9

            modes = {'0':'mirror','1':'constant','2':'nearest','3':'wrap','4':'interp'}
            mode = modes[str(int(solution[4]))]

            # Window length constrains
            if mode == 'interp' and int(2*int(solution[0])+1) > len(x):
                return 9e9

            window_length = int(2*int(solution[0])+1) # must be odd number

            # Polyorder
            if int(solution[1]) >= window_length:
                return 9e9

            polyorder = int(solution[1])

            # Deriv
            if int(solution[2]) < 0:
                return 9e9

            # Delta
            if int(solution[2])!= 0 and float(solution[3]) == 0:
                return 9e9

            delta = float(solution[3])

            filtered_prediction = savgol_filter(prediction,
                                                window_length,
                                                polyorder,
                                                deriv = int(solution[2]),
                                                delta = delta,
                                                mode  = mode,
                                                cval  = float(solution[5]))
        elif filter == 'lfilter':

            if solution[0] <= 0:
                return 9e9
            if solution[1] < 0 or solution[1] > 1:
                return 9e9

            b, a = butter(int(solution[0]),float(solution[1]))
            filtered_prediction = lfilter(b,a,prediction)
        elif filter == 'kalman':

            filtered_prediction = kalman(prediction,
                                            Q = solution[0],
                                            R = solution[1])
        else:
            print('Filter not supported')


        # Interpolate
        if len(filtered_prediction) != len(prediction):
            x  = np.linspace(0, 1, len(prediction))
            xp = np.linspace(0, 1, len(filtered_prediction))
            fp = filtered_prediction
            filtered_prediction = np.interp(x, xp, fp)

        return filtered_prediction

    def fitness_func(solution, solution_idx):

        # Get filtered prediction
        try:
            filtered_prediction = filter_func(solution)
            if isinstance(filtered_prediction,float):
                return 9e9

        except Exception as e:
            print('\nError while computing filtered prediction')
            print(e)
            print(solution,'\n')
            return 9e9

        # Interpolate to compute error
        if len(filtered_prediction) != len(target):
            x  = np.linspace(0, 1, len(target))
            xp = np.linspace(0, 1, len(filtered_prediction))
            fp = filtered_prediction
            filtered_prediction = np.interp(x, xp, fp)

        fitness = np.sqrt(np.mean((target-filtered_prediction)**2))
        return fitness

    fitness_function = fitness_func

    if filter == 'savgol':
        num_genes = 6
        mutation_percent_genes = 2/num_genes * 100
        sample_population = [
                                3,  # Window length
                                1,  # Polyorder
                                0,  # Deriv
                                1.0,# Delta
                                4,  # Mode (interp)
                                0.0 # cval
                                ]
        gene_space = [
                range(1, len(prediction)), # window size
                range(0, 6),               # polyorder
                range(0, 3),               # deriv
                range(-3, 3),              # delta (must be float)
                range(0, 4),               # mode (five modes)
                range(-3, 3)]              # cval (must be float)

    elif filter == 'lfilter':
        num_genes = 2
        mutation_percent_genes = 1/num_genes * 100
        sample_population = [
                                0,  # Order of lowpass filter
                                0.5 # Critical frequency of low pass filter
                                  ]
        gene_space = [
                range(0,6),    # Order
                range(0,1)]    # Critical frequency (must be float)
    elif filter == 'kalman':
        num_genes = 2
        mutation_percent_genes = 1/num_genes * 100
        sample_population = [
                                1e-5,   # Q (process variance)
                                0.1**2, # R (estimate of measurement variance)
                                ]
        gene_space = [
                range(0,1),    # Q (must be float)
                range(0,1)]    # R (must be float)
    else:
        print('Filter not supported')

    initial_population = np.zeros((sol_per_pop,num_genes))
    for i in range(sol_per_pop):
        for j in range(num_genes):
            initial_population[i,j] = np.random.uniform(-0.1, 0.1) * np.array(sample_population[j])

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       initial_population = initial_population,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space = gene_space)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print('\nFilter:',filter)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print('')

    params = {'a':solution}
    line = filter_func(solution)

    return line, params
