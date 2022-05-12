from scipy.signal import savgol_filter, lfilter, butter
from .kalman import kalman
import statsmodels.api as sm
import pygad
import numpy as np

class a_filter(object):
    def __init__(self,filter,params):

        self.filter = filter
        self.params = params


    def filter_this(self,prediction,size=None):

        filter = self.filter
        solution = list(self.params.values())

        if filter == 'savgol':

            window_length = int(solution[0]) # must be odd number
            polyorder = int(solution[1])

            filtered_prediction = savgol_filter(prediction,
                                                window_length,
                                                polyorder)
        elif filter == 'lfilter':

            n = int(solution[0])
            b = [1.0 / n] * n
            a = int(solution[1])

            filtered_prediction = lfilter(b,a,prediction)

        elif filter == 'stl':

            seasonal,trend = np.array(sm.tsa.filters.hpfilter(prediction, lamb=solution[0]))
            filtered_prediction = trend

        elif filter == 'kalman':

            filtered_prediction = kalman(prediction,
                                            Q = solution[0],
                                            R = solution[1])

        else:
            print('Filter not supported')



        x  = np.linspace(0, 1, len(prediction) if not size else size)
        xp = np.linspace(0, 1, len(filtered_prediction))
        fp = filtered_prediction
        filtered_prediction = np.interp(x, xp, fp)

        return filtered_prediction

def savgol_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'savgol')
    return line

def lfilter_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'lfilter')
    return line

def kalman_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'kalman')
    return line

def stl_opt_filter(prediction,target):
    line, params = optimize(prediction,target,'stl')
    return line

def optimize(predictions,targets,filter,
                num_generations = 32,
                num_parents_mating = 16,
                sol_per_pop = 40,
                options = None,
                return_obj=False):

    """ Function to optimize a filter parameters to fit the target as much as possible

            :param predictions (2-D np.ndarray): the predictiona which will be filtered
            :param targets     (2-D np.ndarray): the real behaviours of the signals
            :param filter     (str):        the type of filter

            :optional num_generations = 50             #
            :optional num_parents_mating = 4           #     Visit: https://pygad.readthedocs.io
            :optional sol_per_pop = 8                  #

            :return line (np.ndarray): the prediction once filtered
            :return params     (dict): a dict containing parameters of the filter
    """

    def filter_func(prediction,solution):

        # Get filtered function
        if filter == 'savgol':

            # Window length
            window_length = int(2*int(solution[0])+1) # must be odd number

            # Polyorder
            if int(solution[1]) >= window_length:
                return 9e9

            polyorder = int(solution[1])


            filtered_prediction = savgol_filter(prediction,
                                                window_length,
                                                polyorder)
        elif filter == 'lfilter':

            n = int(solution[0])  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = int(solution[1])

            filtered_prediction = lfilter(b,a,prediction)

        elif filter == 'stl':

            seasonal,trend = np.array(sm.tsa.filters.hpfilter(prediction, lamb=solution[0]))
            filtered_prediction = trend

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

        fitness = 0
        for prediction,target in zip(predictions,targets):
            # Get filtered prediction
            try:
                filtered_prediction = filter_func(prediction,solution)
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

            fitness += np.corrcoef(target, filtered_prediction)[0,1]**2
        return fitness

    def callback_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])

    # Atributes

    if options == None:
        if filter == 'savgol':
            num_genes = 2
            mutation_percent_genes = 1/num_genes * 100
            sample_population = [
                                    3,  # Window length
                                    1,  # Polyorder
                                    ]
            gene_space = [
                    range(int((5-1)/2), int((20-1)/2)),      # window size, later will be transformed window*2+1 because must be odd
                    range(1, 4)]                                            # polyorder

            params_labels = ['Window length', 'Polyorder']

        elif filter == 'lfilter':
            num_genes = 2
            mutation_percent_genes = 1/num_genes * 100
            sample_population = [
                                    1,  # numerator coefficient
                                    1   # denominator coefficient
                                      ]
            gene_space = [
                    range(1,7),   # n
                    range(1,4)]   # a

            params_labels = ['Numerator coefficient', 'Denominator coefficient']
        elif filter == 'stl':
            num_genes = 1
            mutation_percent_genes = 1/num_genes * 100
            sample_population = [
                                    5,   # lamb
                                    ]
            gene_space = [
                    range(0, 200)]   # lamb

            params_labels = ['Hodrick-Prescott smoothing parameter']
        elif filter == 'kalman':
            num_genes = 2
            mutation_percent_genes = 1/num_genes * 100
            sample_population = [
                                    1e-5,   # Q (process variance)
                                    71      # R (estimate of measurement variance)
                                    ]
            gene_space = [
                    np.linspace(0, 200, int(1e4)),   # Q (must be float)
                    np.linspace(0, 200, int(1e4))]   # R (must be float)

            params_labels = ['Q (process variance)','R (estimate measurement of the process variance)']
        else:
            print('Filter not supported')
    else:
        num_genes               = options['num_genes']
        mutation_percent_genes  = options['genes_to_mutate']/num_genes * 100
        sample_population       = options['sample_population']
        gene_space              = options['gene_space']
        params_labels           = options['params_labels']

    initial_population = np.zeros((sol_per_pop,num_genes)); #print('\nCreating a valid initial population')
    for i in range(sol_per_pop):
        test = 9e9
        while (isinstance(test,float) and test == 9e9):
            for j in range(num_genes):
                    if isinstance(sample_population[j],int):
                        initial_population[i,j] = int(np.random.uniform(gene_space[j][0],gene_space[j][-1]))
                    elif isinstance(sample_population[j],float):
                        initial_population[i,j] = float(np.random.uniform(gene_space[j][0],gene_space[j][-1]))
            test = filter_func(predictions[0],initial_population[i,:])
        #print(f'    {i+1}/{sol_per_pop} created -> {initial_population[i,:]}')

    # Initialize ga_instance

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       initial_population = initial_population,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space = gene_space,
                       #callback_generation=callback_gen
                       )


    ga_instance.run()

    # Get the best solution and asign parameters
    solution, solution_fitness, solution_idx = ga_instance.best_solution()


    # Compute line for the best solution
    lines = list()
    for prediction in predictions:
        lines.append(filter_func(prediction,solution))

    # Print parameters and fitness score
    if filter == 'savgol':
        solution[0] = solution[0]*2+1

    params = dict(zip(params_labels, solution))

    print('\nFilter:',filter)
    print("Parameters of the best solution:")
    for key, val in params.items():
        print(f' {key} = {val}')
    print("Fitness value of the best solution (r²) = {solution_fitness}".format(solution_fitness=solution_fitness))
    print('')


    if return_obj:
        return a_filter(filter,params)
    else:
        return lines, params
