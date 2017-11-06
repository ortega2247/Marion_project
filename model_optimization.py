import pandas as pd
import os
import numpy as np
from model_0 import model
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
from simplepso.pso import PSO


directory = os.path.dirname(__file__)
data_path = os.path.join(directory, 'data', 'erk2_data.xlsx')
data = pd.read_excel(data_path)
erk2_norm = data.loc['perk2'].values / np.nanmax(data.loc['perk2'])
erk2_var = np.var(erk2_norm)
p70s6k_norm = data.loc['p70S6K'].values / np.nanmax(data.loc['p70S6K'])
p70s6k_var = np.var(p70s6k_norm)

ntimes = len(data.columns.values)
tmul = 10
tspan = np.linspace(data.columns.values[0], data.columns.values[-1],
                    (ntimes-1) * tmul + 1)

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])
k_ids = [p.value for p in model.parameters_rules()]
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rate_mask])
bounds_radius = 2

solver = ScipyOdeSimulator(model, tspan)
obs_names = ['ERK_p_obs', 'p70S6K_p_obs']
obs_totals = [model.parameters['ERK_1_0'].value, model.parameters['p70S6K_0_0'].value]


def display(position):
    Y=np.copy(position)
    param_values[rate_mask] = 10 ** Y
    sim = solver.run(param_values=param_values)

    plt.plot(data.columns.values, erk2_norm, color='r', marker='.', linestyle=':')
    plt.plot(tspan, sim.all['ERK_p_obs'] / obs_totals[0], color='r')

    plt.plot(data.columns.values, p70s6k_norm, color='b', marker='.', linestyle=':')
    plt.plot(tspan, sim.all['p70S6K_p_obs'] / obs_totals[1], color='b')
    plt.show()

def likelihood(position):
    Y = np.copy(position)
    param_values[rate_mask] = 10 ** Y
    sim = solver.run(param_values=param_values).all
    for obs_name, obs_total in zip(obs_names, obs_totals):
        ysim = sim[obs_name][::tmul]
        ysim_norm = ysim / obs_total
        if obs_name == 'ERK_p_obs':
            e1 = np.sum((erk2_norm - ysim_norm) ** 2 / (2 * erk2_var)) / len(erk2_norm)
        else:
            e2 = np.sum((p70s6k_norm - ysim_norm) ** 2 / (2 * p70s6k_var)) / len(p70s6k_norm)
    error = e1 + e2
    return error,


def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(xnominal)
    pso.set_bounds(2)
    pso.set_speed(-.25, .25)
    pso.run(25, 100)
    display(pso.best)


if __name__ == '__main__':
    run_example()
