#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class HYBRID(Optimizer):


    def __init__(self, epoch: int = 10000, pop_size: int = 100, a_factor: int = 10, R_factor: float = 1.5,
                 alpha: float = 2.0, c1: float = 2.0, c2: float = 2.0, **kwargs: object) -> None:

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.a_factor = self.validator.check_int("a_factor", a_factor, [2, 20])
        self.R_factor = self.validator.check_float("R_factor", R_factor, [0.1, 3.0])
        self.alpha = self.validator.check_float("alpha", alpha, [0.5, 3.0])
        self.c1 = self.validator.check_float("c1", c1, (0, 4.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 4.0))
        self.set_parameters(["epoch", "pop_size", "a_factor", "R_factor", "alpha", "c1", "c2"])
        self.sort_flag = False

    def create_x_y_x1_y1__(self):
        """ Using numpy vector for faster computational time """
        ## Eq. 2
        phi = self.a_factor * np.pi * self.generator.uniform(0, 1, self.pop_size)
        r = phi + self.R_factor * self.generator.uniform(0, 1, self.pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)
        ## Eq. 3
        r1 = phi1 = self.a_factor * np.pi * self.generator.uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)
        x_list = xr / np.max(xr)
        y_list = yr / np.max(yr)
        x1_list = xr1 / np.max(xr1)
        y1_list = yr1 / np.max(yr1)
        return x_list, y_list, x1_list, y1_list

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## 0. Pre-definded
        x_list, y_list, x1_list, y1_list = self.create_x_y_x1_y1__()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([agent.solution for agent in self.pop])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.g_best.solution + self.alpha * self.generator.uniform() * (pos_mean - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        ## 2. Search in space
        pos_list = np.array([agent.solution for agent in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        pop_child = []
        for idx in range(0, self.pop_size):
            idx_rand = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            pos_new = self.pop[idx].solution + y_list[idx] * (self.pop[idx].solution - self.pop[idx_rand].solution) + \
                      x_list[idx] * (self.pop[idx].solution - pos_mean)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)

        ## 3. Swoop
        pos_list = np.array([agent.solution for agent in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            # -1 < E0 < 1
            E0 = 2 * self.generator.uniform() - 1
            # factor to show the decreasing energy of rabbit
            E = 2 * E0 * (1. - epoch * 1.0 / self.epoch)
            J = 2 * (1 - self.generator.uniform())
            delta_X = self.g_best.solution - self.pop[idx].solution

            if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                HHO = delta_X - E * np.abs(J * self.g_best.solution - self.pop[idx].solution)

                BES = self.generator.uniform() * self.g_best.solution + x1_list[idx] * (
                            self.pop[idx].solution - self.c1 * pos_mean) \
                          + y1_list[idx] * (self.pop[idx].solution - self.c2 * self.g_best.solution)

                pos_new = 0.5 * (HHO + BES)  # PROPOSED EQUATION
            else:  # Soft besiege Eq. (4) in paper
                HHO = self.g_best.solution - E * np.abs(delta_X)
        
                BES = self.generator.uniform() * self.g_best.solution + x1_list[idx] * (
                            self.pop[idx].solution - self.c1 * pos_mean) \
                          + y1_list[idx] * (self.pop[idx].solution - self.c2 * self.g_best.solution)
                pos_new = 0.5 * (HHO + BES)  # PROPOSED EQUATION

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
