# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:49:05 2021

@author: csb
"""

import numba as nb
import numpy as np
import pde
import math as math
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid


class toggle_switch_PDE(PDEBase):
    """ toggle switch """

    def __init__(self, Diffusion=0.001, alpha=10, gamma=0.5,  bc=["NeumannBC","NeumannBC"]):
   
        self.alpha = alpha
        self.D = Diffusion
        self.bc = bc  # boundary condition
        self.gamma=gamma,

    def get_initial_state(self, grid):
        """ preparing initial state """
        #u = ScalarField.from_expression(grid, "0.04+100*exp(-x)")
        #v = ScalarField.from_expression(grid, "0.04+100*exp(x-10)")
        
        u = ScalarField(grid, 100, label="Field $u$")
        v = ScalarField(grid, 100, label="Field $v$")
        return FieldCollection([u, v])

    
    def evolution_rate(self, state, t=0):
        """ python implementation of the PDE """
        u, v = state
        rhs = state.copy()
        
        alp=self.alpha
        D=self.D
        gam=self.gamma
        rhs[0] = alp//(1+v**2) -gam*u   +  D*u.laplace('natural')
        
        rhs[1] = alp//(1+v**2) -gam*u   +  D*v.laplace('natural')
        return rhs


# initialize state
        
T = 500.0;
L=10;

grid = pde.CartesianGrid([[0,L], [0, L]], [50, 50])  # generate grid

eq = toggle_switch_PDE()
state = eq.get_initial_state(grid)

# simulate the pde
tracker = PlotTracker(interval=1, plot_args={"vmin": 0, "vmax": 30})
sol = eq.solve(state,t_range=T,dt=1e-4,tracker=tracker) 
sol.plot()