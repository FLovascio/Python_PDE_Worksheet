import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Coordinate:
    def __init__(self,points):
        self.points=points
        self.h=points[1]-points[0]

class Field:
    def __init__(self,f):
        self.y=f

    def set_field(self,new_field):
        self.y=new_field

    def get_field(self):
        return self.y

    def d_x(self,coordinate):
        '''
        takes the first derivative of self with respect to a Coordinate type variable.
        '''
        return np.concatenate((np.array([0.0]),(1.0/(2*coordinate.h))*(self.y[1:-1]-self.y[:-2]),np.array([0.0])))

    def d_xx(self,coordinate):
        return np.concatenate((np.array([0.0]),(1.0/(coordinate.h**2))*(self.y[2:]-2.0*self.y[1:-1]+self.y[:-2]),np.array([0.0])))

class PDE:
    '''
    A generic parabolic/hyperbolic class. Can be used as a parent class for specialised PDE solvers.

    Member variables:
    PDE.x -> The coordinate system on which the PDE is set up (type: Coordinate)
    PDE.f -> The field the PDE is advancing (type: Field)
    PDE.t -> Stores the point in time at which each step of the solution is at (type: [])
    PDE.solution -> Stores the solution to the PDE in time (type: [np.array])
    PDE.dt -> The time derivative function that defines the PDE (type: function)
    PDE.time_initial -> The time coordinate of the initial conditions
    PDE.tau -> The timestep size
    PDE.fieldName -> Name of the field, used for plotting (type: string)

    Member functions:
    PDE.set_initial_conditions(self,y0,t0=0.0) -> reset the initial conditions
    PDE.timestep_euler(self) -> forward Euler timestep method
    PDE.timestep(self) -> the timestepping method used inthe solver, forward Euler by default
    PDE.CFL(self) -> placeholder function for the CFL timestep calculator
    PDE.solve(self, time_final=0.0) -> advances the PDE by taking CFL steps until time_final is reached
    PDE.plot_full_solution(self,x_label="Position (m)",y_label="Time (s)") -> plots the solution
    '''
    def __init__(self,x,y0,dt=None,time_initial=0.0):
        self.dt=dt
        self.f=Field(y0)
        self.x=Coordinate(x)
        self.t=[time_initial]
        self.tau=0
        self.solution=[self.f.y]
        self.time_initial=time_initial
        self.fieldName="y"

    def set_initial_conditions(self,y0,t0=0.0):
        self.f.set_field(y0)

    def timestep_euler(self):
        self.f.set_field(self.f.get_field()+self.tau*self.dt())

    def timestep(self):
        self.timestep_euler()

    def CFL(self):
        pass

    def solve(self, time_final=0.0):
        while self.t[-1] < time_final:
            self.CFL()
            self.timestep()
            self.solution.append(self.f.y)
            self.t.append(self.t[-1]+self.tau)
        self.solution=np.stack(self.solution, axis=0)

    def plot_full_solution(self,x_label="Position (m)",y_label="Time (s)"):
        fig = plt.figure()
        plt.pcolormesh(self.x.points,self.t,self.solution)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        cbar = plt.colorbar()
        cbar.minorticks_on()
        cbar.set_label(self.fieldName)

class HeatEquation(PDE):
    def __init__(self,x,T_initial,time_initial=0.0,thermal_conductivity=1.0):
        super().__init__(x=x,y0=T_initial,time_initial=time_initial,dt=self.heat_equation)
        self.thermal_conductivity=thermal_conductivity
        self.T=self.f
        self.fieldName="Temperature (K)"

    def heat_equation(self):
        return self.thermal_conductivity*self.T.d_xx(self.x)

    def CFL(self):
        self.tau=0.5*(self.x.h**2)
        self.tau=self.tau/self.thermal_conductivity
        self.tau-=0.1*self.tau

class HeatConductionConvectionEquation(PDE):
    def __init__(self,x,T_initial,time_initial=0.0,thermal_conductivity=1.0,cs=1.0):
        super().__init__(x=x,y0=T_initial,time_initial=time_initial,dt=self.heat_equation)
        self.thermal_conductivity=thermal_conductivity
        self.T=self.f
        self.cs=cs
        self.fieldName="Temperature (K)"

    def heat_equation(self):
        return self.T.get_field()*self.cs*self.T.d_x(self.x)+self.thermal_conductivity*self.T.d_xx(self.x)

    def CFL(self):
        self.tau=0.5*(self.x.h**2)
        self.tau=self.tau/self.thermal_conductivity
        self.tau-=0.1*self.tau
