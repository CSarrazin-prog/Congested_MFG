import numpy as np
import scipy.optimize as sopt
from multiprocessing import Process
from multiprocessing import Manager
from pysdot import PowerDiagram
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from Hard_congestion import dist_noncongested
from Hard_congestion import proj_noncongested
from Trajectories import Trajectories
import os

class Energy:
    def __init__(self, domain, radial_func= None, epsilon=1, time=1, lagrangian=lambda x:np.array([np.sum(x**2)/2,x[0],x[1]]), potential_t=None, potential_fin=None):
        self.lagrangian=lagrangian
        self.time=time
        self.radial_func=radial_func
        self.domain=domain
        self.epsilon=epsilon
        self.potential_t=potential_t
        self.potential_fin=potential_fin
        
    
    def compute(self, situation:Trajectories, steps_global=None, center=None, nb_procs=6, verbose=False):
        nb_steps, nb_players, dim=situation.get_dimensions()
        trajectories=situation.get_full_trajectories()
        mass=situation.get_mass()
        if steps_global is None:
            delta=self.time/nb_steps
        else:
            delta=self.time/steps_global
        

        def energy_slice(positions, keys, return_dict, final=False):
            energ=0
            grad=np.zeros_like(positions)

            
            velocity=(positions[1:]-positions[:-1])/delta
            lag_vect=mass.reshape((-1,1))*np.apply_along_axis(func1d=self.lagrangian, axis=2, arr=velocity)
            energ+=delta*np.sum(lag_vect[:,:,0])
            grad[:-1,:,:]+=-lag_vect[:,:,1:]
            grad[1:,:,:]+=lag_vect[:,:,1:]
            
            
            potential = self.potential_fin if final else self.potential_t
            pot_vec=(1 if final else delta)*mass.reshape((-1,1))*np.apply_along_axis(func1d=potential, axis=2, arr=positions[1:])
            energ+=np.sum(pot_vec[:,:,0])
            grad[1:,:,:]+=pot_vec[:,:,1:]

            for time, points in enumerate(positions[1:]):                
                con_val, con_grad=dist_noncongested(points=points, domain=self.domain, center=center, epsilon=self.epsilon, mass=mass, radial_func=self.radial_func, verbose=keys[0])
                energ+=(1. if final else delta)*con_val
                grad[time+1,:,:]+=(1. if final else delta)*con_grad

            return_dict[keys[0]]=energ
            return_dict[keys[1]]=grad


        #creating the processes for the intermediate slices:
        ener_e=0
        ener_grad=np.zeros_like(trajectories)


        q=(nb_steps-2)//nb_procs
        r=nb_steps-2-nb_procs*q


        manager = Manager()
        return_dict = manager.dict()

        if q!=0:
            procs_q=[Process(target=energy_slice, 
                            args=(trajectories[n*q:(n+1)*q+1], 
                                    ["Energy_{}".format(n), "Gradient_{}".format(n)], return_dict)) 
                    for n in range(nb_procs)]
            for p in procs_q:
                p.start()
            for p in procs_q:
                p.join()
            for n in range(nb_procs):
                ener_e+=return_dict["Energy_{}".format(n)]
                ener_grad[n*q:(n+1)*q+1]+=return_dict["Gradient_{}".format(n)]


        #creating the processes for the last slices:
        procs_r=[Process(target=energy_slice,
                        args=(trajectories[nb_procs*q+n:nb_procs*q+n+2], 
                                ["Energy_rest_{}".format(n), "Gradient_rest_{}".format(n)], return_dict, (n==r))) 
                for n in range(r+1)]

        for p in procs_r:
            p.start()
  
        for p in procs_r:
            p.join()
        
        for n in range(r+1):
            ener_e+=return_dict["Energy_rest_{}".format(n)]
            ener_grad[nb_procs*q+n:nb_procs*q+n+2]+=return_dict["Gradient_rest_{}".format(n)]

        return [ener_e, ener_grad[1:]]
    

    def minimize(self, initial_guess:Trajectories, nb_steps, mode, steps_global=None, center=None, nb_procs=6, error=1e-6, solver="sopt.minimize"):
        situation_init=initial_guess.get_sit_init()
        nb_players, dimension=initial_guess.get_dimensions()[1:]
        opt_traj=initial_guess
        kwargs={}
        
        if not steps_global is None:
            nb_steps=steps_global
        
        if mode=='double_barycenters':
            kwargs.update({'domain_double':self.domain, 'radial_func_double':self.radial_func, 'mass_double':initial_guess.get_mass()})
        
        trajectories_flat=initial_guess.get_trajectories_t().flatten()
        
        def functional(traj_flat):
                trajectories=Trajectories(situation_init=situation_init, trajectories=traj_flat.reshape([-1,nb_players,dimension]))
                ener_e,ener_grad=self.compute(situation=trajectories, steps_global=steps_global, center=center, nb_procs=nb_procs)
                return ener_e, ener_grad.flatten()
            
            
        if solver=="sopt.minimize":
            options={'iprint': 10, 'maxfun': 5000, 'maxiter': 5000, 'maxls':50}
            lb=self.domain.min_position()
            up=self.domain.max_position()
            bounds_tab=np.zeros([trajectories_flat.shape[0],2])
            bounds_tab[0::2,0]=lb[0]
            bounds_tab[0::2,1]=up[0]
            bounds_tab[1::2,0]=lb[1]
            bounds_tab[1::2,1]=up[1]
            bounds=sopt.Bounds(bounds_tab[:,0],bounds_tab[:,1])
            
            
            trajectories_flat= (sopt.minimize(fun=functional, x0=trajectories_flat, method='L-BFGS-B', jac=True, bounds=bounds, tol=error
                                       , options=options)).x
            opt_traj.set_trajectories_t(trajectories_flat.reshape([-1,nb_players,dimension]))
            steps_done=opt_traj.get_dimensions()[0]
            
            while steps_done<nb_steps:
                opt_traj.increase_nb_steps(mode=mode, **kwargs)
                trajectories_flat=opt_traj.get_trajectories_t().flatten()
                
                bounds_tab=np.zeros([trajectories_flat.shape[0],2])
                bounds_tab[0::2,0]=lb[0]
                bounds_tab[0::2,1]=up[0]
                bounds_tab[1::2,0]=lb[1]
                bounds_tab[1::2,1]=up[1]
                bounds=sopt.Bounds(bounds_tab[:,0],bounds_tab[:,1])
                
                
                trajectories_flat= (sopt.minimize(fun=functional, x0=trajectories_flat, method='L-BFGS-B', jac=True, bounds=bounds,tol=error
                                       , options=options)).x
                opt_traj.set_trajectories_t(trajectories_flat.reshape([-1,nb_players,dimension]))
                steps_done=opt_traj.get_dimensions()[0]-1
            
            return opt_traj
        
        else:
            TODO