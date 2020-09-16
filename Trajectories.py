import numpy as np 
import os
from multiprocessing import Process
from multiprocessing import Manager
from pysdot import PowerDiagram
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from OTransport import proj_noncongested

class Situation_Init:
    def __init__(self, positions, mass=None):
        self.positions=positions
        if mass is None:
            nb_players=len(positions)
            self.mass=np.ones(nb_players)/nb_players
        else:
            assert(positions.shape[0]==mass.shape[0])
            self.mass=mass

    def get_positions(self):
        return self.positions
    
    
    def get_nb_players(self):
        return self.positions.shape[0]


    def get_dimension(self):
        return self.positions.shape[1]


    def get_mass(self):
        return self.mass


class Trajectories:

    def __init__(self, situation_init=None, trajectories=None, mass=None):

        if situation_init is None:
            self.situation_init=Situation_Init(positions=trajectories[0], mass=mass)
            self.trajectories_t=trajectories[1:]

        else: 
            assert(isinstance(situation_init, Situation_Init))
            self.situation_init=situation_init
            if trajectories is None:
                self.trajectories_t=np.array([situation_init.get_positions()])
            else:
                self.trajectories_t=trajectories
        

    def get_sit_init(self):

        return self.situation_init
    

    def get_trajectories_0(self):

        return self.situation_init.get_positions()

    
    def get_trajectories_t(self):

        return self.trajectories_t


    def set_trajectories_t(self, trajectories_t):

        self.trajectories_t[:]=trajectories_t


    def get_dimensions(self):

        steps, players, dim=self.trajectories_t.shape
        return steps+1, players, dim


    def get_full_trajectories(self):

        total_shape=self.get_dimensions()
        result=np.zeros(total_shape)
        result[0,:,:]=self.get_trajectories_0()
        result[1:,:,:]=self.get_trajectories_t()
        return result


    def get_mass(self):

        return self.situation_init.get_mass()


    def increase_nb_steps(self, mode='double_static', **kwargs):
    
        trajectories=self.get_full_trajectories()
        nb_steps, nb_players, dimension=trajectories.shape
        
        if mode == 'increment':
            new_trajectories = np.zeros((nb_steps+1, nb_players, dimension))
            new_trajectories[:-1,:,:]=trajectories[:,:,:]
            new_trajectories[-1,:,:]=trajectories[-1,:,:]
        
        if mode == 'double_static':
            new_trajectories = np.zeros((2 * (nb_steps)-1, nb_players, dimension))
            new_trajectories[0::2, :, :] = trajectories[:, :, :]
            new_trajectories[1::2, :, :] = trajectories[:-1, :, :]
        
        if mode == 'double_middle':
            new_trajectories = np.zeros((2 * (nb_steps)-1, nb_players, dimension))
            new_trajectories[0::2, :, :] = trajectories[:, :, :]
            new_trajectories[1::2, :, :] = 0.5*(trajectories[:-1, :, :]+trajectories[1:,:,:])
        
        if mode == 'double_barycenters':
            new_trajectories = np.zeros((2 * (nb_steps), nb_players, dimension))
            new_trajectories[0::2, :, :] = trajectories[:, :, :]
            i=0
            for position in trajectories[:]:
                laguerre=proj_noncongested(points=position, domain=kwargs['domain_double'], radial_func=kwargs['radial_func_double'], mass=kwargs['mass_double'])
                new_trajectories[2*i+1, :, :] = laguerre.centroids()
                i+=1

        print("Nb_steps=", new_trajectories.shape[0])
        self.trajectories_t=new_trajectories[1:]
    

    def display_trajectories(self, root, domain, radial_func, nb_procs=6):
        trajectories=self.get_full_trajectories()
        nb_steps = trajectories.shape[0]
        os.makedirs(root+"/trajectories/", exist_ok=True)
        domain.display_boundaries_vtk(root+"/boundaries_domain.vtk")

        def projection_slice(positions,t_init):
            
            for index, pos in enumerate(positions):               

                laguerre = proj_noncongested(points=pos, mass=self.get_mass(), domain=domain, radial_func=radial_func)
                laguerre.display_vtk(root+"/trajectories/trajectories_time%d.vtk" % (t_init+index),
                             points=True, centroids=False)
            

        #creating the processes for the intermediate slices
        q=nb_steps//nb_procs
        r=nb_steps-nb_procs*q

        procs_q=[Process(target=projection_slice, 
                        args=(trajectories[n*q:(n+1)*q],n*q)) 
                for n in range(nb_procs)]

        for p in procs_q:
            p.start()
        for p in procs_q:
            p.join()
        


        #creating the processes for the last positions
        procs_r=[Process(target=projection_slice,
                        args=([trajectories[nb_procs*q+n],],nb_procs*q+n)) 
                                for n in range(r)]
        for p in procs_r:
            p.start()
        for p in procs_r:
            p.join()
        

        
        np.save(root+"/alltrajectories",trajectories)