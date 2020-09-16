import numpy as np
import scipy.interpolate as sinter
from pysdot import PowerDiagram
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall


from Energy import Energy
from Trajectories import Trajectories
from Trajectories import Situation_Init

domain = ConvexPolyhedraAssembly()
domain.add_box([-1,-1], [10, 10])

point_infinity_1 = np.array([10, 6])
center_circ=[6,6]
smoothing=0.01
def potential_circ(x):
    radius=x-center_circ
    smoothed=np.sum(radius**2)
    result=np.zeros(3)
    result[0]=(smoothed-9)**2
    result[1:]=(smoothed-9)*4*radius
    result/=5
    return result

def potential_infinity(x):
    radius=x-point_infinity_1
    result=np.zeros(3)
    result[0]=np.sum(radius**2)
    result[1:]= radius*2
    return result

def potential_total(x):
    pot_inf=potential_infinity(x)
    pot_c=potential_circ(x)
    return pot_inf+pot_c


nb_double = 6
time = 15
nb_players_x=20
nb_players_y=20
nb_players=nb_players_x*nb_players_y
dimension=2
mass=10*np.ones(nb_players)/nb_players
epsilon=0.01
radial_func=RadialFuncInBall()
center=None


positions_init=np.random.rand(nb_players, 2)
positions_init=4*positions_init

#X,Y = np.meshgrid(np.linspace(0,3.5,nb_players_x),
#                  np.linspace(0,3.5,nb_players_y))
#positions_init = np.zeros((nb_players,2))
#positions_init[:,0] = X.flatten()
#positions_init[:,1] = Y.flatten() 

situation_init=Situation_Init(positions=positions_init, mass=mass)
initial_guess=Trajectories(situation_init=situation_init)


energy=Energy(domain=domain, radial_func=RadialFuncInBall(), epsilon=epsilon, time=time, lagrangian=lambda x:3*np.array([np.sum(x**2)/2,x[0],x[1]]), potential_t=potential_circ, potential_fin=potential_infinity)
optimal_trajectories=energy.minimize(initial_guess=initial_guess, center=center, nb_steps=2**nb_double, steps_global= 2**nb_double, mode='double_middle', nb_procs=30, error=1e-8)


root="article_{}p_con_5".format(nb_players)
optimal_trajectories.display_trajectories(root=root, domain=domain, radial_func=radial_func)
print("results stored in ",root)


import matplotlib.pyplot as plt
positions = np.load(root + "/alltrajectories.npy")

ee = .01

def show_contour(ax):
    ax.plot([-1,10,10,-1,-1],
             [-1,-1,10,10,-1],'k', linewidth=6)
    ax.set_axis_off()
    ax.set_xlim([-1-ee,10+ee])
    ax.set_ylim([-1-ee,10+ee])
    ax.set_aspect('equal')

def save_fig(name):
    plt.savefig(name, bbox_inches='tight',dpi=fig.dpi)
    
fig = plt.figure(figsize=(12,12))
bb = [-1-ee,-1-ee,10+ee,10+ee]
ax = fig.add_axes(bb)
for i in range(nb_players):
    plt.plot(positions[:,i,0], positions[:,i,1],'k',alpha=0.7)
show_contour(ax)
save_fig(root+"/all-trajectories.png")