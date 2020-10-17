import numpy as np
import scipy.interpolate as sinter
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from agd import Eikonal


from Energy import Energy
from Trajectories import Trajectories
from Trajectories import Situation_Init

domain = ConvexPolyhedraAssembly()
domain.add_box([0,0], [8, 8])
domain.add_box([8, 3.5], [11, 4.5])
domain.add_box([11,0],[19,8])
eps_density=1e-3
domain.add_box(min_pos=[8,0], max_pos=[11,3.5], coeff=eps_density)
domain.add_box(min_pos=[8,4.5], max_pos=[11,8], coeff=eps_density)

point_infinity_1 = np.array([18, 7])
point_infinity_2=np.array([18,1])
hfmInput = Eikonal.dictIn({
    'model': 'Isotropic2',
    'order': 2.,
    'cost':1.
})
gridScale = 0.1
hfmInput.SetRect(sides=[[-0.5,19.5],[-0.5,8.5]], gridScale=gridScale)
hfmInput.update({'seeds': [point_infinity_1,point_infinity_2]})
hfmInput['arrayOrdering'] = 'RowMajor'
X,Y = hfmInput.Grid()
x_min=X[0,0]
y_min=Y[0,0]
velocity=np.ones_like(X)
boundaries = np.logical_and(abs(X - 9.5) < 1.5, abs(Y - 4) > 0.5)
velocity[boundaries]=1e-1
hfmInput.update({'speed':velocity})
hfmInput['exportValues'] = 1.
hfmInput['exportGeodesicFlow'] = 1
hfmOutput = hfmInput.Run()



Spline = sinter.SmoothBivariateSpline(X.flatten(),Y.flatten(), hfmOutput['values'].flatten(), s=10)
        
def potential_t(x):
    return np.zeros(3)

def potential_fin(x):
    return np.array([Spline.ev(x[0],x[1]), Spline.ev(x[0],x[1],dx=1),Spline.ev(x[0],x[1],dy=1)])



nb_double = 8
time = 600
nb_players_x=20 
nb_players_y=20
nb_players=nb_players_y*nb_players_x
dimension=2
mass=50*np.ones(nb_players)/nb_players
epsilon=.1
radial_func=RadialFuncInBall()
center=None

X,Y = np.meshgrid(np.linspace(0.3,7.7,nb_players_x),
                  np.linspace(0.3,7.7,nb_players_y))
positions_init = np.zeros((nb_players,2))
positions_init[:,0] = X.flatten()
positions_init[:,1] = Y.flatten()


situation_init=Situation_Init(positions=positions_init, mass=mass)
initial_guess=Trajectories(situation_init=situation_init)


energy=Energy(domain=domain, radial_func=radial_func, epsilon=epsilon, time=time, lagrangian=lambda x:4*np.array([np.sum(x**2)/2,x[0],x[1]]), potential_t=potential_t, potential_fin=potential_fin)
optimal_trajectories=energy.minimize(initial_guess=initial_guess, center=center, nb_steps=2**nb_double, steps_global= 2**nb_double, mode='double_middle', nb_procs=30, error=1e-8, solver="sopt.minimize")


root="Trajectories_non_convex"
optimal_trajectories.display_trajectories(root=root, domain=domain, radial_func=radial_func)
print("results stored in ",root)


import matplotlib.pyplot as plt
positions = np.load(root + "/alltrajectories.npy")
np.save(root+"/parameters",np.array([nb_double, time, nb_players, np.sum(mass), epsilon]))

ee = .01

def show_contour(ax):
    ax.plot([0,8,8,11,11,19,19,11,11,8,8,0,0],
             [0,0,3.5,3.5,0,0,8,8,4.5,4.5,8,8,0],'k', linewidth=6)
    ax.set_axis_off()
    ax.set_xlim([0-ee,19+ee])
    ax.set_ylim([0-ee,8+ee])
    ax.set_aspect('equal')

def save_fig(name):
    plt.savefig(name, bbox_inches='tight',dpi=fig.dpi)
    
fig = plt.figure(figsize=(20,9))
bb = [0-ee,0-ee,19+ee,8+ee]
ax = fig.add_axes(bb)
for i in range(nb_players):
    plt.plot(positions[:,i,0], positions[:,i,1],'k',alpha=1)
show_contour(ax)
save_fig(root+"/all-trajectories.png")



