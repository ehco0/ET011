import numpy as np 

from scipy.stats import qmc 

N = 9 

bounds = {
    'Aspect': (3, 20), 
    'Taper': (0.1, 1.2), 
    'Re': (1000000, 25000000), 
    'Sweep': (-10,45),
    'Dihedral': (-10, 10), 
    'Twist': (-2, 8)
    
} 

airfoil_choices = ['22112'] * 7 + ['22116'] * 2 
np.random.shuffle(airfoil_choices) 

sampler = qmc.LatinHypercube(d=len(bounds))
unit_samples = sampler.random(n=N) 

l_bounds = np.array([v[0] for v in bounds.values()]) 
u_bounds = np.array([v[1] for v in bounds.values()]) 

samples = qmc.scale(unit_samples, l_bounds, u_bounds)
param_names = list(bounds.keys()) 

print('Generated Designs:\n') 

for i, row in enumerate(samples):
    print(f'Design {i+1}:')
    for name, value in zip(param_names, row):
        print(f'  {name}: {value:.3f}')
        print(f'  Airfoil: {airfoil_choices[i]}')
        print() 