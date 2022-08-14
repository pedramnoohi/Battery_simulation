from math import ceil
from typing import Iterator, Tuple
import numpy as np
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models.poisson import laplace, mass
from skfem import MeshTri, Basis, ElementTriP2
from skimage.data import binary_blobs
from nanomesh import Image
#setup for the mesh
length=100
blobs=binary_blobs(length=length, volume_fraction=0.20,seed=2102)
plane=Image(blobs)
#generating the mesh from the plane
sk_mesh = plane.generate_mesh(opts='q30a10')
sk_mesh.points= (sk_mesh.points) / length / 2 - 1

'''
    Comparing mesh:
    Taking full mesh and comparing DOF location with mesh without the blobs in them.
    Then saving the corresponding index in full mesh. This would signify the lithium particle location
'''

# importing mesh into skfem
from skfem import MeshTri
triangles = sk_mesh.get('triangle')
p = triangles.points.T
t = triangles.cells.T
lithium_list =[]
lyte_list = []
f = -1
for ele in triangles.cell_data["physical"]:
    f = f + 1
    if ele == 2:
        lithium_list.append(f)
    else:
        lyte_list.append(f)
lithium_list=np.array(lithium_list)
lyte_list=np.array(lyte_list)
m = MeshTri(p, t,_subdomains={"lithium":lithium_list,"electrolyte":lyte_list})
m=m.with_boundaries({"l": lambda x: x[1] == -1,"r": lambda x:x[0]==1})
e = ElementTriP1()
basis = Basis(m, e)

from skfem.helpers import dot, grad

#diffusion dictionary
Diffusivity_coefficient={"lithium":0.000001 ,"electrolyte":0.05}
#time intervals
dt = 0.01

basis0 = basis.with_element(ElementTriP1())
diffusivity = basis0.zeros()
diffusivity = basis.zero_w()
for subdomain, elements in m.subdomains.items():
    diffusivity[elements] = Diffusivity_coefficient[subdomain]
#for s in m.subdomains:
   # diffusivity[basis0.get_dofs(elements=s)] = Diffusivity_coefficient[s]

@BilinearForm
def laplace1(u,v,w):
    return dot(w["diffusivity"]*grad(u),grad(v))
#diffusivity=basis0.interpolate(diffusivity)


#assembly of FEM
L0 = asm(laplace1, basis,diffusivity=diffusivity)
M0 = asm(mass, basis)

#time stepping with crank nicholson scheme
theta = 0.5
lhs = M0 + theta * L0 * dt
rhs = M0 - (1 - theta) * L0 * dt
print(type(lhs))
backsolve = splu(lhs.T).solve
#initial condition and boundary condition of having 200 units at the bottom of the domain.
u_init=np.zeros(len(basis.doflocs.prod(0)))
for ele in basis.get_dofs("l").nodal['u']:
    u_init[ele] = 200.
#each iteration going through crank nicholson scheme and root finding.
def evolve(t: float,
           u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    while t <100000:
        for ele in basis.get_dofs("l").nodal['u']:
            u[ele]=200

        t, u = t + dt, backsolve(rhs @ u)
        for ele in basis.get_dofs("l").nodal['u']:
            u[ele]=200
        yield t, u

if __name__ == '__main__':

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    from skfem.visuals.matplotlib import plot

    parser = ArgumentParser(description='heat equation in a rectangle')
    parser.add_argument('-g', '--gif', action='store_true',
                        help='write animated GIF', )
    args = parser.parse_args()
    '''
        CHECK THIS
    '''
    ax = plot(m, u_init, shading='gouraud')
    title = ax.set_title('t = 0.00')
    field = ax.get_children()[0]  # vertex-based temperature-colour
    fig = ax.get_figure()
    fig.colorbar(field)

    def update(event):
        t, u = event
        title.set_text(f'$t$ = {t:.2f}')
        field.set_array(u)

    animation = FuncAnimation(
        fig,
        update,
        evolve(0., u_init),
        repeat=False,
        interval=50,
    )
    if args.gif:
        animation.save(Path(__file__).with_suffix('.gif'), 'imagemagick')
    else:
        plt.show()