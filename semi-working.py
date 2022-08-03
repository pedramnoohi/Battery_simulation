
from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu

from skfem import *
from skfem.models.poisson import laplace, mass
from skfem import MeshTri, Basis, ElementTriP2
from skimage.data import binary_blobs
from nanomesh import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
length=100
blobs=binary_blobs(length=length, volume_fraction=0.20,seed=2102)
plane=Image(blobs)

mesh = plane.generate_mesh(opts='q30a10')
mesh.points=(mesh.points)/length/2  -1

'''
    Comparing mesh
'''
from skfem import MeshTri
triangles = mesh.get('triangle')
p = triangles.points.T
t = triangles.cells.T

m = MeshTri(p, t)
m=m.with_boundaries({"l": lambda x: x[1] == -1,"r": lambda x:x[0]==1})
e = ElementTriP1()
basis = Basis(m, e)

triangles.remove_cells(label=2)

p1 = triangles.points.T
t1 = triangles.cells.T
m1 = MeshTri(p1, t1)
basis1 = Basis(m1, e)

x_basis=basis.doflocs[0]
y_basis=basis.doflocs[1]
x_basis1=basis1.doflocs[0]
y_basis1=basis1.doflocs[1]
A = np.dstack((x_basis, y_basis))
A = A[0]
B = np.dstack((x_basis1, y_basis1))
B = B[0]
s = -1
list1 = []
for ele in A:
    s = s + 1
    n = -1
    list2 = []
    for elem in B:

        n = n + 1
        if np.array_equal(ele, elem) == True:
            pass
        else:
            list2.append(n)
    if len(list2) == len(B):
        list1.append(s)
print(len(list1))
print(len(x_basis1)-len(x_basis))
from skfem.helpers import dot, grad

@BilinearForm
def laplace1(u,v,_):
    return dot(grad(u),grad(v))
L0 = asm(laplace, basis)
M0 = asm(mass, basis)




[leng,leng]=L0.shape
D_lith=0.00001
D_ele=0.001
L0=L0.todense()
for ele in range(leng):
    for elem in range(leng):
        if ele in list1 and ele==elem:
            L0[ele, ele]=L0[ele,ele]*D_lith
        elif ele in list1 and elem in list1:
            L0[ele,elem]=L0[ele,elem]*D_lith
            L0[elem,ele]=L0[elem,ele]*D_lith
        elif ele in list1 or elem in list1:
            L0[ele,elem]=L0[ele,elem]*(D_lith+D_ele)/2
            L0[elem,ele]=L0[elem,ele]*(D_lith+D_ele)/2
        else:
            L0[ele,elem]=L0[ele,elem]*D_ele
            L0[elem,ele]=L0[elem,ele]*D_ele
L0=scipy.sparse.csr_matrix(L0)
dt = 10

theta = 0.5
A = M0 + theta * L0 * dt
B = M0 - (1 - theta) * L0 * dt

backsolve = splu(A.T).solve
u_init=np.zeros(len(basis.doflocs.prod(0)))
for ele in basis.get_dofs("l").nodal['u']:
    u_init[ele] = 200.

def evolve(t: float,
           u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    while t <100000:


        t, u = t + dt, backsolve(B @ u)
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