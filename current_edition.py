from math import ceil
from typing import Iterator, Tuple
import numpy as np
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models.poisson import laplace, mass
from skfem import MeshTri, Basis, ElementTriP2
from skimage.data import binary_blobs
from nanomesh import Image
from skfem import MeshTri
from skfem.helpers import dot, grad
from skfem.io.json import from_file

class Mesh()

    def __init__(self, length=100, volume_fraction=0.1, seed=2102):

        #setup for the mesh
        self.length=length
        self.volume_fraction = volume_fraction
        self.blobs=binary_blobs(length=self.length, volume_fraction=self.volume_fraction,seed=seed)

        self.plane=Image(blobs)
        #generating the mesh from the plane
        self.sk_mesh = self.plane.generate_mesh(opts='q30a10')
        self.sk_mesh.points= (self.sk_mesh.points) / self.length / 2 - 1


    def get_dof_coordinate(self):
        '''
            Comparing mesh:
            Taking full mesh and comparing DOF location with mesh without the blobs in them.
            Then saving the corresponding index in full mesh. This would signify the lithium particle location
        '''

        # importing mesh into skfem
        triangles = sk_mesh.get('triangle')
        p  = triangles.points.T
        t = triangles.cells.T

        m = MeshTri(p, t)
        m=m.with_boundaries({"l": lambda x: x[1] == -1,"r": lambda x:x[0]==1})
        e = ElementTriP1()
        basis = Basis(m, e)

        #now removing the lithium particles
        triangles.remove_cells(label=2)

        #generating new mesh without the particles, and corressponfing basis
        p1 = triangles.points.T
        t1 = triangles.cells.T
        m1 = MeshTri(p1, t1)
        basis1 = Basis(m1, e)

        #getting the x and y locations of each DOF for the basis for each mesh
        x_basis=basis.doflocs[0]
        y_basis=basis.doflocs[1]
        x_basis1=basis1.doflocs[0]
        y_basis1=basis1.doflocs[1]

        #stacking them so that each elemnt represents a specific point.
        A = np.dstack((x_basis, y_basis))
        A = A[0]
        B = np.dstack((x_basis1, y_basis1))
        B = B[0]

        return A, B

    def compare_mesh(self, A, B):
        '''
            Looping through all elements in A. being the list of locations of each DOF for the large mesh.
            Then for each coupled point, loop through each location in B. If the location is the same then do nothing, if 
            they are different add it to a list.
            At the end of the second loop we check if the length of list2 is the same as the full length of B (meaning that that DOF does not share location with
            B) so that means its a particle.
        '''
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



        x_locs=[]
        y_locs=[]
        triangles=sk_mesh.get("triangle")
        return triangles

    def create_json(self, triangles)
        '''
            Creating the Json file for the mesh with subdomains lithium and electrolyte.
        '''
        #this is probabat the problm as the triangls is not based on the mesh but on the mesh.
        #so p and t must be based on the basis and not the mesh itself.
        p=triangles.points
        t=basis.element_dofs.T
        json_object={}

        json_object["p"]=p.tolist()
        json_object["t"]=t.tolist()

        elec_list=[]
        i=-1
        #getting indexes for the electrolytes.
        for ele in list(range(len(basis.doflocs[0]))):
            if ele not in list1:
                elec_list.append(ele)
        boundaries=basis.get_dofs().nodal["u"].tolist()
        #boundaries of mesh
        #putting everything in the json dictionary.
        json_object["subdomains"]={"electrolyte":elec_list,"lithium":list1}
        json_object["boundaries"]={"perimeter":boundaries}
        import json
        with open('mesh10.json', 'w') as fp:
            json.dump(json_object, fp)




class Solver():

    '''
        FEM starts.
    '''

    def __init__(self, dlithium=0, delec=0.01. dt = 0.1)


        #diffusion dictionary
        self.Diffusivity_coefficient={"lithium":dlithium,"electrolyte":delec}
        #time intervals
        self.dt = dt


    def import_mesh(self, fname="mesh10.json")

        #importing mesh
        self.mesh2 = from_file(fname)
        self.basis2=Basis(mesh2,e)

        self.basis0 = basis2
        self.diffusivity = basis0.zeros()
        #for s in mesh2.subdomains:

        #   for ele in mesh2.subdomains[s]:
        #      print(mesh2.subdomains[s])
            #    diffusivity[ele] = Diffusivity_coefficient[s]

            #putting the diffusion coefficients

        #getting diffusivity array with corresponding indexes for elements that are in lithium or electrolyte
        for s in self.mesh2.subdomains:
            self.diffusivity[self.basis0.get_dofs(elements=s)] = self.Diffusivity_coefficient[s]

    @BilinearForm
    def laplace1(u,v,w):
        return dot(w["diffusivity"]*grad(u),grad(v))
    #diffusivity=basis0.interpolate(diffusivity)

    def assemble_system(self, theta=0.5)

        #assembly of FEM
        self.L0 = asm(laplace1, self.basis2, diffusivity=self.basis0.interpolate(diffusivity))
        sel.M0 = asm(mass, basis2)

        #time stepping with crank nicholson scheme
        
        A = self.M0 + theta * self.L0 * self.dt
        B = self.M0 - (1 - theta) * self.L0 * self.dt

        self.backsolve = splu(A.T).solve

    def init_system(self)
        #initial condition and boundary condition of having 200 units at the bottom of the domain.
        u_init=np.zeros(len(basis.doflocs.prod(0)))
        for ele in basis.get_dofs("l").nodal['u']:
            u_init[ele] = 200.

    #each iteration going through crank nicholson scheme and root finding.
    def evolve(self, t: float,
            u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
        while t <100000:
            for ele in basis.get_dofs("l").nodal['u']:
                u[ele]=200

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