import numpy as np
from typing import Iterator, Tuple
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models.poisson import mass
from skfem.helpers import dot, grad
import skfem
import scipy

class FEM:
    '''
    Does finite element method on unstructured mesh for diffusion equation.

    Args:
        dt(float): time steps
        t_max(float): final time
        initial_temp(float): temperature at bottom boundary
        mesh(skfem.mesh.mesh_tri_1.MeshTri1): mesh of domain
        basis(skfem.assembly.basis.cell_basis.CellBasis): basis of assembly
        Diffusion_coefficient(dict): dictionary including diffusion coefficient for subdomains
    Attributes:
        dt(float): where time steps is stored
        t_max(float): where final time is stored
        initial_temp(float): where temperature at bottom boundary is stored
        m(skfem.mesh.mesh_tri_1.MeshTri1): where mesh of domain is stored
        basis(skfem.assembly.basis.cell_basis.CellBasis): where basis of assembly is stored
        Diffusion_coefficient(dict):where dictionary including diffusion coefficient for subdomains is stored
    '''

    def __init__(self, dt: float, t_max: float, initial_temp: float, mesh: skfem.mesh.mesh_tri_1.MeshTri1
                 , basis: skfem.assembly.basis.cell_basis.CellBasis, Diffusivity_coefficient: dict):

        self.Diffusivity_coefficient = Diffusivity_coefficient
        self.m = mesh
        self.basis = basis
        self.dt = dt
        self.t_max = t_max
        self.initial_temp = initial_temp

    def diffusivity(self) -> np.ndarray:
        """
        Creates a diffusivity array that represents each index with its own coefficient given its subdomain
        Returns:
            diffusivity(np.ndarray): returns diffusivity array
        """
        diffusivity = self.basis.zero_w()
        for subdomain, elements in self.m.subdomains.items():
            diffusivity[elements] = self.Diffusivity_coefficient[subdomain]
        return diffusivity

    def assembly(self, type: int) -> scipy.sparse._csr.csr_matrix:
        """
        Assembles the matrices for FEM given diffusion equation and basis.
        Args:
            type(int): a variable that tells method which matrix to return,A or B
        Returns:
            lhs(scipy.sparse._csr.csr_matrix): next time step
            rhs(scipy.sparse._csr.csr_matrix): this time step matrx
        """
        self.type = type

        @BilinearForm
        def laplace1(u, v, w):
            """
            Takes test function v and function u with and sets up laplacian matrix
            Args:
                u():Our concentration function
                v():test function
            Returns:
                weak_form(np.ndarray): laplacian matrix with altering diffusion coefficent given subdomain

            """

            weak_form=dot(w["diffusivity"] * grad(u), grad(v))
            return weak_form

        @BilinearForm
        def drift(u,v,w):
            """
            Drift component
            Args:
                u():Our concentration function
                v():test function
            Returns:
                weak_form_drift(np.ndarray): laplacian matrix with altering diffusion coefficent given subdomain

            """
            _,y=w.x
            weak_form_drift=v*1*grad(u)[1]
            return weak_form_drift
        T0=asm(drift,self.basis)
        L0 = asm(laplace1, self.basis, diffusivity=self.diffusivity())
        M0 = asm(mass, self.basis)
        theta = 0.5
        lhs = M0 + theta * (L0+T0) * self.dt
        rhs = M0 - (1 - theta) * (L0+T0) * self.dt
        if type == 0:
            return lhs
        if type == 1:
            return rhs

    def initial_condition(self) -> np.ndarray:
        """
        Function gets initial condition based on the initial temperature that was provided by user
        Returns:
            u_init(np.ndarray): array with each element being index of basis with correct initial condition depending on location

        """
        u_init = np.zeros(len(self.basis.doflocs.prod(0)))

        for ele in self.basis.get_dofs("l").nodal['u']:
            u_init[ele] = self.initial_temp
        return u_init

    def frame(self, t: float,
              u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
        """
        Gets value for each dof at time t
        Args:
            t(float):time
            u(float):value of previous times for each dof
        Yields:
            t(float): time+dt
            u(np.ndarray):value of points at next time point for each dof
        """
        self.t = t
        self.u = u
        backsolve = splu(self.assembly(0).T).solve
        while self.t < self.t_max:
            #for ele in self.basis.get_dofs("l").nodal['u']:
                #u[ele] = self.initial_temp

            t, u = t + self.dt, backsolve(self.assembly(1) @ u)
            if t<=4:
                for ele in self.basis.get_dofs("l").nodal['u']:
                    u[ele] = self.initial_temp
            else:
                pass
            yield t, u

    def simulate(self):
        """
        Creating simulation of every frame as called by frame() method.

        Returns:
            animation(matplotlib.animation.FuncAnimation):animation of diffusion equation GEM on mesh

        """

        from argparse import ArgumentParser
        from pathlib import Path
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt

        from skfem.visuals.matplotlib import plot

        parser = ArgumentParser(description='heat equation in a rectangle')
        parser.add_argument('-g', '--gif', action='store_true',
                            help='write animated GIF', )
        args = parser.parse_args()
        ax = plot(self.m, self.initial_condition(), shading='gouraud')
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
            self.frame(0., self.initial_condition()),
            repeat=False,
            interval=50,
        )
        if args.gif:
            animation.save(Path(__file__).with_suffix('.gif'), 'imagemagick')
        else:
            plt.show()
