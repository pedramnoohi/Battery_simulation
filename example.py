import oop_edition
from oop_edition import *
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
Diffusivity_coefficient={"lithium":0.000001 ,"electrolyte":0.05}

mesh=oop_edition.Mesh(plane=plane,length=length).skfem_mesher()
basis=oop_edition.Mesh(plane=plane,length=length).basis()
initial=oop_edition.FEM(dt=0.01,t_max=1000,initial_temp=200,mesh=mesh,basis=basis,Diffusivity_coefficient=Diffusivity_coefficient)
initial.simulate()
