
from mesher import Mesh
from fem_wrapper import FEM

#setup for the mesh
length=100
blobs=binary_blobs(length=length, volume_fraction=0.20,seed=2102)
plane=Image(blobs)
Diffusivity_coefficient={"lithium":0.000001 ,"electrolyte":0.05}

# This is weird, you create two instance of the same object with the same parameter !
mesh=Mesh(plane=plane,length=length).skfem_mesher()
basis=Mesh(plane=plane,length=length).basis()

# I would do that ... 
# mesher = Mesh(plane=plane,length=length)
# mesh = mesher.skfem_mesher()
# basis = mesher.basis()

initial=FEM(dt=0.01,t_max=1000,initial_temp=200,mesh=mesh,basis=basis,Diffusivity_coefficient=Diffusivity_coefficient)
initial.simulate()
