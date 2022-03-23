using CUDA
using CairoMakie
using ColorSchemes

const α  = 1e-4                                                                            # Diffusivity
const L  = 0.1                                                                             # Length
const W  = 0.1                                                                             # Width
const Nx = 66                                                                              # No.of steps in x-axis
const Ny = 66                                                                              # No.of steps in y-axis
const Δx = L/(Nx-1)                                                                        # x-grid spacing
const Δy = W/(Ny-1)                                                                        # y-grid spacing
const Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))                                         # Largest stable time step

function diffuse!(du, u, p,t)
    dijij = view(du, 2:Nx-1, 2:Ny-1)
    dij  = view(u, 2:Nx-1, 2:Ny-1)
    di1j = view(u, 1:Nx-2, 2:Ny-1)
    dij1 = view(u, 2:Nx-1, 1:Ny-2)
    di2j = view(u, 3:Nx  , 2:Ny-1)
    dij2 = view(u, 2:Nx-1, 3:Ny  )                                                  # Stencil Computations

    @. dijij  = α  * (
        (di1j - 2 * dij + di2j)/Δx^2 +
        (dij1 - 2 * dij + dij2)/Δy^2)                                               # Apply diffusion

    @. du[1, :]  += α  * (2*u[2, :] - 2*u[1, :])/Δx^2
    @. du[Nx, :] += α  * (2*u[Nx-1, :] - 2*u[Ny, :])/Δx^2
    @. du[:, 1]  += α   * (2*u[:, 2]-2*u[:, 1])/Δy^2
    @. du[:, Ny] += α * (2*u[:, Nx-1]-2*u[:, Ny])/Δy^2                  # update boundary condition (Neumann BCs)

end

u_GPU= CUDA.zeros(Nx,Ny)
u_GPU[28:38, 28:38] .= 5

fig , pltpbj = plot(u_GPU; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
    figure = (resolution = (600, 400), font = "CMU Serif"),
        axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
        xlabelsize = 15, ylabelsize = 15))
        Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction")
display(fig)


using DifferentialEquations, DiffEqGPU

tspan = (0.0, 1.0)
prob = ODEProblem(diffuse!, u_GPU, tspan)
sol = solve(prob, Euler(), dt=Δt)
# sol2 = solve(prob, alg_hins=[:stiff], save_everystep=false)
# sol3 = solve(prob, KenCarp5())


#sol4 = solve(prob, QNDF())
#sol5 = solve(prob, FBDF())
#sol6 = solve(prob, Vern8())
#sol7 = solve(prob, Tsit5())
#sol8 = solve(prob, Feagin14())
#sol110 = solve(prob,RadauIIA5(), dt=Δt)
#*sol11 = solve(prob, SSPRK22(), dt=Δt)
#*sol12 = solve(prob,AitkenNeville(), dt=Δt)


for i =  2 : length(sol)
    fig , pltpbj = plot(sol.u[i]; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
    figure = (resolution = (600, 400), font = "CMU Serif"),
        axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
        xlabelsize = 15, ylabelsize = 15))
        Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction")
    display(fig)
end






