using CairoMakie
using ColorSchemes

α  = 1e-4                                                                            # Diffusivity
L  = 0.1                                                                             # Length
W  = 0.1                                                                             # Width
Nx = 66                                                                              # No.of steps in x-axis
Ny = 66                                                                              # No.of steps in y-axis
Δx = L/(Nx-1)                                                                        # x-grid spacing
Δy = W/(Ny-1)                                                                        # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))                                         # Largest stable time step

function diffuse!(u,α, Δt, Δx, Δy)
    dij  = view(u, 2:Nx-1, 2:Ny-1)
    di1j = view(u, 1:Nx-2, 2:Ny-1)
    dij1 = view(u, 2:Nx-1, 1:Ny-2)
    di2j = view(u, 3:Nx  , 2:Ny-1)
    dij2 = view(u, 2:Nx-1, 3:Ny  )                                                  # Stencil Computations

    @. dij += α * Δt * (
        (di1j - 2 * dij + di2j)/Δx^2 +
        (dij1 - 2 * dij + dij2)/Δy^2)                                               # Apply diffusion

    @. u[1, :] += α * Δt * (2*u[2, :] - 2*u[1, :])/Δx^2
    @. u[Nx, :] += α * Δt * (2*u[Nx-1, :] - 2*u[Ny, :])/Δx^2
    @. u[:, 1] += α * Δt * (2*u[:, 2]-2*u[:, 1])/Δy^2
    @. u[:, Ny] += α * Δt * (2*u[:, Nx-1]-2*u[:, Ny])/Δy^2                  # update boundary condition (Neumann BCs)

end


u= zeros(Nx,Ny)
u[28:38, 28:38] .= 5

fig , pltpbj = plot(u; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
    figure = (resolution = (600, 400), font = "CMU Serif"),
        axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
        xlabelsize = 15, ylabelsize = 15))
        Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction")
display(fig)

for i in 1:1000
       diffuse!(u, α, Δt, Δx, Δy)
    if i % 20 == 0
fig , pltpbj = plot(u; colormap  = :viridis ,markersize = 5, linestyle = ".-", 
    figure = (resolution = (600, 400), font = "CMU Serif"),
        axis =  ( xlabel ="Grid points (Nx)", ylabel ="Grid points (Ny)", backgroundcolor = :white,
        xlabelsize = 15, ylabelsize = 15))
        Colorbar(fig[1,2], limits = (0, 5),label = "Heat conduction")
display(fig)
    end
end