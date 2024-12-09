module VBMC

function f()
    rand(100,100)
end

using Symbolics

T = 10 #timesteps in Markov chain
N = 3 #Different y values
Nₓ,Nᵤ = 5,4
@variables U x̃[1:Nₓ] x[1:Nₓ] ũ[1:Nᵤ] u[1:Nᵤ]

D = Differential(U)
z = U * log(U)
D(z) |> expand_derivatives

L = +(x...)

x = Array(1:5)
+( (x |> ff) ...)
ff = x -> 2x

ff = function(x)
    Qx_star(x)^(1/U) / U
end


P₁ = function(x₁,u₁,y₁)
    1/(N * Nₓ * Nᵤ)
end

Pₜ = function(xₜ,uₜ,yₜ, x_prev, u_prev, y_prev)
    1/(N * Nₓ * Nᵤ)
end

P = function(u,x,y)
    *(P₁(x[1],u[1],y[1]), (2:T |> t -> Pₜ(x[t],u[t],y[t], x[t-1],u[t-1],y[t-1]))...)
end

T
P(1,1,1)
randn

copy(x[1]) == x[1]

E = function(u,x,y)
    log(P(u,x,y))
end

Qx_star = function(x)
    exp(E(u,x,y) .* Q_u(u))
end




