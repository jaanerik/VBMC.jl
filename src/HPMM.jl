"""
Homogeneous Hidden Pairwise Markov Model
"""

struct Emission
    fun::Function #rand , (u,x) -> val
    rev::Function #val, (u,x) -> rand (can be pdf-ed by distr)
end

struct EmissionDistribution{T} <: Sampleable{Univariate,T}
    distr::Sampleable{Univariate,T}
    emission::Emission
    U::Int
    X::Int
end

struct HpmmDistribution <: AbstractTmmDistribution
    P1::ReshapedCategorical
    Pt::TransitionDistribution
    Pem::EmissionDistribution
end

struct Alpha
    mat::AbstractArray{ULogarithmic,3}
    U::Int
    X::Int
    T::Int
    Alpha(mat::AbstractArray{<:Real,3}) = begin
        U, X, T = size(mat)
        new(mat .|> ULogarithmic, U, X, T)
    end
end
struct Beta
    mat::AbstractArray{ULogarithmic,3}
    U::Int
    X::Int
    T::Int
    Beta(mat::AbstractArray{<:Real,3}) = begin
        U, X, T = size(mat)
        new(mat .|> ULogarithmic, U, X, T)
    end
end

@doc """
Creates a struct for marginal, element probabilities.
Uses Alpha, Beta with LogarithmicNumbers for small values.
"""
struct HpmmAnalyser
    hpmm::TMM
    dist::HpmmDistribution

    alpha::Alpha
    beta::Beta
    U::Int
    X::Int
    HpmmAnalyser(hpmm, dist; isygiven = true) = begin
        U = dist.Pt.U
        X = dist.Pt.X
        T = size(hpmm.U)[1]
        alpha, beta = Alpha(ones(U, X, T)), Beta(ones(U, X, T))
        fillalpha!(alpha, hpmm, dist; isygiven = isygiven)
        fillbeta!(beta, hpmm, dist; isygiven = isygiven)
        new(hpmm, dist, alpha, beta, U, X)
    end
end

function flatten(x::AbstractArray{Tuple{Int,Int}})
    reduce(vcat, x)
end
function get_emmission_mat(y::Real, Pem::EmissionDistribution)
    begin
        Iterators.product(1:Pem.U, 1:Pem.X) |>
        collect |>
        flatten .|>
        (t -> Pem.emission.rev(y, t[1], t[2])) .|>
        (R -> pdf(Pem.distr, R)) |>
        P -> reshape(P, Pem.U, Pem.X)
    end
end

function fillalpha!(alpha::Alpha, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds alpha.mat[:, :, 1] = dist.P1 |> getweights
    if !isygiven
        @inbounds alpha.mat[:, :, 1] .*= get_emmission_mat(hpmm.Y[1], dist.Pem)
    end
    @inbounds for t = 2:alpha.T
        for (u, x) in Iterators.product(1:alpha.U, 1:alpha.X)
            alpha.mat[u, x, t] = alpha.mat[:, :, t-1] .* dist.Pt.mat[u, x, :, :] |> sum
        end
        if !isygiven
            alpha.mat[:, :, t] .*= get_emmission_mat(hpmm.Y[t], dist.Pem)
        end
    end
end
function fillbeta!(beta::Beta, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds beta.mat[:, :, beta.T] .= 1
    @inbounds for t in 1:beta.T-1 |> reverse
        for (u, x) in Iterators.product(1:beta.U, 1:beta.X)
            beta.mat[u, x, t] = beta.mat[:, :, t+1] .* dist.Pt.mat[:, :, u, x] |> sum
        end
        if !isygiven
            beta.mat[:, :, t] .*= get_emmission_mat(hpmm.Y[t+1], dist.Pem)
        end
    end
end

# Multiple dispatch

function Base.rand(ed::EmissionDistribution, u::Int, x::Int)
    randVal = ed.distr |> rand
    ed.emission.fun(randVal, u, x)
end

@doc """
Draw the first sample from 
a hidden homogeneous pairwise markov chain.
Returns @TmmStep with u, x, y fields
"""
function Base.rand(hpmm_gen::HpmmDistribution)
    p = rand(hpmm_gen.P1)
    e = rand(hpmm_gen.Pem, p[1], p[2])
    return TmmStep(p[1], p[2], e)
end

@doc """
Draw the next sample from 
a hidden homogeneous pairwise markov chain
based on previous @TmmStep
Returns @TmmStep with u, x, y fields
"""
function Base.rand(hpmm_gen::HpmmDistribution, prev::TmmStep)
    p = rand(hpmm_gen.Pt, prev)
    e = rand(hpmm_gen.Pem, p[1], p[2])
    TmmStep(p[1], p[2], e)
end

function pdf(Pem::EmissionDistribution, emission::Real, u::Int, x::Int)
    pdf(Pem.distr, Pem.emission.rev(emission, u, x))
end

function pdf(Pem::EmissionDistribution, emission::Real)
    Iterators.product(1:Pem.U, 1:Pem.X) |> pdf(Pem.distr, Pem.emission.rev(emission, u, x))
end


# function pdf(analyser::HpmmAnalyser, u::Int, x::Int, t::Int)
#     analyser.alpha.mat[u, x, t] * analyser.beta.mat[u, x, t]
# end

function pdf(
    analyser::HpmmAnalyser,
    t::Int;
    u::Union{Nothing,Int} = nothing,
    x::Union{Nothing,Int} = nothing,
)
    if isnothing(x)
        1:analyser.X .|>
        (x -> analyser.alpha.mat[u, x, t] * analyser.beta.mat[u, x, t]) |>
        sum
    elseif isnothing(u)
        1:analyser.U .|>
        (u -> analyser.alpha.mat[u, x, t] * analyser.beta.mat[u, x, t]) |>
        sum
    else
        analyser.alpha.mat[u, x, t] * analyser.beta.mat[u, x, t]
    end
end

#Iterators.product(1:U,1:X) |> collect |> flatten |> x -> reshape(x, U, X)

"""
@btime begin
    Iterators.product(1:U,1:X) |> 
    collect |> 
    flatten .|> 
    (t -> Pe.emission.rev(val, t[1], t[2])) |>
    R -> pdf(Pe.distr, R) |>
    P -> reshape(P, U, X)
end #2.551 μs (47 allocations: 2.21 KiB)

@btime begin
    Iterators.product(1:U,1:X) |> 
    collect |> 
    flatten .|> 
    (t -> revf(val, t[1], t[2])) |>
    R -> pdf(Pe.distr, R) |>
    P -> reshape(P, U, X)
end #2.120 μs (35 allocations: 1.74 KiB)
"""
