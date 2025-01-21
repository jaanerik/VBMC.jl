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

@doc """
Creates a struct for marginal, element probabilities.
Uses Alpha, Beta with LogarithmicNumbers for small values.

kwarg isygiven true calculates P(⋅|y) and sums to 1, 
false calculates P(⋅,y) and sums to P(y).
"""
struct HpmmAnalyser
    hpmm::TMM
    dist::HpmmDistribution

    alpha::AlphaBeta
    beta::AlphaBeta
    U::Int
    X::Int
    isygiven::Bool
    HpmmAnalyser(hpmm, dist; isygiven = true) = begin
        U = dist.Pt.U
        X = dist.Pt.X
        T = size(hpmm.U)[1]
        isygiven = isygiven
        alpha, beta = AlphaBeta(ones(U, X, T)), AlphaBeta(ones(U, X, T))
        fillalpha!(alpha, hpmm, dist; isygiven = isygiven)
        fillbeta!(beta, hpmm, dist; isygiven = isygiven)
        new(hpmm, dist, alpha, beta, U, X)
    end
end

function flatten(x::AbstractArray{Tuple{Int,Int}})
    reduce(vcat, x)
end
function get_emission_mat(y::Real, Pem::EmissionDistribution)
    begin
        Iterators.product(1:Pem.U, 1:Pem.X) |>
        collect |>
        flatten .|>
        (t -> Pem.emission.rev(y, t[1], t[2])) .|>
        (R -> pdf(Pem.distr, R)) |>
        P -> reshape(P, Pem.U, Pem.X)
    end
end

function fillalpha!(alpha::AlphaBeta, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds alpha.mat[:, :, 1] = dist.P1 |> getweights
    if !isygiven
        @inbounds alpha.mat[:, :, 1] .*= get_emission_mat(hpmm.Y[1], dist.Pem)
    end
    @inbounds for t = 2:alpha.T
        for (u, x) in Iterators.product(1:alpha.U, 1:alpha.X)
            alpha.mat[u, x, t] = alpha.mat[:, :, t-1] .* dist.Pt.mat[u, x, :, :] |> sum
        end
        if !isygiven
            alpha.mat[:, :, t] .*= get_emission_mat(hpmm.Y[t], dist.Pem)
        end
    end
end
function fillbeta!(beta::AlphaBeta, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds beta.mat[:, :, beta.T] .= 1
    @inbounds for t in 1:beta.T-1 |> reverse
        emissionmat =
            isygiven ? ones(beta.U, beta.X) : get_emission_mat(hpmm.Y[t+1], dist.Pem)
        for (u, x) in Iterators.product(1:beta.U, 1:beta.X)
            beta.mat[u, x, t] =
                emissionmat .* beta.mat[:, :, t+1] .* dist.Pt.mat[:, :, u, x] |> sum
        end
    end
end

# Multiple dispatch

function Base.getindex(
    Pe::EmissionDistribution,
    u::Union{Int64,Colon},
    x::Union{Int64,Colon},
    y::Real,
)
    pdf(Pe.distr, Pe.emission.rev(y, u, x))
end

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

# function pdf(Pem::EmissionDistribution, emission::Real)
#     Iterators.product(1:Pem.U, 1:Pem.X) |> pdf(Pem.distr, Pem.emission.rev(emission, u, x))
# end

function pdf(
    analyser::HpmmAnalyser,
    t::Int;
    u::Union{Nothing,Int} = nothing,
    uprev::Union{Nothing,Int} = nothing,
    x::Union{Nothing,Int} = nothing,
    xprev::Union{Nothing,Int} = nothing,
)
    if isnothing(uprev) & isnothing(xprev)
        _pdfqt(analyser, t; u = u, x = x)
    else
        _pdfqtqprev(analyser, t; u = u, x = x, uprev = uprev, xprev = xprev)
    end
end

function _pdfqt(
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

function _pdfqtqprev(
    analyser::HpmmAnalyser,
    t::Int;
    u::Union{Nothing,Int} = nothing,
    uprev::Union{Nothing,Int} = nothing,
    x::Union{Nothing,Int} = nothing,
    xprev::Union{Nothing,Int} = nothing,
)
    pe = 1.0
    y = analyser.hpmm.Y[t]
    if isnothing(x)
        Iterators.product(1:analyser.X, 1:analyser.X) .|>
        (
            ((x, xprev),) ->
                _pdfqtqprev(analyser, t; u = u, uprev = uprev, x = x, xprev = xprev)
        ) |>
        sum
    elseif isnothing(u)
        Iterators.product(1:analyser.U, 1:analyser.U) .|>
        (
            ((u, uprev),) ->
                _pdfqtqprev(analyser, t; u = u, uprev = uprev, x = x, xprev = xprev)
        ) |>
        sum
    else
        analyser.alpha.mat[uprev, xprev, t-1] *
        analyser.dist.Pt.mat[u, x, uprev, xprev] *
        _emissionorone(analyser, y, u, x) *
        analyser.beta.mat[u, x, t]
    end
end

function _emissionorone(analyser, y, u, x)
    if !analyser.isygiven
        pdf(analyser.dist.Pem.distr, analyser.dist.Pem.emission.rev(y, u, x))
    else
        1
    end
end

#Note assumption that U, X are given in temporal order. X[1] is at t=1
function pdf(
    analyser;
    U::Union{Nothing,AbstractArray{<:Int}} = nothing,
    X::Union{Nothing,AbstractArray{<:Int}} = nothing,
)
    if isnothing(U)
        enumerate(X) .|> (((t, x),) -> pdf(analyser, t; x = x)) |> prod
    elseif isnothing(X)
        enumerate(U) .|> (((t, u),) -> pdf(analyser, t; u = u)) |> prod
    else
        enumerate(zip(U, X)) .|> (((t, (u, x)),) -> pdf(analyser, t; u = u, x = x)) |> prod
    end
end
