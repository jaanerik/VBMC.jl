@doc """
    Homogeneous Hidden Pairwise Markov Model

    U, X are subtype of @Int, Y is subtype of @Real.
"""
struct HPMM
    U::AbstractArray{<:Int}
    X::AbstractArray{<:Int}
    Y::AbstractArray{<:Real}
end

struct EmissionDistribution{T} <: Sampleable{Univariate,T}
    distr::Sampleable{Univariate,T}
    emissionFun::Function
    U::Int
    X::Int
end

struct HpmmDistribution <: AbstractTmmDistribution
    P1::ReshapedCategorical
    Pt::TransitionDistribution
    Pem::EmissionDistribution
end

# Distribution construction


function createEmissionDistribution(emissionFun, U::Int, X::Int)
    EmissionDistribution(emissionFun, U, X)
end

# Overrides

function Base.rand(ed::EmissionDistribution, u::Int, x::Int)
    randVal = ed.distr |> rand
    ed.emissionFun(randVal, u, x)
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
function Base.rand(hpmm_gen::HpmmDistribution, uprev::Int, xprev::Int, yprev::Real)
    p = rand(hpmm_gen.Pt, TmmStep(uprev, xprev, yprev))
    e = rand(hpmm_gen.Pem, p[1], p[2])
    TmmStep(p[1], p[2], e)
end
