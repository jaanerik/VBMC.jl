module VBMC

using Distributions
using Random
using StatsBase

include("utils.jl")
include("TMM.jl")
include("HPMM.jl")

@doc """
    Draw a HPMM chain sample of size T from
    a hidden homogeneous pairwise markov chain.
    Returns HPMM
"""
function Base.rand(tmm_gen::AbstractTmmDistribution, T::Int)
    println("Calling rand from VBMC")
    U = Array{Int}(undef, T, 1)
    X = Array{Int}(undef, T, 1)
    Y = Array{Real}(undef, T, 1)
    first = rand(tmm_gen)
    U[1], X[1], Y[1] = first.u, first.x, first.y
    for t in 2:T
        next = rand(tmm_gen, U[t-1], X[t-1], Y[t-1])
        U[t], X[t], Y[t] = next.u, next.x, next.y
    end
    HPMM(U,X,Y)
end

export 
    HpmmDistribution,
    TMM,
    TmmStep,
    TMM,
    HPMM,
    AbstractTmmDistribution,
    ReshapedCategorical,
    TransitionDistribution,
    EmissionDistribution

VBMC
end #module