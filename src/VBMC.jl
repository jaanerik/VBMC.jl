module VBMC

using Distributions
using Random
using StatsBase

include("utils.jl")
include("tmm.jl")
include("hpmm.jl")
include("mc.jl")

export HpmmDistribution,
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
