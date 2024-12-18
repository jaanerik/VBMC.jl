module VBMC

using Distributions
using Random
using StatsBase
using LogarithmicNumbers
import Distributions: pdf

include("utils.jl")
include("tmm.jl")
include("hpmm.jl")
include("mc.jl")

export HpmmDistribution,
    TMM,
    TmmStep,
    HPMM,
    TmmDistribution,
    AbstractTmmDistribution,
    ReshapedCategorical,
    TransitionDistribution,
    EmissionDistribution,
    Emission,
    HpmmAnalyser,
    Alpha,
    Beta,
    pdf,
    getweights #

VBMC
end #module
