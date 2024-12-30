module VBMC

using Distributions
using Random
using StatsBase
using LogarithmicNumbers
import Distributions: pdf

include("utils.jl")
include("TMM.jl")
include("HPMM.jl")
include("mc.jl")
include("vb.jl")

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
    AlphaBeta,#
    MarkovChain,
    VB,
    pdf

VBMC
end #module
