module VBMC

using Distributions
using Random
using StatsBase

println("=====")
pwd() |> println
readdir("../src/") |> println
println("=====")
include("HPMM.jl")

export HPMM

end