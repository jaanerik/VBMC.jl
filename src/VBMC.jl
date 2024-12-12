module VBMC

using Distributions
using Random
using StatsBase

println("=====")
pwd() |> println
readdir("../src/") |> println
println("=====")
include("./hpmm.jl")

export HPMM

end