module VBMC

using Distributions
using Random
using StatsBase

println("=====")
pwd() |> println
println("=====")
readdir("./") |> println
println("=====")
readdir("../") |> println
println("=====")
include("HPMM.jl")

export HPMM

end