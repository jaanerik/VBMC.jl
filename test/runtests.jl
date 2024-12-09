using VBMC
using Test
using TOML
using BenchmarkTools

@testset "VBMC.jl" begin
    x = f()

    vars = TOML.parsefile("Constants.toml")
    T, U, X, Y = vars["T"], vars["U"], vars["X"], vars["Y"]

    @test T == 10
end