using VBMC
using VBMC:createFirstDistribution
using Test
using TOML
using BenchmarkTools
using Distributions
using Random

@testset "VBMC" begin
    Random.seed!(123)
    vars = TOML.parsefile("Constants.toml")
    T, U, X, Y = vars["T"], vars["U"], vars["X"], vars["Y"]

    @testset "TMM" begin
        P₁z = createFirstDistribution(U, X)
        # P_yem = EmissionDistribution(1,1,Normal(0,1))
         #shape U*X*Y where sum[i,j,:] = 1
        @show P₁z
        Pₜ = Normal(0, 1)
        @test true
    end

    @test T == 10 && U == 3 && X == 2
end