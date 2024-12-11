using VBMC
using VBMC:createFirstDistribution,createTransitionDistribution,createEmissionDistribution
using Test
using TOML
using BenchmarkTools
using Distributions
using Random
using StatsBase

@testset "VBMC" begin
    Random.seed!(123)
    #cd("test")
    vars = TOML.parsefile("Constants.toml")
    T, U, X, Y = vars["T"], vars["U"], vars["X"], vars["Y"]
    @test T == 10 && U == 3 && X == 2

    P1 = createFirstDistribution(U, X)
    Pt = createTransitionDistribution(U, X)
    Pe = createEmissionDistribution(U,X)
    @testset "HPMM helpers" begin
        @test size(rand(P1), 1) == 2
        @test size(Pt.mat) == (U,X,U,X) #let (:,:,j,k) denote P(·,· | u_j, x_k)
        @test size(rand(Pt,1,1), 1) == 2 

        Pe = createEmissionDistribution(U,X)
        @test typeof(rand(Pe,1,1)) <: Real
    end

    @testset "HPMM sampling" begin
        hpmm = HPMM(P1, Pt, Pe)
        @show hpmm
    end

end