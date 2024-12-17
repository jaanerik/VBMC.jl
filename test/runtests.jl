using Test
using TOML
using BenchmarkTools
using VBMC
# using VBMC:ReshapedCategorical,EmissionDistribution,HpmmDistribution,TransitionDistribution
include("test_utils.jl")

@testset "VBMC" begin
    Random.seed!(123)
    if "test" in readdir()
        cd("test")
    end

    vars = TOML.parsefile("Constants.toml")
    T, U, X, Y = vars["T"], vars["U"], vars["X"], vars["Y"]
    @test T == 10 && U == 3 && X == 2

    P1 = begin
        Vector(1:U*X) |>
        Dirichlet |>
        rand |>
        (x -> Categorical(x)) |>
        (d -> ReshapedCategorical(d, U, X))
    end

    #P(·,· | u_i, x_j) sums to 1 for fixed i, j
    Pt = reshape(rand(Dirichlet(1:U*X), U * X), (U, X, U, X)) |> TransitionDistribution

    function emissionFun(randVal, u, x)
        randVal + x
    end
    Pe = EmissionDistribution{Continuous}(Normal(0, 1), emissionFun, U, X)

    @testset "HPMM helpers" begin
        @test size(rand(P1), 1) == 2
        @test size(Pt.mat) == (U, X, U, X) #let (:,:,j,k) denote P(·,· | u_j, x_k)
        @test size(rand(Pt, TmmStep(1, 1, 1)), 1) == 2
        @test typeof(rand(Pe, 1, 1)) <: Real
    end

    @testset "HPMM sampling with normal emission" begin
        hpmm_gen = HpmmDistribution(P1, Pt, Pe)
        hpmm = rand(hpmm_gen, T)
        @test begin
            (typeof(hpmm.X) <: AbstractArray{<:Int}) &
            (typeof(hpmm.U) <: AbstractArray{<:Int}) &
            (typeof(hpmm.Y) <: AbstractArray{<:Real})
        end
        @test hpmm.X |> size == (T, 1)
    end

    @testset "General TMM sampling" begin
        P1 = begin
            Vector(1:U*X*Y) |>
            Dirichlet |>
            rand |>
            (x -> Categorical(x)) |>
            (d -> ReshapedCategorical(d, U, X, Y))
        end

        #P(·,·,· | u_i, x_j, y_k) sums to 1 for fixed i, j, k
        Pt = begin
            rand(Dirichlet(1:U*X*Y), U * X * Y) |>
            W -> reshape(W, (U, X, Y, U, X, Y)) |> TransitionDistribution
        end
        @test Pt.mat[:, :, :, 1, 1, 1] |> sum ≈ 1.0
    end
end
