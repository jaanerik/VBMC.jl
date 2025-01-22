using Test
using TOML
using BenchmarkTools
using VBMC

using Distributions
using Random
using StatsBase
using LogarithmicNumbers

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

#P(·,· | u_prev, x_prev) = Pt[:,:,u_prev,x_prev] sums to 1 for fixed u_prev, x_prev
Pt = reshape(rand(Dirichlet(1:U*X), U * X), (U, X, U, X)) |> TransitionDistribution

function emissionf(randVal, u, x)
    randVal .+ x
end
function revf(emission, u, x)
    emission .- x
end #the dot vectorizes, could use X instead
emission = Emission(emissionf, revf)

Pe = EmissionDistribution{Continuous}(Normal(0, 1), emission, U, X)

@testset "HPMM helpers" begin
    @test size(rand(P1), 1) == 2
    @test size(Pt.mat) == (U, X, U, X) #let (:,:,j,k) denote P(·,· | u_j, x_k)
    @test size(rand(Pt, TmmStep(1, 1, 1)), 1) == 2
    @test typeof(rand(Pe, 1, 1)) <: Real
end

@testset "HPMM normal" begin
    hpmm_gen = HpmmDistribution(P1, Pt, Pe)
    hpmm = rand(hpmm_gen, T)
    @test begin
        (typeof(hpmm.X) <: AbstractArray{<:Int}) &
        (typeof(hpmm.U) <: AbstractArray{<:Int}) &
        (typeof(hpmm.Y) <: AbstractArray{<:Real})
    end
    @test hpmm.X |> size == (T,)
end

@testset "TMM sampling" begin
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

    tmm_gen = TmmDistribution(P1, Pt)
    tmm = rand(tmm_gen, T)
    @test size(tmm.Y) == (T,)
end

@testset "Forward-backward" begin
    dist = HpmmDistribution(P1, Pt, Pe)
    hpmm = rand(dist, T)
    analyser = HpmmAnalyser(hpmm, dist) #@btime 423.041 μs (6579 allocations: 209.28 KiB)
    u, x, t = sample(1:U), sample(1:X), sample(1:T)
    @test Iterators.product(1:U, 1:X) .|>
          (p -> pdf(analyser, t; u = p[1], x = p[2])) |>
          sum |>
          float ≈ 1.0

    @test 1:U .|> (u -> pdf(analyser, t; u = u)) |> sum |> float ≈ 1.0
    @test 1:X .|> (x -> pdf(analyser, t; x = x)) |> sum |> float ≈ 1.0
    @test pdf(analyser; X = hpmm.X) > 0
    @test pdf(analyser; U = hpmm.U) > 0
    @test pdf(analyser; U = hpmm.U, X = hpmm.X) > 0
    @test Iterators.product(1:X, 1:X) .|>
          (((x, xprev),) -> pdf(analyser, 2; x = x, xprev = xprev)) |>
          sum |>
          float ≈
          Iterators.product(1:U, 1:U) .|>
          (((u, uprev),) -> pdf(analyser, 2; u = u, uprev = uprev)) |>
          sum |>
          float

    analyser = HpmmAnalyser(hpmm, dist, isygiven = false)

    py =
        Iterators.product(1:U, 1:X) .|>
        (p -> pdf(analyser, 1; u = p[1], x = p[2])) |>
        sum |>
        float

    @test 1:U .|> (u -> pdf(analyser, 1; u = u)) |> sum |> float ≈ py
    @test 1:X .|> (x -> pdf(analyser, 1; x = x)) |> sum |> float ≈ py
    @test 1:X .|> (x -> pdf(analyser, 2; x = x)) |> sum |> float ≈ py
    @test Iterators.product(1:X, 1:X) .|>
          (((x, xprev),) -> pdf(analyser, 2; x = x, xprev = xprev)) |>
          sum |>
          float ≈
          Iterators.product(1:U, 1:U) .|>
          (((u, uprev),) -> pdf(analyser, 2; u = u, uprev = uprev)) |>
          sum |>
          float
end

@testset "Variational Bayes iteration converges" begin
    dist = HpmmDistribution(P1, Pt, Pe)
    hpmm = rand(dist, T)
    mcu = MarkovChain(U, T)
    mcx = MarkovChain(X, T)

    function norm(A::AbstractArray; p = 2)
        sum(abs.(A) .^ p)^(1 / p)
    end

    val = 1.0
    for _ = 1:200
        tmpP1, tmpPt = mcx.P1 |> deepcopy, mcx.Pt |> deepcopy
        fillalphaX!(mcx, mcu, P1, Pt, Pe, hpmm.Y)
        fillbetaX!(mcx, mcu, Pt, Pe, hpmm.Y)
        VBMC.fillPtx!(mcx, mcu, Pt, Pe, hpmm.Y)

        fillalphaU!(mcu, mcx, P1, Pt, Pe, hpmm.Y)
        fillbetaU!(mcu, mcx, Pt, Pe, hpmm.Y)
        VBMC.fillPtu!(mcu, mcx, Pt, Pe, hpmm.Y)

        signeda = mcx.Pt .|> Logarithmic
        signedb = tmpPt .|> Logarithmic
        val = norm(signeda .- signedb)
        if val < 1.0e-14
            break
        end
    end
    @test val < 1.0e-14
end
