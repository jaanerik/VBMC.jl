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

    Pt = reshape( rand(Dirichlet(1:U*X),U*X), (U,X,U,X)) |> TransitionDistribution
    
    function emissionFun(randVal, u, x)
        randVal + x
    end
    Pe = EmissionDistribution{Continuous}(Normal(0,1), emissionFun, U, X)

    @testset "HPMM helpers" begin
        @test size(rand(P1), 1) == 2
        @test size(Pt.mat) == (U,X,U,X) #let (:,:,j,k) denote P(·,· | u_j, x_k)
        @test size(rand(Pt,1,1), 1) == 2 
        @test typeof(rand(Pe,1,1)) <: Real
    end

    @testset "HPMM sampling with normal emission" begin
        hpmm_gen = HpmmDistribution(P1, Pt, Pe)
        hpmm = rand(hpmm_gen, T)
        @test begin
            (typeof(hpmm.X) <: Vector{<:Int}) & 
            (typeof(hpmm.U) <: Vector{<:Int}) & 
            (typeof(hpmm.Y) <: Vector{<:Real})
        end
        @test hpmm.X |> size == (T,)
    end

end

function abc(kwargs...)
    println(kwargs)
end

foo = Dict(:a=>1, :b=>2, :c=>3)
function m(;a=0,b=0,c=0)
    println(a)
    println((a,b,c))
end
m(a=2)