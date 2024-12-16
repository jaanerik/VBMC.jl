@doc """
    Homogeneous Hidden Pairwise Markov Model

    U, X are subtype of @Int, Y is subtype of @Real.
"""
struct HPMM
    U::AbstractArray{<:Int}
    X::AbstractArray{<:Int}
    Y::AbstractArray{<:Real}
end

struct ReshapedCategorical <: Sampleable{Multivariate , Discrete } 
    d::Categorical
    U::Int
    X::Int
end

struct TransitionDistribution <: Sampleable{Multivariate , Discrete }
    mat::AbstractArray{<:Real}
    U::Int
    X::Int

    TransitionDistribution(
        tmat::AbstractArray{<:Real}
        ) = new(tmat, size(tmat)[1],size(tmat)[2])
end

struct EmissionDistribution{T} <: Sampleable{Univariate, T}
    distr::Sampleable{Univariate, T}
    emissionFun::Function
    U::Int
    X::Int
end

struct HpmmDistribution <: AbstractTmmDistribution
    P1::ReshapedCategorical
    Pt::TransitionDistribution
    Pem::EmissionDistribution
end

# Distribution construction


function createEmissionDistribution(emissionFun, U::Int, X::Int)
    EmissionDistribution(emissionFun,U,X)
end

# Overrides

function Base.rand(rc::ReshapedCategorical)
    i = rand(rc.d)
    reshapeIndex(i, rc.U, rc.X)
end

function Base.rand(td::TransitionDistribution, uprev::Int, xprev::Int)
    
    w = td.mat[:,:,uprev,xprev] |> flatten |> Weights
    s = sample(1:td.U*td.X, w)
    reshapeIndex(s, td.U, td.X)
end

function Base.rand(ed::EmissionDistribution, u::Int, x::Int)
    randVal = ed.distr |> rand
    ed.emissionFun(randVal, u, x)
end

# function Base.rand(
#     P1::ReshapedCategorical,
#     Pt::TransitionDistribution,
#     T::Int)
#     pmm = Array{Array{Int},2}(undef, T, 1)
#     for t in 2:T
#         uprev, xprev = pmm[t-1][1], pmm[t-1][2]
#         pmm[t] = rand(Pt, uprev, xprev)
#     end
#     vcat((pmm .|> transpose)...)
# end

@doc """
    Draw the first sample from 
    a hidden homogeneous pairwise markov chain.
    Returns TmmStep with u, x, y fields
"""
function Base.rand(hpmm_gen::HpmmDistribution)
    p = rand(hpmm_gen.P1)
    e = rand(hpmm_gen.Pem, p[1], p[2])
    return TmmStep(p[1], p[2], e)
end

function Base.rand(hpmm_gen::HpmmDistribution, uprev::Int, xprev::Int, yprev::Real)
    p = rand(hpmm_gen.Pt, uprev, xprev)
    e = rand(hpmm_gen.Pem, p[1], p[2])
    TmmStep(p[1], p[2], e)
end

# Helpers

function reshapeIndex(i::Int, U::Int, X::Int)
    i = i-1
    j = (i%U)+1
    k = div(i,U)+1
    [j,k]
end

function reshapeIndex(t::Array{Int}, U, X)
    @assert size(t) == (2,)
    (t[2]-1)*U + t[1]
end

function getWeights(rc::ReshapedCategorical)
    reshape(rc.d.p, (rc.U, rc.X))
end

@doc """
Flattens an array using reduce
"""
function flatten(x::AbstractArray{<:Real}) reduce(vcat, x) end