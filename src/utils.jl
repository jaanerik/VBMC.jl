abstract type AbstractTmmDistribution end

struct TmmStep
    u::Int
    x::Int
    y::Number
end

struct ReshapedCategorical <: Sampleable{Multivariate,Discrete}
    d::Categorical
    U::Int
    X::Int
    Y::Union{Int,Nothing}

    ReshapedCategorical(d, U, X) = new(d, U, X, nothing)
    ReshapedCategorical(d, U, X, Y) = new(d, U, X, Y)
end

struct TransitionDistribution <: Sampleable{Multivariate,Discrete}
    mat::AbstractArray{<:Real}
    U::Int
    X::Int
    Y::Union{Int,Nothing}

    TransitionDistribution(tmat::AbstractArray{<:Real}) =
        if ndims(tmat) == 4
            new(tmat, size(tmat)[1], size(tmat)[2], nothing)
        else
            shape = size(tmat)
            new(tmat, shape[1], shape[2], shape[3])
        end
end

struct AlphaBeta
    mat::AbstractArray{ULogarithmic}
    U::Int
    X::Int
    T::Int
    Z::Int
    AlphaBeta(mat::AbstractArray{<:Real,3}) = begin
        U, X, T = size(mat)
        new(mat .|> ULogarithmic, U, X, T, 0)
    end
    AlphaBeta(mat::AbstractArray{<:Real,2}) = begin
        Z, T = size(mat)
        new(mat .|> ULogarithmic, 0, 0, T, Z)
    end
end

# Multiple dispatch

@doc """
    rand(::AbstractTmmDistribution, ::Int)

Draw a TMM chain sample of size T from
a hidden homogeneous pairwise markov chain.
"""
function Base.rand(tmm_gen::AbstractTmmDistribution, T::Int)::TMM
    U = Array{Int}(undef, T, 1) |> vec
    X = Array{Int}(undef, T, 1) |> vec
    ytype = typeof(tmm_gen) == TmmDistribution ? Int : Real
    Y = Array{ytype}(undef, T, 1) |> vec
    first = rand(tmm_gen)
    U[1], X[1], Y[1] = first.u, first.x, first.y
    for t = 2:T
        next = rand(tmm_gen, TmmStep(U[t-1], X[t-1], Y[t-1]))
        U[t], X[t], Y[t] = next.u, next.x, next.y
    end
    TMM(U, X, Y)
end

function Base.rand(rc::ReshapedCategorical)
    i = rand(rc.d)
    isnothing(rc.Y) ? reshapeindex(i, rc.U, rc.X) : reshapeindex(i, rc.U, rc.X, rc.Y)
end

function Base.rand(td::TransitionDistribution, prev::TmmStep)
    if isnothing(td.Y)
        w = td.mat[:, :, prev.u, prev.x] |> flatten |> Weights
        s = sample(1:td.U*td.X, w)
        reshapeindex(s, td.U, td.X)
    else
        w = td.mat[:, :, :, prev.u, prev.x, prev.y] |> flatten |> Weights
        s = sample(1:td.U*td.X*td.Y, w)
        reshapeindex(s, td.U, td.X, td.Y)
    end
end

function reshapeindex(i::Int, U::Int, X::Int)
    i = i - 1
    j = (i % U) + 1
    k = div(i, U) + 1
    [j, k]
end

function reshapeindex(i::Int, U::Int, X::Int, Y::Int)
    i = i - 1
    l = div(i, U * X) + 1
    i = i - (l - 1) * U * X
    j = (i % U) + 1
    k = div(i, U) + 1
    [j, k, l]
end

function reshapeindex(t::Array{Int}, U, X)
    if size(t) == (2,)
        (t[2] - 1) * U + t[1]
    else
        (t[3] - 1) * U * X + (t[2] - 1) * U + t[1]
    end
end

# Helpers

function getweights(rc::ReshapedCategorical)
    if isnothing(rc.Y)
        reshape(rc.d.p, (rc.U, rc.X))
    else
        reshape(rc.d.p, (rc.U, rc.X, rc.Y))
    end
end

function Base.getindex(
    P1::ReshapedCategorical,
    u::Union{Int64,Colon},
    x::Union{Int64,Colon},
)
    getweights(P1)[u, x]
end
function Base.getindex(
    Pt::TransitionDistribution,
    u::Union{Int64,Colon},
    x::Union{Int64,Colon},
    uprev::Union{Int64,Colon},
    xprev::Union{Int64,Colon},
)
    Pt.mat[u, x, uprev, xprev]
end
function Base.getindex(alphabeta::AlphaBeta, z::Union{Int64,Colon}, t::Union{Int64,Colon})
    alphabeta.mat[z, t]
end
function Base.setindex!(alphabeta::AlphaBeta, val::Real, z::Int64, t::Int64)
    alphabeta.mat[z, t] = val
end
function Base.setindex!(
    alphabeta::AlphaBeta,
    val::AbstractArray{<:Real},
    z::Colon,
    t::Union{Int64,Colon},
)
    alphabeta.mat[z, t] = val
end

@doc """
Flattens an array using reduce
"""
function flatten(x::AbstractArray{<:Real})
    reduce(vcat, x)
end

function normalise(X::AbstractArray{<:Real})
    X ./ sum(X)
end

function normaliseDims1(X::AbstractArray{<:Real})
    X ./ sum(X, dims=1)
end