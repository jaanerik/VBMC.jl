abstract type AbstractTmmDistribution end

struct TmmStep
    u::Int
    x::Int
    y::Real
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

# Overrides

function Base.rand(rc::ReshapedCategorical)
    i = rand(rc.d)
    isnothing(rc.Y) ? reshapeIndex(i, rc.U, rc.X) : reshapeIndex(i, rc.U, rc.X, rc.Y)
end

function Base.rand(td::TransitionDistribution, prev::TmmStep)
    if isnothing(td.Y)
        w = td.mat[:, :, prev.u, prev.x] |> flatten |> Weights
        s = sample(1:td.U*td.X, w)
        reshapeIndex(s, td.U, td.X)
    else
        w = td.mat[:, :, :, prev.u, prev.x, prev.y] |> flatten |> Weights
        s = sample(1:td.U*td.X*td.Y, w)
        reshapeIndex(s, td.U, td.X, td.Y)
    end
end

function reshapeIndex(i::Int, U::Int, X::Int)
    i = i - 1
    j = (i % U) + 1
    k = div(i, U) + 1
    [j, k]
end

function reshapeIndex(i::Int, U::Int, X::Int, Y::Int)
    i = i - 1
    l = div(i, U * X) + 1
    i = i - (l - 1) * U * X
    j = (i % U) + 1
    k = div(i, U) + 1
    [j, k, l]
end

function reshapeIndex(t::Array{Int}, U, X)
    if size(t) == (2,)
        (t[2] - 1) * U + t[1]
    else
        (t[3] - 1) * U * X + (t[2] - 1) * U + t[1]
    end
end

# Helpers

function getWeights(rc::ReshapedCategorical)
    if isnothing(rc.Y)
        reshape(rc.d.p, (rc.U, rc.X))
    else
        reshape(rc.d.p, (rc.U, rc.X, rc.Y))
    end
end

@doc """
Flattens an array using reduce
"""
function flatten(x::AbstractArray{<:Real})
    reduce(vcat, x)
end
