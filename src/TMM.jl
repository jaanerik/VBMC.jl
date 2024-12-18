struct TMM
    U::AbstractArray{Int}
    X::AbstractArray{Int}
    Y::AbstractArray{<:Real}
end

struct TmmDistribution <: AbstractTmmDistribution
    P1::ReshapedCategorical
    Pt::TransitionDistribution
end

function Base.rand(tmm_gen::TmmDistribution)
    p = rand(tmm_gen.P1)
    return TmmStep(p[1], p[2], p[3])
end

function Base.rand(tmm_gen::TmmDistribution, prev::TmmStep)
    p = rand(tmm_gen.Pt, prev)
    return TmmStep(p[1], p[2], p[3])
end
