struct TMM
    U::AbstractArray{Int}
    X::AbstractArray{Int}
    Y::AbstractArray{<:Real}
end

struct TmmDistribution
    P1::ReshapedCategorical
    Pt::TransitionDistribution
end

function Base.rand(hpmm_gen::TmmDistribution)
    p = rand(hpmm_gen.P1)
    e = rand(hpmm_gen.Pem, p[1], p[2])
    return TmmStep(p[1], p[2], e)
end

function Base.rand(hpmm_gen::TmmDistribution, uprev::Int, xprev::Int, yprev::Real)
    p = rand(hpmm_gen.Pt, TmmStep(uprev, xprev, yprev))
    e = rand(hpmm_gen.Pem, p[1], p[2])
    TmmStep(p[1], p[2], e)
end