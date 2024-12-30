struct VB
    T::Int
    U::Int
    X::Int
    P1::ReshapedCategorical
    Pt::TransitionDistribution
    Y::AbstractArray{<:Real}
    Qu::MarkovChain
    alphau::AlphaBeta
    betau::AlphaBeta
    Qx::MarkovChain
    alphax::AlphaBeta
    betax::AlphaBeta

    VB(T, U, X, P1, Pt, Y) = new(
        T,
        U,
        X,
        P1,
        Pt,
        Y,
        MarkovChain(U, T),
        AlphaBeta(ones(U, T)),
        AlphaBeta(ones(U, T)),
        MarkovChain(X, T),
        AlphaBeta(ones(X, T)),
        AlphaBeta(ones(X, T)),
    )
end

"""    hpmm::TMM
    dist::HpmmDistribution

    alpha::Alpha
    beta::Beta
    U::Int
    X::Int
    isygiven::Bool
    HpmmAnalyser(hpmm, dist; isygiven = true) = begin
        U = dist.Pt.U
        X = dist.Pt.X
        T = size(hpmm.U)[1]
        isygiven = isygiven
        alpha, beta = Alpha(ones(U, X, T)), Beta(ones(U, X, T))
        fillalpha!(alpha, hpmm, dist; isygiven = isygiven)
        fillbeta!(beta, hpmm, dist; isygiven = isygiven)
        new(hpmm, dist, alpha, beta, U, X)
    end
end"""

function flatten(x::AbstractArray{Tuple{Int,Int}})
    reduce(vcat, x)
end

function fillalpha!(alpha::AlphaBeta, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds alpha.mat[:, :, 1] = dist.P1 |> getweights
    if !isygiven
        @inbounds alpha.mat[:, :, 1] .*= get_emmission_mat(hpmm.Y[1], dist.Pem)
    end
    @inbounds for t = 2:alpha.T
        for (u, x) in Iterators.product(1:alpha.U, 1:alpha.X)
            alpha.mat[u, x, t] = alpha.mat[:, :, t-1] .* dist.Pt.mat[u, x, :, :] |> sum
        end
        if !isygiven
            alpha.mat[:, :, t] .*= get_emmission_mat(hpmm.Y[t], dist.Pem)
        end
    end
end
function fillbeta!(beta::AlphaBeta, hpmm::TMM, dist::HpmmDistribution; isygiven = false)
    @inbounds beta.mat[:, :, beta.T] .= 1
    @inbounds for t in 1:beta.T-1 |> reverse
        emissionmat =
            isygiven ? ones(beta.U, beta.X) : get_emmission_mat(hpmm.Y[t+1], dist.Pem)
        for (u, x) in Iterators.product(1:beta.U, 1:beta.X)
            beta.mat[u, x, t] =
                emissionmat .* beta.mat[:, :, t+1] .* dist.Pt.mat[:, :, u, x] |> sum
        end
    end
end
