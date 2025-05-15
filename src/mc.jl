"""
    Markov Chain and forward-backward, Viterbi algorithm
    implemented
"""

@doc """
Let matrix element i,j,t denote
the probability P_t(z_i | z_j)
at timestep t, prev element z_j.

P_t at t=1 is meaningless and exists for notational comfort.
"""
struct InhomogeneousTransitionDistribution
    mat::AbstractArray{ULogarithmic,3}
end

@doc """
Creates Markov Chain with uniform P1, P_t for each t > 2.
"""
struct MarkovChain
    P1::AbstractArray{ULogarithmic}
    Pt::AbstractArray{ULogarithmic,3}
    Z::Int
    alpha::AlphaBeta
    beta::AlphaBeta

    MarkovChain(Z::Int, T::Int) = new(
        ones(Z) ./ Z,
        ones(Z, Z, T) ./ (Z*Z),
        Z,
        AlphaBeta(ones(Z, T) ./ Z),
        begin 
            beta = AlphaBeta(ones(Z, T) ./ Z)
            beta[:, T] .= 1
            beta
        end,
    )
end

function lnr1x(x::Int, U::Int, P1::ReshapedCategorical, Q1::AbstractArray{<:ULogarithmic})
    1:U .|> (u -> log(P1[u, x]) * Q1[u]) |> sum
end
function lnrtx(
    x::Int,
    xprev::Int,
    U::Int,
    Pt::TransitionDistribution,
    Qt::AbstractArray{<:ULogarithmic,2},
)
    Iterators.product(1:U, 1:U) .|>
    (((u, uprev),) -> log(Pt[u, x, uprev, xprev]) * Qt[u, uprev]) |>
    sum
end
function lnr1u(u::Int, X::Int, P1::ReshapedCategorical, Q1::AbstractArray{<:ULogarithmic})
    1:X .|> (x -> log(P1[u, x]) * Q1[x]) |> sum
end
function lnrtu(
    u::Int,
    uprev::Int,
    X::Int,
    Pt::TransitionDistribution,
    Qt::AbstractArray{<:ULogarithmic,2},
)
    Iterators.product(1:X, 1:X) .|>
    (((x, xprev),) -> log(Pt[u, x, uprev, xprev]) * Qt[x, xprev]) |>
    sum
end

function lnbtx(
    y::Real,
    Pe::EmissionDistribution{Continuous},
    x::Int,
    U::Int,
    Q1::AbstractArray{<:ULogarithmic},
)
    1:U .|> (u -> log(Pe[u, x, y]) * Q1[u]) |> sum
end
function lnbtu(
    y::Real,
    Pe::EmissionDistribution{Continuous},
    u::Int,
    X::Int,
    Q1::AbstractArray{<:ULogarithmic},
)
    1:X .|> (x -> log(Pe[u, x, y]) * Q1[x]) |> sum
end

function p1x(
    x::Int,
    U::Int,
    P1::ReshapedCategorical,
    Q1::AbstractArray{<:ULogarithmic},
    y1::Real,
    Pe::EmissionDistribution,
)
    @inline exp(lnr1x(x, U, P1, Q1)) * exp(lnbtx(y1, Pe, x, U, Q1))
end
function p1u(
    u::Int,
    X::Int,
    P1::ReshapedCategorical,
    Q1::AbstractArray{<:ULogarithmic},
    y1::Real,
    Pe::EmissionDistribution,
)
    @inline exp(lnr1u(u, X, P1, Q1)) * exp(lnbtu(y1, Pe, u, X, Q1))
end

function ptx(
    x::Int,
    xprev::Int,
    U::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Qt2::AbstractArray{ULogarithmic,2},
    Qt::AbstractArray{<:ULogarithmic},
)
    @inline exp(lnrtx(x, xprev, U, Pt, Qt2)) * exp(lnbtx(y, Pe, x, U, Qt))
end
function ptu(
    u::Int,
    uprev::Int,
    X::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Qt2::AbstractArray{ULogarithmic,2},
    Qt::AbstractArray{<:ULogarithmic},
)
    @inline exp(lnrtu(u, uprev, X, Pt, Qt2)) * exp(lnbtu(y, Pe, u, X, Qt))
end

function fillalphaX!(
    mcx::MarkovChain,
    mcu::MarkovChain,
    P1::ReshapedCategorical,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    Q1 = mcu.P1
    @inbounds mcx.alpha.mat[:, 1] = 1:mcx.Z .|> (x -> p1x(x, mcu.Z, P1, Q1, Y[1], Pe))

    for t = 2:mcx.alpha.T
        @inbounds mcx.alpha.mat[:, t] =
            Iterators.product(1:mcx.Z, 1:mcx.Z) .|>
            (
                ((x, xprev),) ->
                    ptx(x, xprev, mcu.Z, Y[t], Pt, Pe, mcu.Pt[:, :, t],mcu.alpha[:,t].*mcu.beta[:,t]) *
                    mcx.alpha[xprev, t-1]
            ) |>
            Q -> sum(Q, dims = 2) #|> normalise
    end
end
function fillalphaU!(
    mcu::MarkovChain,
    mcx::MarkovChain,
    P1::ReshapedCategorical,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    Q1 = mcx.P1
    @inbounds mcu.alpha.mat[:, 1] = 1:mcu.Z .|> (u -> p1u(u, mcx.Z, P1, Q1, Y[1], Pe))

    for t = 2:mcu.alpha.T
        @inbounds mcu.alpha.mat[:, t] =
            Iterators.product(1:mcu.Z, 1:mcu.Z) .|>
            (
                ((u, uprev),) ->
                    ptu(u, uprev, mcx.Z, Y[t], Pt, Pe, mcx.Pt[:, :, t],mcx.alpha[:,t].*mcx.beta[:,t]) *
                    mcu.alpha[uprev, t-1]
            ) |>
            Q -> sum(Q, dims = 2) #|> normalise
    end
end

function fillbetaX!(
    mcx::MarkovChain,
    mcu::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    @inbounds mcx.beta.mat[:, mcx.beta.T] .= 1

    @inbounds for t in 1:mcx.beta.T-1 |> reverse
        @inbounds mcx.beta.mat[:, t] =
            Iterators.product(1:mcx.Z, 1:mcx.Z) .|>
            (
                ((x, xnext),) ->
                    ptx(xnext, x, mcu.Z, Y[t+1], Pt, Pe, mcu.Pt[:, :, t+1],mcu.alpha[:,t+1].*mcu.beta[:,t+1]) * #t -> t+1 changed
                    mcx.beta[xnext, t+1]
            ) |>
            Q -> sum(Q, dims = 2) #|> normalise
    end
end
function fillbetaU!(
    mcu::MarkovChain,
    mcx::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    @inbounds mcu.beta.mat[:, mcu.beta.T] .= 1

    @inbounds for t in 1:mcu.beta.T-1 |> reverse
        @inbounds mcu.beta.mat[:, t] =
            Iterators.product(1:mcu.Z, 1:mcu.Z) .|>
            (
                ((u, unext),) ->
                    ptu(unext, u, mcx.Z, Y[t+1], Pt, Pe, mcx.Pt[:, :, t+1],mcx.alpha[:,t+1].*mcx.beta[:,t+1]) * #t -> t+1 changed
                    mcu.beta[unext, t+1]
            ) |>
            Q -> sum(Q, dims = 2) #|> normalise
    end
end

function fillPtx!(
    mcx::MarkovChain,
    mcu::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    mcnorm = norm(mcx)
    T = mcx.alpha.T
    mcx.alpha.mat[:,1:T-1] .= mcx.alpha.mat[:,1:T-1] ./ mcnorm#^0.5
    mcx.beta.mat[:,1:T-1] .= mcx.beta.mat[:,1:T-1] #./ mcnorm#^0.5
    mcx.alpha.mat[:,T] .= mcx.alpha.mat[:,T] ./ mcnorm
    mcx.P1[:] = mcx.alpha[:, 1] .* mcx.beta[:, 1]
    for t = 2:mcx.alpha.T
        @inbounds mcx.Pt[:, :, t] =
            Iterators.product(1:mcx.Z, 1:mcx.Z) .|> (
                ((x, xprev),) ->
                    mcx.alpha[xprev, t-1] *
                    ptx(x, xprev, mcu.Z, Y[t], Pt, Pe, mcu.Pt[:, :, t],mcu.alpha[:,t].*mcu.beta[:,t]) *
                    mcx.beta[x, t] 
            )

    end
end
function fillPtu!(
    mcu::MarkovChain,
    mcx::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
    Y::AbstractArray{<:Real},
)
    mcnorm = norm(mcu)
    T = mcu.alpha.T
    mcu.alpha.mat[:,1:T-1] .= mcu.alpha.mat[:,1:T-1] ./ mcnorm#^0.5
    mcu.beta.mat[:,1:T-1] .= mcu.beta.mat[:,1:T-1] #./ mcnorm^0.5
    mcu.alpha.mat[:,T] .= mcu.alpha.mat[:,T] ./ mcnorm
    mcu.P1[:] = mcu.alpha[:, 1] .* mcu.beta[:, 1] 
    for t = 2:mcx.alpha.T
        @inbounds mcu.Pt[:, :, t] =
            Iterators.product(1:mcu.Z, 1:mcu.Z) .|>
            (
                ((u, uprev),) ->
                    mcu.alpha[uprev, t-1] *
                    ptu(u, uprev, mcx.Z, Y[t], Pt, Pe, mcx.Pt[:, :, t],mcx.alpha[:,t].*mcx.beta[:,t]) *
                    mcu.beta[u, t]
            )
    end
end


function pdf(mc::MarkovChain, z::Union{Int,Colon}, t::Int)
    @inbounds mc.alpha.mat[z, t] .* mc.beta.mat[z, t]
end

#For t > 1
function pdfu(
    mcu::MarkovChain,
    u::Int,
    uprev::Int,
    t::Int,
    y::Real,
    mcx::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    Σ = ptu(u, uprev, mcx.Z, y, Pt, Pe, mcx.Pt[:, :, t],mcx.alpha[:,t].*mcx.beta[:,t])
    @inbounds mcu.alpha[uprev, t-1] * mcu.beta[u, t] * Σ
end
function pdfx(
    mcx::MarkovChain,
    x::Int,
    xprev::Int,
    t::Int,
    y::Real,
    mcu::MarkovChain,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    Σ = ptx(x, xprev, mcu.Z, y, Pt, Pe, mcu.Pt[:, :, t],mcu.alpha[:,t].*mcu.beta[:,t])
    @inbounds mcx.alpha[uprev, t-1] * mcx.beta[u, t] * Σ
end

function pdfu(
    mcu::MarkovChain,
    mcx::MarkovChain,
    t::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    Iterators.product(1:mcu.Z, 1:mcu.Z) .|>
    (((u, uprev),) -> pdfu(mcu, u, uprev, t, y, mcx, Pt, Pe))
end
function pdfx(
    mcx::MarkovChain,
    mcu::MarkovChain,
    t::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    Iterators.product(1:mcx.Z, 1:mcx.Z) .|>
    (((x, xprev),) -> pdfx(mcx, x, xprev, t, y, mcu, Pt, Pe))
end

function qxxprev(
    x::Int,
    xprev::Int,
    mcu::MarkovChain,
    t::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    ptx(x, xprev, mcu.Z, y, Pt, Pe, mcu.Pt[:, :, t],mcu.alpha[:,t].*mcu.beta[:,t])
end
function quuprev(
    u::Int,
    uprev::Int,
    mcx::MarkovChain,
    t::Int,
    y::Real,
    Pt::TransitionDistribution,
    Pe::EmissionDistribution,
)
    ptu(u, uprev, mcx.Z, y, Pt, Pe, mcx.Pt[:, :, t],mcx.alpha[:,t].*mcx.beta[:,t])
end

@doc """
Returns MAP path.
"""
function viterbi(mc::MarkovChain)
    Z, T = mc.alpha.mat |> size
    paths = ones(Z, T) .|> Int
    paths[:, 1] = 1:Z
    weights = ones(Z, T) .|> ULogarithmic
    weights[:, 1] = mc.P1
    tmp = ones(Z)
    for t = 2:T
        for z = 1:Z
            tmp .= (mc.Pt[z, :, t] ./ mc.alpha[:, t-1] ./ mc.beta[:, t]) .* weights[:, t-1]
            source, weight = argmax(tmp), maximum(tmp)
            weights[z, t] = weight
            paths[z, t] = source
        end
    end
    tmp = ones(T) .|> Int
    tmp[T] = argmax(weights[:, T])
    for t in 2:T |> reverse
        tmp[t-1] = paths[tmp[t], t]
    end
    tmp
end

function norm(mc::MarkovChain)
    mc.alpha[:,1] .* mc.beta[:, 1] |> sum
end