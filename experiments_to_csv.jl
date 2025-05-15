using RxInfer
using TOML
using VBMC
using Distributions
using Random
using StatsBase
using LogarithmicNumbers
using CSV
using DataFrames
using LinearAlgebra

Random.seed!(1234)
if "test" in readdir()
    cd("test")
end

function elbo(mcx, mcu)
    "Assumes P1, Pt, Pe as global vars."
    elbo = 0.
    elbo += Iterators.product(1:U, 1:X) .|>
    (
        ((u,x),) -> log(P1[u,x]*Pe[u,x,hpmm.Y[1]]/(mcx.P1[x] * mcu.P1[u])) * mcx.P1[x] * mcu.P1[u]
    ) |> sum
    elbo2 = 0.
    for t in 2:T
        elbo2 += Iterators.product(1:U, 1:X, 1:U, 1:X) .|>
        (
            ((u,x,uprev,xprev),) -> 
            begin
                qtu = VBMC.quuprev(u, uprev, mcx, t, hpmm.Y[t], Pt, Pe)
                qtx = VBMC.qxxprev(x, xprev, mcu, t, hpmm.Y[t], Pt, Pe)
                log(
                    Pt[u,x,uprev,xprev]*Pe[u,x,hpmm.Y[t]]/(qtu * qtx)
                ) * mcx.Pt[x,xprev,t] * mcu.Pt[u,uprev,t]
            end
        ) |> sum
    end
    elbo + elbo2
end

struct EmissionNode{T <: Real} <: ContinuousUnivariateDistribution
    y :: T
    wt :: T
end

struct MetaTransition
    mat :: Matrix{Float64}
end

struct HmmTransition{T <: Real} <: DiscreteMultivariateDistribution
    wpast :: AbstractArray{T}
    wt :: AbstractArray{T}
end

vars = TOML.parsefile("Constants.toml")
T, U, X, Y = vars["T"], vars["U"], vars["X"], vars["Y"]
T = 20
noise_std = vars["noise_std"]P
function emissionf(randVal, u, x)
    randVal .+ x
end
function revf(emission, u, x)
    emission .- x
end #the dot vectorizes, could use X instead
emission = Emission(emissionf, revf)

function create_model()
    P1 = begin
        ones(U*X) |>
        Dirichlet |>
        rand |>
        (x -> Categorical(x)) |>
        (d -> ReshapedCategorical(d, U, X))
    end
    Pt = reshape(rand(ones(U*X) |> Dirichlet, U * X), (U, X, U, X)) |> TransitionDistribution
    Pe = EmissionDistribution{Continuous}(Normal(0,noise_std), emission, U, X)
    dist = HpmmDistribution(P1, Pt, Pe)
    hpmm = rand(dist, T)
    dist, hpmm
end

function find_viterbiu(posteriors_w)
    posteriors_w .|> dist -> (sum(reshape(dist.p, U, X), dims=2) |> Iterators.flatten |> collect |> argmax)
end
# vmpus = find_viterbiu(a.posteriors[:w][20]);
function get_digits(val, base) digits(val, base=base, pad=T) .|> x -> x+1 end

function get_pdf(xs)
    prevalphas = ones(U) .|> ULogarithmic
    for t in 1:T
        if t == 1
            p1 = 0.
            for u in 1:U
                prevalphas[u] = P1[u,xs[t]] * Pe[u,xs[t],hpmm.Y[t]]
            end
        else
            pts = Iterators.product(1:U, 1:U) .|> ((u,uprev),) -> Pt[u,xs[t],uprev,xs[t-1]] * Pe[u,xs[t],hpmm.Y[t]]
            prevalphas = pts' * prevalphas
        end
    end
    prevalphas |> sum
end

function get_percentile(path)
    len = length(probabilities)
    pos = length(probabilities[probabilities .<= get_pdf(path)])
    pos/len
end

function get_x(w)
    VBMC.reshapeindex(w, U, X)[2]
end
function get_u(w)
    VBMC.reshapeindex(w, U, X)[1]
end

@node HmmTransition Stochastic [wt, wp]

@rule HmmTransition(:wp, Marginalisation) (q_wt :: Categorical, meta::MetaTransition) = begin
    G = q_wt.p
    A = meta.mat
    ηs = exp.(log.(A)' * G)
    νs = ηs ./ sum(ηs)
    return Categorical(νs...)
end
@rule HmmTransition(:wt, Marginalisation) (q_wp :: Categorical, meta::MetaTransition) = begin
    F = q_wp.p
    A = meta.mat #reshape(Pt.mat, U*X, U*X)
    ηs = exp.(log.(A) * F) # .* B # | clamp(⋅,tiny,one) | exp maybe or smth?
    νs = ηs ./ sum(ηs)
    return Categorical(νs...)
end
@marginalrule HmmTransition(:wt_wp) (q_wt::Categorical, q_wp::Categorical, meta::MetaTransition) = begin
    A = meta.mat
    #This is copied from ReactiveMP.jl transition/marginals.jl, however I think this marginalrule is never called
    B = Diagonal(probvec(q_wt)) * A * Diagonal(probvec(q_wp))
    P = map!(Base.Fix2(/, sum(B)), B, B) # inplace version of B ./ sum(B)
    return Contingency(P, Val(false))
    # F, G = q_wp.p, q_wt.p
    # AA = meta.mat
    # ηs = exp.(log.(A) * F) # .* B
    # ps = ηs .* G
    # ps = ps ./ sum(ps)
    # return (wt = Categorical(ps...), wp = q_wp)
end
@average_energy HmmTransition (q_wt::Categorical, q_wp::Categorical, meta::MetaTransition) = begin
    A = meta.mat
    G, F = q_wp.p, q_wt.p
    F' * log.(A) * G
end

@node EmissionNode Stochastic [y, wt]

@rule EmissionNode(:wt, Marginalisation) (q_y::PointMass, ) = begin 
    B = map(1:U*X) do w pdf(Normal(0,noise_std), q_y.point-get_x(w)) end
    return Categorical(B./sum(B)...)
end
@rule EmissionNode(:y, Marginalisation) (q_wt :: Categorical, ) = begin
    B = map(1:U*X) do w pdf(Normal(0,noise_std), y-get_x(w)) end
    G = q_wt.p
    return PointMass(exp(log.(B)' * G))
end

@marginalrule EmissionNode(:y_wt) (q_wt::Categorical, q_y::PointMass) = begin
    B = map(1:U*X) do w pdf(Normal(0,noise_std), y-get_x(w)) end
    G = q_wt.p
    ps = log(B) .* G
    ps = ps./sum(ps)
    return (y = q_y, wt = Categorical(ps...)) 
end
@average_energy EmissionNode (q_y::PointMass, q_wt::Categorical) = begin
    B = map(1:U*X) do w pdf(Normal(0,1), q_y.point-get_x(w)) end
    F = q_wt.p
    F' * log.(B)
end

constraints = @constraints begin
    q(w) = q(w[begin])..q(w[end])
end

init = @initialization begin
    # Note T is hardcoded for now
    for t in 2:T
        q(w[t]) = vague(Categorical, U*X)
    end
end;

function infer_vmp(y, model)
    infer(
        model = model(),
        constraints = constraints,
        initialization = init,
        data = (y = y,),
        options = (limit_stack_depth = 500,),
        free_energy = true,
        showprogress=true,
        iterations = 20,    
        warn = false
    )
end
function find_viterbi(posteriors_w)
    posteriors_w .|> dist -> (sum(reshape(dist.p, U, X), dims=1) |> Iterators.flatten |> collect |> argmax)
end

function find_belief_prop_path()
    mcu = MarkovChain(U, T)
    mcx = MarkovChain(X, T)

    function norm(A::AbstractArray; p = 2)
        sum(abs.(A) .^ p)^(1 / p)
    end

    elboold = 0.
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
        elbonew = elbo(mcx ,mcu)
        if (elbonew - elboold)/elboold |> abs < 1.e-17
            break
        end
        elboold = elbonew
    end
    VBMC.viterbi(mcx)
end

function find_all_probabilities()
    probabilities = zeros(0) .|> ULogFloat64
    maxprob = ULogFloat64(0.)
    best_path = zeros(T) .|> Int
    for x in 0:(X^T-1)
        xs = get_digits(x, X)
        total_prob = get_pdf(xs)
        if maxprob < total_prob
            maxprob = total_prob
            best_path .= xs
        end
        append!(probabilities, total_prob)
    end
    probabilities, best_path
end

@eval vmp_percs, bp_percs, original_percs = zeros(0), zeros(0), zeros(0)
@eval vmp_paths, bp_paths, original_paths, naive_paths = zeros(0) .|> Int, zeros(0) .|> Int, zeros(0) .|> Int, zeros(0) .|> Int
function iterate_append_percs!()
    @eval dist, hpmm = create_model()
    @eval P1,Pt,Pe = dist.P1, dist.Pt, dist.Pem
    @eval probabilities, best_path = find_all_probabilities()
    @eval p1 = P1.d.p

    @eval @model function hidden_markov_model(y)
        w[1] ~ Categorical(p1)
        y[1] ~ EmissionNode(w[1])
        for t in 2:length(y)
            w[t] ~ HmmTransition(w[t-1]) where { meta = MetaTransition(reshape(Pt.mat, U*X, U*X)) }
            y[t] ~ EmissionNode(w[t])
        end
    end

    @eval vmp_path = infer_vmp(hpmm.Y, hidden_markov_model) |> a -> a.posteriors[:w] |> last |> find_viterbi
    @eval vmp_perc = vmp_path |> get_percentile
    @eval bp_path = find_belief_prop_path()
    @eval bp_perc = bp_path |> get_percentile
    @eval original_perc = hpmm.X |> get_percentile
    
    function hamming(xs1,xs2) sum(xs1 .!= xs2) end
    f = x -> abs(x-1) < abs(x-2) ? 1 : 2
    naivepath = hpmm.Y .|> f

    append!(vmp_paths, hamming(best_path, vmp_path))
    append!(bp_paths, hamming(best_path, bp_path))
    append!(original_paths, hamming(best_path, hpmm.X))
    append!(naive_paths, hamming(best_path, naivepath))
    append!(vmp_percs, vmp_perc), append!(bp_percs, bp_perc), append!(original_percs, original_perc)
end

for k in 1:50
    iterate_append_percs!()
    if k % 5 == 0
        println(k*2, "%")
    end
end
BP,VMP,ORIG,hamming_vmp,hamming_bp,hamming_orgin,hamming_naive
df = DataFrame(
    BP = bp_percs, VMP = vmp_percs, ORIG = original_percs,
    hamming_vmp = vmp_paths, hamming_bp = bp_paths, hamming_orig = original_paths,
    hamming_naive = naive_paths
    )
CSV.write("uniform_dirichlet_percs_std125_t20.csv", df)