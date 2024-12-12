"""
    Homogeneous Hidden Pairwise Markov Model
"""

struct HPMM
    U<:Array{<:Int}
    X<:Array{<:Int}
    Y<:Array{<:Real}
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

struct EmissionDistribution <: Sampleable{Univariate, Discrete }
    emission::Normal
    U::Int
    X::Int
end

struct HpmmDistribution <: Sampleable{Univariate , Continuous }
    P_first::ReshapedCategorical
    P_tr::TransitionDistribution
    P_em::EmissionDistribution
end

# Distribution construction

function createFirstDistribution(U::Int, X::Int)
    """
    Dirichlet alphas are currently 1,2,...,U*X due to happenstance.
    """
    begin
        Vector(1:U*X) |> 
        Dirichlet |> 
        rand |> 
        (x -> Categorical(x)) |> 
        (d -> ReshapedCategorical(d, U, X))
    end
end

function createTransitionDistribution(U::Int, X::Int)
    """
    Weights sampling fixed for now.
        For random fixed i,j sum[:,:,i,j] == 1
    """
    mat = reshape( rand(Dirichlet(1:U*X),U*X), (U,X,U,X))
    TransitionDistribution(mat)
end

function createEmissionDistribution(U::Int, X::Int)
    EmissionDistribution(Normal(),U,X)
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
    noise = rand(ed.emission)
    x + noise
end

function Base.rand(
    P1::ReshapedCategorical,
    P_tr::TransitionDistribution,
    T::Int)
    pmm = Array{Tuple{Int,Int},2}(undef, 10, 1)
    pmm[1] = rand(P1)
    for t in 2:T
        uprev, xprev = pmm[t-1][1], pmm[t-1][2]
        pmm[t] = rand(P_tr, uprev, xprev)
    end
    pmm
end

function Base.rand(hpmm::HpmmDistribution, T::Int)
    """
    Returns Matrix of size T of a Triplet Markov Chain,
    where first element in (U,X) tuple and Y is observed variable.
    [
        (U,X)   Y
        ....
    ]
    """
    pmm = rand(hpmm.P_first, hpmm.P_tr, T)
    ems = pmm .|> (p -> rand(Pe, p[1], p[2]))
    hcat(pmm, ems)
end

# Helpers

function reshapeIndex(i::Int, U::Int, X::Int)
    i = i-1
    j = (i%U)+1
    k = div(i,U)+1
    j,k
end

function reshapeIndex(t::Tuple{Int, Int}, U, X)
    (t[2]-1)*U + t[1]
end

function getWeights(rc::ReshapedCategorical)
    reshape(rc.d.p, (rc.U, rc.X))
end

function flatten(x::AbstractArray{<:Real}) reduce(vcat, x) end

# function pdf(d::EmissionDistribution,u::Int,x::Int,y<:Real)
#     pdf(d.Y_distr, )
# end