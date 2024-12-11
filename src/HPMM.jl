"""
    Homogeneous Hidden Pairwise Markov Model
"""

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

struct HPMM <: Sampleable{Univariate , Continuous }
    P_first::ReshapedCategorical
    P_tr::TransitionDistribution
    P_em::EmissionDistribution
end

# Constructors

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