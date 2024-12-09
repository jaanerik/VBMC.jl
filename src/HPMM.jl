function createFirstDistribution(U::Int, X::Int)
    """
    Dirichlet alphas are currently 1,2,...,U*X due to happenstance.
    """
    Vector(1:U*X) |> Dirichlet |> rand |> (x -> Categorical(x))
end

function createNextTimestepDistribution(U::Int, X::Int)
    """
    Weights sampling fixed for now.
        For random fixed i,j sum[i,j,:,:] == 1
    """
    mat = zeros((U,X,U,X))
    for u in U
        for x in X
            mat[u,x,:,:] = reshape(rand(Dirichlet(1:U*X)), (U,X))
        end
    end
    mat
end

# struct ReshapedCategorical
#     d::Categorical
#     p::Vector<Real>
#     X::Int
#     support
    
# end

# Base.show(io::IO, n::ReshapedCategorical) = print(io, "$(n.table)")

# function pdf(d::EmissionDistribution,u::Int,x::Int,y<:Real)
#     pdf(d.Y_distr, )
# end