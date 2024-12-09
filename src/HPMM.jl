function createFirstDistribution(U::Int, X::Int)
    Vector(1:U*X) |> Dirichlet |> rand |> (x -> Categorical(x))
end

struct ReshapedCategorical
    d::Categorical
    p::Vector<Real>
    X::Int
    support
    
end

Base.show(io::IO, n::ReshapedCategorical) = print(io, "$(n.table)")

# function pdf(d::EmissionDistribution,u::Int,x::Int,y<:Real)
#     pdf(d.Y_distr, )
# end