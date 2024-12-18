"""
    Markov Chain and forward-backward, Viterbi algorithm
    implemented
"""

@doc """
Let matrix element i,j,k denote
the probability P_k(z_i | z_j)
at timestep k, prev element z_j.

P_1 is meaningless and exists for notational comfort.
"""
struct InhomogeneousTransitionDistribution
    mat::AbstractArray{Float64,3}
end

struct MarkovChain
    P1::Categorical
    Pt::InhomogeneousTransitionDistribution

    MarkovChain(Z::Int, T::Int) =
        new(Categorical(Z), InhomogeneousTransitionDistribution(ones(Z, Z, T) ./ Z))
end

function forward_backward!(alpha::Alpha, beta::Beta, mc::MarkovChain) end
