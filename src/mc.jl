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
    mat::AbstractArray{Float64,3}
end

@doc """
Creates Markov Chain with uniform P1, P_t for each t > 2.
"""
struct MarkovChain
    P1::Categorical
    Pt::InhomogeneousTransitionDistribution
    alpha::AlphaBeta
    beta::AlphaBeta

    MarkovChain(Z::Int, T::Int) =
        new(Categorical(Z), InhomogeneousTransitionDistribution(ones(Z, Z, T) ./ Z))
end

function forward_backward!(alpha::AlphaBeta, beta::Beta, mc::MarkovChain) end
