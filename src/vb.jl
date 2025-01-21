struct VB
    T::Int
    U::Int
    X::Int
    P1::ReshapedCategorical
    Pt::TransitionDistribution
    Pe::EmissionDistribution
    Y::AbstractArray{<:Real}
    Qu::MarkovChain
    Qx::MarkovChain

    VB(T, U, X, P1, Pt, Y) = new(T, U, X, P1, Pt, Y, MarkovChain(U, T), MarkovChain(X, T))
end
