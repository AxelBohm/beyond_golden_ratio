using LinearAlgebra
using Random
"""
    From Juditski & Nemirovki tutorial. Problem Policemen and Burglar.
"""

include("../utils/utils.jl")

function policeman_burglar(n=50)

    folder_name = "policeman_burglar"
    n_grad_eval = 1000
    z0 = ones(2*n)/n

    Random.seed!(42)
    A = policeman_and_burglar_matrix(n)
    L = opnorm(A)

    problem = MatrixGame(A, proj_twice_simplex)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end

function policeman_and_burglar_matrix(n, th=0.8)
    w = abs.(randn(n))
    C = reshape(abs.([i - j for i in 1:n, j in 1:n]), (n, n))
    A = w .*(1 .- exp.(-th .* C))
    return A
end