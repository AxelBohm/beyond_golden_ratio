using LinearAlgebra
using Random
using Distributions
"""
   Two test problems from Nemirovski et al. "Robust stochastic approximation approach to stochastic programming"
"""

include("../utils/utils.jl")

function gen_matrix_problem1(n=50)

    n_grad_eval = 1000
    z0 = ones(2*n)/n

    folder_name = "nemirovski1"

    # problem setup
    A = nemirovski1(n)
    L = opnorm(A)

    problem = MatrixGame(A, proj_twice_simplex)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end

function nemirovski1(n, α=1)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i, j] = ((i + j - 1) / ( 2n - 1 ))^α
        end
    end
    return A
end

# -----------------------------------------------------------------------------
# second problem
# -----------------------------------------------------------------------------


function gen_matrix_problem2(n=50)

    folder_name = "nemirovski2"

    n_grad_eval = 1000
    z0 = ones(2*n)/n

    # problem setup
    A = nemirovski2(n)
    L = opnorm(A)

    problem = MatrixGame(A, proj_twice_simplex)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end

function nemirovski2(n, α=1)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i, j] = ((abs(i - j) + 1) / ( 2n - 1 ))^α
        end
    end
    return A
end

# -----------------------------------------------------------------------------
# third problem
# -----------------------------------------------------------------------------


function gen_random_matrix_game(n=50)

    Random.seed!(42)
    folder_name = "random_matrix_game"

    n_grad_eval = 10000
    z0 = ones(2*n)/n

    # problem setup

    A = rand(Normal(0, 1), n, n)
    L = opnorm(A)

    problem = MatrixGame(A, proj_twice_simplex)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end