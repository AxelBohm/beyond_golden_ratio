using LinearAlgebra

include("../utils/utils.jl")

function ryu_huber()
    folder_name  = "ryu-huber"

    n_grad_eval = 5000
    # L = 1. way too slow
    L = .2
    z0 = [1., 0.]
    sol = [0., 0.]

    F = generate_F_huber()
    problem = VariationalInequality(F, u -> u, sol=sol)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end

function generate_F_huber()

    ϵ = 5 * 10^(-5)
    δ = 0.01

    # gradient of huber function
    huber_grad(x) = norm(x) > ϵ ? (ϵ*sign(x)) : x

    function F(u)
        x, y= u[1], u[2]
        return [(1-δ)*huber_grad(x) + δ*y; -(1-δ)*huber_grad(y) - δ*x]
    end
    return F
end