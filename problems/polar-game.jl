using LinearAlgebra: norm
using FStrings
using ForwardDiff

include("../utils/utils.jl")
"""
   From "Escaping limit cycles, Example 3"
   Limit cycles at ||(x,y)|| = 1 and ||(x,y)|| = 1/2
"""
# a = 3/2 # => ρ ∈ (-2/L, -1/L)     Here, no convergence guarantees work
# a = 1   # => ρ ∈ (-1/L, -2/3L)    Here, the guarantees of Pethick et al. apply
# a = 1/2 # => ρ ∈ (-1/4L, -1/4.5L) Here, my convergence guarantees work

function polar_game(a, n_grad_eval=1000)
    """The larger a, the more difficult the problem becomes."""

    folder_name = f"Polar-Game a={a:.2f}"

    z0 = [-1., -0.8]
    sol = [0., 0.]

    proj(u) = clamp.(u, -11/10, 11/10)
    F, H = generate_polar(a)
    polar_game_VI = VariationalInequality(F, proj, sol=sol, H=H)
    L, ρ = compute_problem_params(polar_game_VI)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)

    return polar_game_VI, params
end

function generate_polar(a)
    # setup problem
    ψ(x,y) = 1/4 * a * x * (-1+x^2+y^2)*(-1 + 4x^2 + 4y^2)

    function F(u)
        x, y = u[1], u[2]
        return [ ψ(x, y) - y; ψ(y, x) + x]
    end

    Hamiltonian(u) = norm(F(u))^2/2
    H(u) = ForwardDiff.gradient(Hamiltonian, u)

    return F, H
end

function compute_problem_params(VI::VariationalInequality, lb=-1.1, ub=1.1)

    F = VI.F
    J(u) = ForwardDiff.jacobian(F, u)
    norm_hessian(u) = opnorm(J(u))

    x = lb:0.01:ub
    y = copy(x)
    Lip = 0.
    ρ = 0.
    for xi in x, yj in y
        u = [xi, yj]
        Lip = max(Lip, norm_hessian(u))
        Fu = F(u)
        if norm(Fu) > 0.0000001
            ρ = min(ρ, Fu' * (u - VI.sol) / norm(Fu)^2)
        end
    end
    ρ = -2ρ #see definition of ρ

    return Lip, ρ
end
