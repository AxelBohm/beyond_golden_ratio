using LinearAlgebra
using Random
using Distributions

include("../utils/utils.jl")

function lower_bound(param, n_grad_eval=2000)
    """param corresponds to a^2 and should be in (0, \\infty). The smaller it the
    more difficult the problem becomes."""

    z0 = [0.5, 0]
    sol = [0, 0]
    proj = identity

    a = sqrt(param)
    b = -1
    F, L, ρ, H = generate_F_lb(a, b)

    lower_bound = VariationalInequality(F, proj, sol=sol, H=H)
    folder_name = "lower-bound a=sqrt($(param))"

    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)
    return lower_bound, params
end

function generate_F_lb(a, b)

    lipschitz_const = sqrt(a^2 + b^2)
    ρ = -2b / (a^2 + b^2) # compare pethick2022escaping

    function F(z)
        x, y = z[1], z[2]
        return [a*y + b*x, b*y - a*x]
    end
    function H(z)
        x, y = z[1], z[2]
        return [(a*y+b*x)*b + (a*x-b*y)*a, (a*y+b*x)*a + (b*y-a*x)*b]
    end

    return F, lipschitz_const, ρ, H
end