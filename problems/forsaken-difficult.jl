using LinearAlgebra
include("../utils/utils.jl")

function forsaken_difficult(n_grad_eval=400)

    L_inverse = 0.08
    L = 1/L_inverse
    ρ = 0.477761
    # lr = L_inverse
    z0 = [0.5, 0.5]
    # z0 = [-1.5, -1.5] # for this starting point HGD converges to a spurious minimum of the Hamiltonian
    sol = [0.0780267, 0.411934]

    folder_name = "Forsaken"

    ψ(x) = x/2 - 2x^3 + x^5
    ψ_p(x) = 1/2 - 6x^2 + 5x^4

    function F(u)
        x, y = u[1], u[2]
        return [ψ(x) + (y-0.45); -x + ψ(y)]
    end
    # H(u) = (x/2 - 2x^3 + x^5 + (y-0.45))^2 + (-x + y/2 - 2y^3 + y^5)^2
    function H(u)
        x, y = u[1], u[2]
        return [(ψ(x)+ (y-0.45))*ψ_p(x) - (-x + ψ(y)); (ψ(x) + (y-0.45)) + (-x + ψ(y))*ψ_p(y)]
    end


    proj(u) = clamp.(u, -10, 10)

    problem = VariationalInequality(F, proj, sol=sol, H=H)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=ρ)

    return problem, params
end