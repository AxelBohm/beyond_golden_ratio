using LinearAlgebra

"""
L(x, y) = 1/2 x^T H x  - h^T x - < Ax -b , y >
is 1-smooth
"""

include("../utils/utils.jl")

function linearly_constrained_qp(dim=100, n_grad_eval=10^4)
    folder_name  = "constrained-QP"

    L = 1.
    z0 = zeros(2*dim)

    # setup problem
    F, H = generate_F_QP(dim)
    problem = VariationalInequality(F, u -> u, H=H, dim=2*dim)
    params = ProblemParams(z0, n_grad_eval, folder_name, L, ρ=0)

    return problem, params
end

function generate_F_QP(dim)

    # ones in anti-diag and -1 in off-anti-diag
    A = 1/4 * [(i==dim-j+1) + (i==dim-j)*(-1) for i in 1:dim, j in 1:dim]
    b = 1/4 * ones(dim)
    h = zeros(dim)
    h[end] = 1/4
    H = 2 * A' * A

    function F(u)
        x = @view u[1:dim]
        y = @view u[dim+1:end]
        return [H*x - h - A'*y ; A*x - b]
    end
    function Hamiltonian(u)
        x = @view u[1:dim]
        y = @view u[dim+1:end]
        return [H'*(H*x - h - A'*y) + A'*(A*x - b) ; - A*(H*x - h - A'*y)]
    end

    return F, Hamiltonian
end
