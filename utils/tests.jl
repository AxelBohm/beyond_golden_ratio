using LinearAlgebra
using Plots

include("methods.jl")
include("utils.jl")
include("plotting.jl")

ϵ = 0.1
s = 0.3

R = [-1 ϵ; -ϵ 0]
S = [s s; 1 1.0]

z0 = [1, 0, 1, 0]
sol = [0, 1, 0, 1]

# setup problem
F = generate_F(R, S)
compute_norm_grad = generate_gradient_norm(F)
eval_vi = generate_eval_vi(F, sol)
compute_primal_dual_gap = generate_pd_gap(ϵ, s)

@test compute_norm_grad([0, 1, 0, 1]) == 0


R = [-0.6 -0.3; 0.6 -0.3]
S = [0.9 0.5; 0.8 0.4]

xs = 0.82535
ys = 0.050485
sol = [xs, 1-xs, ys, 1-ys]


# setup problem
F = generate_F(R, S)
compute_norm_grad = generate_gradient_norm(F)
eval_vi = generate_eval_vi(F, sol)


@test F([0, 1, 0, 1]) ≈ [0.1875, 0, -3, 0]


@test simplex_projection([1, 0]) == [1, 0]
@test simplex_projection([2/3, 1/3]) == [2/3, 1/3]
@test simplex_projection([1/4, 3/4]) == [1/4, 3/4]
@test simplex_projection([0, 1]) == [0, 1]
@test simplex_projection([2, 2]) == [1/2, 1/2]
@test simplex_projection([0, 4]) == [0, 1]

@test proj_twice_simplex([0, 1, 0, 1]) == [0, 1, 0, 1]
@test proj_twice_simplex([1, 0, 1, 0]) == [1, 0, 1, 0]
@test proj_twice_simplex([2/3, 1/3, 1, 0]) == [2/3, 1/3, 1, 0]
@test proj_twice_simplex([2, 2, 1, 0]) == [1/2, 1/2, 1, 0]
