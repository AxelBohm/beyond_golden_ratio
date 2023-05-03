include("../utils/utils.jl")
include("../utils/runner.jl")
include("../utils/methods.jl")

include("../problems/problems.jl")


###############################################################
# spotlight showcasing the benefit of adaptive stepsizes.

algorithms = [algorithm(eg, "EG", (; γ=1.)),
              algorithm(ogda, "OGDA", (; γ=1.)),
              algorithm(golden_ratio, "GRAAL"),
              algorithm(golden_ratio, "aGRAAL", (; adaptive=true))]

run_instance(algorithms, policeman_burglar)

###############################################################
# monotone instances
# shows that \phi=2 is better than golden ratio with constant stepsizes

algorithms = [algorithm(ogda, "OGDA", (; γ=1.)),
              algorithm(golden_ratio, "GRAAL ϕ=2", (; ϕ=2.)),
              algorithm(golden_ratio, "GRAAL ϕ=φ")]

run_instance(algorithms, gen_matrix_problem1)
run_instance(algorithms, gen_random_matrix_game)
run_instance(algorithms, linearly_constrained_qp)

###############################################################
# simple weak minty instances

run_instance(algorithms, () -> polar_game(1/3, 2000))

###############################################################
# hard instances


# the "famous" Forsaken
algorithms = [algorithm(golden_ratio, "aGRAAL ϕ=1.4", (; ϕ=1.4, adaptive=true)),
              algorithm(golden_ratio, "aGRAAL ϕ=1.5", (; ϕ=1.5, adaptive=true)),
              algorithm(adaptive_EG, "CurvatureEG+")]
run_instance(algorithms, forsaken_difficult)


# curvature EG diverges here
algorithms = [algorithm(golden_ratio, "GRAAL ϕ=1.3", (; ϕ=1.3)),
              algorithm(golden_ratio, "aGRAAL ϕ=1.3", (; ϕ=1.3, adaptive=true)),
              algorithm(golden_ratio, "aGRAAL ϕ=1.4", (; ϕ=1.4, adaptive=true)),
              algorithm(adaptive_EG, "CurvatureEG+")]
run_instance(algorithms, () -> polar_game(3))


# also showing that smaller γ is really necessary (sometimes)
algorithms = [algorithm(golden_ratio, "aGRAAL ϕ=1.14 γ=1.6", (; ϕ=1.14, γ=1.6, adaptive=true)),
              algorithm(golden_ratio, "aGRAAL ϕ=1.14 γ=1.1", (; ϕ=1.14, γ=1.1, adaptive=true)),
              algorithm(golden_ratio, "aGRAAL ϕ=1.06 γ=1.1", (; ϕ=1.06, γ=1.1, adaptive=true)),
              algorithm(adaptive_EG, "CurvatureEG+")]
run_instance(algorithms, () -> lower_bound(3.7, 20000))