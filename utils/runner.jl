include("utils.jl")
include("plotting-makie.jl")

function run_instance(algorithms, get_problem)
    """Give a list of algorithms and a problem instance run all algorithms on this problem and create plots."""
    data = []
    VI, problem_params = get_problem()
    for algo in algorithms
        cb = Callback(label=algo.label)
        algo.method(VI, problem_params, cb; algo.params...)
        push!(data, cb)
    end
    create_performance_plots(data, VI, problem_params, algorithms)
end