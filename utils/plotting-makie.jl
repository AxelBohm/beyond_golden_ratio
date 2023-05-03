using LinearAlgebra
using CairoMakie

include("utils.jl")

cycle = Cycle([:color, :marker], covary=true)
# fontsize_theme = Theme(fontsize = 30)
fontsize_theme = Theme(fontsize = 28)
set_theme!(fontsize_theme, Lines=(linewidth=4, cycle=cycle), Scatter=(cycle=cycle,))

function my_scatterline!(ax, x, y, k, num_algs, num_mark, label)
    """I frequently want a line with custom spaced markers. While there is a
    scatterline function in Makie it places a marker on every datapoint. The
    easiest way to get the desired behaviour (which is similar to matplotlibs
    `mark_every`) is to merge a line and scatterplot."""

    step = Int(floor(length(y) / num_mark))
    start = k * Int(ceil(step / num_algs))

    # Skip markers if no valid values to avoid error (DID NOT HELP)
    if length(y) == 0 || step < 1
        println("warning")
        return
    end

    try
        clamp!(y, -100000, 1000000)
        lines!(ax, x, y; label = label)
        scatter!(ax, x[start:step:length(x)], y[start:step:length(x)]; markersize = 20, label = label)
    catch e
        println("Some problems with plotting")
    end

end


function plot_heatmap(VI::VariationalInequality, path)
    """Plot to illustrate how the VI behaves on the feasible set"""
    if isfile("$path/minty-vi-sign-heatmap.png")
        return
    end
    sol = VI.sol
    x = 0:0.001:1
    y = copy(x)
    data = fill(0., length(x), length(y))
    data_ρ = copy(data)
    data_sign = copy(data)
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        tmp = eval_vi(VI, [xi, 1-xi, yj, 1-yj])
        data[j, i] = tmp
        data_sign[j, i] = sign.(tmp)
        data_ρ = tmp/ compute_norm_grad(VI, [xi, 1-xi, yj, 1-yj], 1)
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    heatmap!(ax, x, y, data', title="Minty VI heatmap")
    scatter!(ax, [sol[1]], [sol[3]], marker=:star5, markersize=30, color=:red, label="solution")
    axislegend()
    save("$path/minty-vi-heatmap.png", fig)

    fig = Figure()
    ax = Axis(fig[1, 1])
    heatmap!(ax, x, y, data_sign', colorbar=false, c=cgrad([:yellow, :purple], 2, categorical=true))
    scatter!(ax, [sol[1]], [sol[3]], marker=:star5, markersize=30, color=:red, label="solution")
    axislegend()
    save("$path/minty-vi-sign-heatmap.png", fig)
    println(minimum(data_ρ))
end

function norm_grad_plot(data, num_mark=6)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="squared operator norm")
    n = length(data)

    for (k, alg) in enumerate(data)
        x = alg.x_ticks
        y = alg.store_grad
        my_scatterline!(ax, x, y, k, n, num_mark, alg.label)
    end
    axislegend(ax, merge=true)
    return fig
end


function norm_grad_log_plot(data, path, num_mark=6)
    """Semi-log plot for operator norm."""
    xscale = identity
    ax, fig = norm_grad_helper(data, num_mark, xscale)

    create_separate_figure_for_legend(ax, "legend", path)
    axislegend(ax, merge=true)

    return fig
end


function norm_grad_loglog_plot(data, path, num_mark=6)
    """loglog plot with reference lines for convergence rates."""
    xscale = log10
    ax, fig = norm_grad_helper(data, num_mark, xscale)

    # reference lines
    x = data[1].x_ticks .+ 1
    lines!(ax, x, 10*data[1].store_grad[1] ./ x, label="1/k", color=:grey, linestyle=:dash)
    lines!(ax, x, 10*data[1].store_grad[1] ./ x.^2, label="1/k^2", color=:grey, linestyle=:dot)
    # Legend(fig[1, 2], ax, merge=true)
    # axislegend(ax, merge=true)
    # separate legend
    create_separate_figure_for_legend(ax, "legend_with_helper", path)
    return fig
end

function create_separate_figure_for_legend(ax, name, path)
    fig_legend = Figure()
    Legend(fig_legend[1, 1], ax, merge=true, orientation=:horizontal)
    save("$path/$name.png", fig_legend)
end


function norm_grad_helper(data, num_mark, xscale)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="squared operator norm", yscale=log10, xscale=xscale)
    n = length(data)
    # min = 10000
    max = 0
    for (k, alg) in enumerate(data)
        inds = alg.store_grad .> 10^(-15)  # cut off here for better plots
        y = alg.store_grad[inds]
        x = alg.x_ticks[inds] .+ (xscale == log10)
        # min = minimum([min, minimum(y)])
        if !any(isnan, y) && !any(isinf, y)
            max = maximum([max, maximum(y)])
            my_scatterline!(ax, x, y, k, n, num_mark, alg.label)
        end
        # upper_lim = minimum([data[1].store_grad[1]*2, 2*max])
        # ylims!(high=upper_lim)
    end
    # axislegend(ax, merge=true)
    # Legend(fig[1, 2], ax, merge=true)
    # If one method deminates all others I don't care
    # ylims!(low=maximum([min, 10^(-15)]))
    # ylims!(maximum([min, 10^(-15)]), minimum([data[1].store_grad[1]*2, max]))
    # println(upper_lim)
    # ylims!(high=upper_lim)
    return ax, fig
end




function pd_gap_plot(data, num_mark=6)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="primal dual gap")
    n = length(data)
    for (k, alg) in enumerate(data)
        my_scatterline!(ax, alg.x_ticks, alg.pd_gap, k, n, num_mark, alg.label)
    end
    axislegend(ax, merge=true)
    return fig
end

function generate_lr_plot(data, L=missing, num_mark=6)
    """Plot the learning rates across iterations. Especially interesting for the adaptive methods."""

    fig = Figure()
    # ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="step size", yscale=log10)
    ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="step size")
    n = length(data)
    for (k, alg) in enumerate(data)
        inds = alg.lr .> 0
        x = alg.x_ticks[inds]
        y = alg.lr[inds]
        # my_scatterline!(ax, x, y, k, n, num_mark, alg.label)
        scatter!(ax, x, y, label=alg.label)
    end
    if !ismissing(L)
        x = data[1].x_ticks
        lines!(ax, x, 1/L*ones(length(x)); label="1/L", color=:grey, linestyle=:dash)
    end
    axislegend(ax, merge=true)
    return fig
end


function generate_lr_ratio_plot(data, num_mark=6)
    """Plot the learning rates across iterations. Especially interesting for the adaptive methods."""

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="operator evaluations", ylabel="extrapolation / update")
    n = length(data)
    for (k, alg) in enumerate(data)
        inds = alg.lr_ratio .> 0
        x = alg.x_ticks[inds]
        y = alg.lr_ratio[inds]
        my_scatterline!(ax, x, y, k, n, num_mark, alg.label)
    end
    axislegend(ax, merge=true)
    return fig
end

function create_performance_plots(data, VI::AbstractVI, problem_params, algorithms)
    """Create various plots illustrating the performance of the generated trajectory and save them."""

    path = problem_params.path

    fig = norm_grad_plot(data)
    save("$path/norm-grad-iter.png", fig)

    try
        fig = norm_grad_log_plot(data, path)
        save("$path/norm-grad-iter-log.png", fig)

        fig = norm_grad_loglog_plot(data, path)
        save("$path/norm-grad-iter-loglog.png", fig)
    catch e
        println("diverging methods kill the log plots")
    end

    try
        fig = generate_lr_plot(data, problem_params.L)
        save("$path/step-size.png", fig)

        fig = generate_lr_ratio_plot(data)
        save("$path/step-size-ratio.png", fig)
    catch e
        println("lr plot not working")
    end

end

function plot_iterates(data, VI::AbstractVI, path, num_mark=10)
    """Plot trajectory of iterates (only possible if problem is 2-d)"""

    F = VI.F
    fig = Figure()
    ax = Axis(fig[1, 1])
    n = length(data)
    for (k, alg) in enumerate(data)
        mat = reshape(alg.iterates, 2, :)
        x = convert(Vector{Float64}, mat[1, :])
        y = convert(Vector{Float64}, mat[2, :])
        clamp!(x, -10, 10)
        clamp!(y, -10, 10)
        my_scatterline!(ax, x, y, k, n, num_mark, alg.label)
    end

    minval = -1.5
    maxval = 1.5
    if occursin("nonconvex-linear", path); maxval = 2 end
    if occursin("follow-the-ridge", path); minval, maxval = -5, 5 end
    f(u) = Point2f(-F(u))
    streamplot!(ax, f, minval..maxval, minval..maxval, color=:grey, linewidth=0.4)
    axislegend(merge=true)
    xlims!(minval, maxval)
    ylims!(minval, maxval)
    save("$path/iterates.png", fig)
end