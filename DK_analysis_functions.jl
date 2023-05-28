using BehaviorDataNIR, CaAnalysis
using JLD2, PyCall, PyPlot, FlavellBase, StatsBase, ProgressMeter, Statistics, HDF5, Impute, Plots, CaAnalysis
using DSP, TotalVariation, MultipleTesting, Distributions, Colors, ColorSchemes, MultivariateStats, LinearAlgebra, InformationMeasures, Random


# input:
# data_uid: a string of the unique identifier of the dataset
# outputs:
# behavioral variables and neural data in the original data_dict
# note that velocity here is filtered; also note that head_curv_deriv has 1:max_t-1 timepoints

function reload_data(data_uid)
    path_h5 = "/data1/candy/data/processed_h5/$(data_uid)-data.h5"

    data = h5open(path_h5, "r") do h5f
        read(h5f)
    end

    velocity = data["behavior"]["velocity"]
    speed = abs.(velocity)
    rev_bin = data["behavior"]["reversal_vec"]
    rev_start_end = data["behavior"]["reversal_events"]
    stage_x = data["behavior"]["stage_x"]
    stage_y = data["behavior"]["stage_y"]
    
    pumping = data["behavior"]["pumping"]
    
    head_curvature = data["behavior"]["head_angle"]
    head_curv_deriv = CaAnalysis.derivative(head_curvature)
    
    traces_array = data["gcamp"]["trace_array"]
    traces_array_F_F20 = data["gcamp"]["traces_array_F_F20"]
    traces_array_original = data["gcamp"]["trace_array_original"]
    
    time_encounter = data["timing"]["time_food_encounter"]
    
    velocity, speed, rev_bin, rev_start_end, stage_x, stage_y, pumping, head_curvature, head_curv_deriv, traces_array, traces_array_F_F20, traces_array_original, time_encounter
end


function reload_data_neuropal(data_uid)
    path_h5 = "/data1/candy/data/processed_h5/$(data_uid)-data.h5"

    data = h5open(path_h5, "r") do h5f
        read(h5f)
    end

    velocity = data["behavior"]["velocity"]
    speed = abs.(velocity)
    rev_bin = data["behavior"]["reversal_vec"]
    rev_start_end = data["behavior"]["reversal_events"]
    
    head_curvature = data["behavior"]["head_angle"]
    head_curv_deriv = CaAnalysis.derivative(head_curvature)
    
    traces_array = data["gcamp"]["trace_array"]
    traces_array_F_F20 = data["gcamp"]["traces_array_F_F20"]
    traces_array_original = data["gcamp"]["trace_array_original"]
    
    velocity, speed, rev_bin, rev_start_end, head_curvature, head_curv_deriv, traces_array, traces_array_F_F20, traces_array_original
end



# inputs:
# data_uid
# time_encounter: 
# velocity, speed, rev_start_end, pumping, head_curvature, head_curv_deriv: behaviral variables imported from reload_data()
# times (optional kwarg): a series of all useful timepoints
# output:
# a plot

function plot_behaviors(data_uid, time_encounter, velocity, speed, rev_start_end, pumping, head_curvature, head_curv_deriv; times=times, time_exc=66)
    fig, (ax2, ax4, ax3) = plt.subplots(3, figsize=(8,3))

    ax2[:set_xlim](times[1], times[end])
    ax2[:set_ylim](-0.005, 0.105)
    ax2.plot(times, speed[times], c="black")
    ax2.axvline(time_encounter, c="black", linestyle="dotted")
    # ax2.axvline(time_encounter+635, c="green")
    for i = 1:size(rev_start_end,1)
        start = rev_start_end[i,1]
        stop = rev_start_end[i,2]
        ax2.axvspan(start, stop, color="purple", alpha=0.2)
    end
    ax2.set_ylabel("locomotion\nspeed\n(mm/s)", fontsize=11)
    ax2.xaxis.set_ticks([])
    ax2.spines["right"].set_visible([])
    ax2.spines["top"].set_visible([])
    ax2.spines["bottom"].set_visible([])
    
    ax4[:set_xlim](times[1], times[end])
    ax4[:set_ylim](-1.05, 1.05)
    ax4.plot(times, head_curvature[times], c="black")
    ax4.axvline(time_encounter, c="black", linestyle="dotted")
    ax4.set_ylabel("head\ncurvature\n(rad)", fontsize=11)
    ax4.xaxis.set_ticks([])
    ax4.spines["right"].set_visible([])
    ax4.spines["top"].set_visible([])
    ax4.spines["bottom"].set_visible([])

    ax3[:set_xlim](times[1], times[end])
    ax3[:set_ylim](0,20)
    ax3.plot(times, pumping[times].*4, c="black")
    ax3.axvline(time_encounter, c="black", linestyle="dotted")
    ax3.set_ylabel("pumping\nrate (Hz)", fontsize=11)
    ax3.xaxis.set_ticks([])
    ax3.spines["right"].set_visible([])
    ax3.spines["top"].set_visible([])
    ax3.set_xlabel("time (min)", fontsize=11)
    ax3.xaxis.set_ticks([times[1], mean(times), times[end]])
    ax3.xaxis.set_ticklabels(["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"])
end



# inputs:
# stage_x, stage_y: stage information imported from reloade_data()
# data_uid: eg "2022-01-16-01"
# time_encounter: a scalar, imported from reload_data()
# output:
# a plot

function plot_path(data_uid, stage_x, stage_y, times; time_encounter = Int.(time_encounter))
    f = plt.scatter(stage_x[times], stage_y[times], c=times, s=2, cmap="plasma")
    title(data_uid, fontsize=14)
    axis("off")
    ax = plt.colorbar(f)
    # ax.set_ticks([times[1], mean(times), times[end]])
    # ax.set_ticklabels(["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"])
    ax.set_ticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
    ax.set_ticklabels(["0", "2", "4", "6", "8", "10", "12", "14", "16"])
    ax.set_label(label="time (min)", fontsize=14)
    
    if isempty(time_encounter)==false
        plt.scatter(stage_x[time_encounter], stage_y[time_encounter], marker="d", c="black")
    end
end



function unitRange_trace(trace)
    dt = fit(UnitRangeTransform, trace)
    normalizedTrace = StatsBase.transform(dt, trace)
    
    return normalizedTrace
end


# inputs:
# traces_array: a neuron-by-timepoint array of standardized neural data
# k (optional kwarg): overlapping window size, larger k makes signal more smoothed
# λ (optonal kwarg): sparsity parameter for smoothing
# fluc_thresh (optional kwarg): threshold above which a single timepoint is removed and interpolated, set to a defeault of 3 for zscored traces between -1 and 4
# output:
# smooth_traces_array: gstv-smoothed + extreme fluctuation-interpolated traces_array

function smooth_neural_data(traces_array; k=100, λ=0.025, fluc_thresh_std=3)

    smooth_traces_array = zeros(size(traces_array))
    interpolated_traces_array = zeros(size(traces_array))
    
    for n = 1:size(traces_array,1)
        fluc_thresh = fluc_thresh_std * std(traces_array[n,:])
        for i = 2:size(traces_array,2)-1
            if abs(traces_array[n,i]-traces_array[n,i-1])>fluc_thresh
                traces_array[n,i] = (traces_array[n,i+1]+traces_array[n,i-1])/2  # interpolate extreme values
            end
        end
        interpolated_traces_array[n,:] = gstv(traces_array[n,:], 100, 0.01)  # very mild total variation denoising regardless of specified parameters
        smooth_traces_array[n,:] = gstv(traces_array[n,:], k, λ)  # total variation denoising by specified parameters
    end
    
    smooth_traces_array, interpolated_traces_array
end



# inputs:
# data_uid: a string
# smooth_traces_array: output from smooth_neural_data()
# time_encounter: a scalar
# output:
# a plot

function heatmap_neural_data(data_uid, smooth_traces_array, times, time_encounter; clim=(-1.5,3), size=(500,400))
    Plots.heatmap(smooth_traces_array[:,times], clim=clim, size=size,
        xtick=([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600],["0", "2", "4", "6", "8", "10", "12", "14", "16"]), fontsize=11, colorbar_title="z-scored neural activity", colorbar_titlefontsize=10,
    colorbar_tickfontrotation = 90)
    
    # xtick=([0, length(times)/2, length(times)],["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"])
    vline!([time_encounter], linewidth=3, color=:lightgrey, linestyle=:dot, label=nothing)
    xlabel!("time (min)", guidefontsize=11)
    ylabel!("neuron", guidefontsize=11)
end



# inputs:
# traces_array_original: non-standardized traces array, output from reload_data()
# noise_thresh: arbitrary definition of noise threshold
# outputs:
# snr: a vector containing SNR of each neuron

function measure_snr(traces_array, post_encounter_times, noise_thresh)
    num_neurons = size(traces_array,1)
    snr = zeros(num_neurons)
    for n = 1:num_neurons
        # traces_array_original = unitRange_trace(traces_array_original)
        snr[n] = std(traces_array[n,post_encounter_times])
    end
    return snr
end



# inputs:
# data_uid: a string
# nsm: a vector of nsm activity
# times: a series
# color: a string
# time_encounter: a scalar output from reload_data()
# output:
# a plot

function plot_nsm(data_uid, nsm, time_encounter; times=times, color=:black, size=(500,100))
    Plots.plot(nsm', label=nothing, linewidth=3, color=color, xlabel="time (min)", ylabel="NSM activity (AU)", title=data_uid,
        fontsize=14, guidefontsize=14, tickfontsize=14, show=true, foreground_color_border=:lightgrey, size=size, yticks = [],
        xlims=(0,1600), xticks=([0,200,400,600,800,1000,1200,1400,1600], [0,2,4,6,8,10,12,14,16]))
        # xtick=([0, length(times)/2, length(times)],["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"]))
    vline!([time_encounter], label=nothing, linewidth=2, color=:lightgrey)
end



# inputs:
# nsm and beh: vectors of the same length
# y_label: a string
# output:
# a scatter plot

function scatter_beh(nsm, beh, y_label; times = times)
    Plots.scatter(nsm[times], beh[times], label=nothing, xlabel="NSM activity (AU)", ylabel=y_label, 
        markersize=6, markerstrokewidth=0, alpha=0.5, 
        tickfontsize=12, guidefontsize=14, foreground_color_border=:lightgrey)
end



# inputs:
# x and y are 2 vectors of the same length
# output:
# cossim: cosine similarity, 0 means x and y are orthogonal, 1 means x and y are varying in the same direction

function get_cosine_similarity(x,y)
    cossim = dot(x,y) / (norm(x)*norm(y))
    
    cossim
end



# inputs:
# nsm: a vector of smoothed neural trace of NSM, or average of 2 traces of a pair of NSM
# times: a series that indicate post-encounter timepoints
# acor_thresh (optional kwarg): default is 0.2
# output:
# shift_range: idx of times during which nsm has little correlation with itself

function nsm_autocor(nsm, times; acor_thresh=0.2)
    acor = autocor(nsm[times], 1:length(times)-1)
    idx = findall(x->abs.(x) .< acor_thresh, acor)
    exclude = Vector{Float64}([])
    for i = 1:length(idx)
        if idx[i]<100
            push!(exclude, i)
        else 
            for j = 1:20
                if idx[i]+j ∉ idx
                    push!(exclude, i)
                end                
            end
        end
    end
    exclude = unique(exclude)
    shift_range = setdiff(idx, idx[Int.(exclude)])
    shift_range = shift_range[findall(x->x.<round(length(times)/2), shift_range)]
    
    shift_range, acor
end



# inputs:
# nsm: a vector of smoothed neural trace of NSM, or average of 2 traces of a pair of NSM
# times: a series that indicate post-encounter timepoints
# outputs:
# a plot

function plot_nsm_autocor(nsm, times)
    shift_range, acor = nsm_autocor(nsm, times)
    Plots.bar(acor, ylims=(-1,1), xlims=(0, length(times)), guidefontsize=14, tickfontsize=14, fontsize=14,
        xtick=([0, length(times)/2, length(times)],["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"]), 
        xlabel="time (min)", title="NSM autocorrelation", label=nothing, foreground_color_border=:lightgrey)
    vspan!([shift_range[1], shift_range[end]], color=:black, label=nothing, alpha=0.1)
end



function make_kernel(a,b,c,d,e)
    rise_phase = a:b:c
    decay_phase = c:-d:e
    kernel = vcat(rise_phase, decay_phase[2:end])
    
    return kernel, length(rise_phase), length(decay_phase)
end



function make_differentiator_kernel(a,b,c)
    k = collect(a:b:c)
    len1 = Int.((length(k)+1)/2)
    len2 = Int.((length(k)+1)/2)
    
    # kernel = zeros(length(k))
    # kernel[1:len1-1] .= a
    # kernel[len1+1:length(k)] .= c
    
    return k, len1, len2
end



function plot_kernel(kernel, len1, len2)
    Plots.plot(xticks=([-16.5,0,16.5], ["-10","0","10"]), xlabel="time (sec)", ylabel="weight", tickfontsize=14, guidefontsize=14, grid=:off, xlims=(-17,17))
    plot!(collect(-len1+1:len2-1), kernel, c="black", linewidth=3, label=nothing)
    hline!([0], linestyle=:dot, c="grey", label=nothing)
    vline!([0], linestyle=:dot, c="grey", label=nothing)
end



function unitRange_trace(trace)
    dt = fit(UnitRangeTransform, trace)
    normalizedTrace = StatsBase.transform(dt, trace)
    
    return normalizedTrace
end



function conv_nsm(nsm, post_encounter_times, kernel, len1, len2)
    start_t = post_encounter_times[1] + len1
    stop_t = post_encounter_times[end] - len2
    nsm_convolved = zeros(stop_t - start_t + 1)
    k = 1
    for t = collect(start_t:stop_t)
        nsm_toi = nsm[t-len1+1 : t+len2-1]
        nsm_convolved[k] = sum(nsm_toi .* kernel)
        k = k+1
    end
    
    return nsm_convolved, start_t, stop_t
end



function test_nsm_kernels_differentiator(nsm, fit_target, post_encounter_times, piezo_rate)
    if piezo_rate == 1.65
        speeds = [8,4,2,1]
        metric = zeros(3,4)
        for (i,s) = enumerate(speeds)
            kernel, len1, len2 = make_differentiator_kernel(-8,s,8)

            nsm_convolved, start_t, stop_t = conv_nsm(nsm, post_encounter_times, kernel, len1, len2)
            # corr = corspearman(nsm_derivative[len1+1:length(nsm_derivative)-len2], nsm_convolved)
            gof = corspearman(fit_target[start_t:stop_t], nsm_convolved)
            metric[:,i] = [s,0,gof]
        end
    elseif piezo_rate == 1.35
        speeds = [8,4,2,1]./1.65.*1.35
        metric = zeros(3,4)
        for (i,s) = enumerate(speeds)
            kernel, len1, len2 = make_differentiator_kernel(-8/1.65*1.35,s,8/1.65*1.35)

            nsm_convolved, start_t, stop_t = conv_nsm(nsm, post_encounter_times, kernel, len1, len2)
            # corr = corspearman(nsm_derivative[len1+1:length(nsm_derivative)-len2], nsm_convolved)
            gof = corspearman(fit_target[start_t:stop_t], nsm_convolved)
            metric[:,i] = [s,0,gof]
        end
    end
    
    return metric
end



function test_nsm_kernels_centered(nsm, fit_target, post_encounter_times, piezo_rate)
    if piezo_rate == 1.65
        speeds_1 = [16,8,4,2,1]
        speeds_2 = [16,8,4,2,1]
        metric = zeros(3,18)
        i = 1
        for s1 = speeds_1
            for s2 = speeds_2
                if s1<16 && s2<2
                    continue
                elseif s1<2 && s2<16
                    continue
                end
                kernel, len1, len2 = make_kernel(0,s1,16,s2,0)

                nsm_convolved, start_t, stop_t = conv_nsm(nsm, post_encounter_times, kernel, len1, len2)
                gof = corspearman(nsm_convolved, fit_target[start_t:stop_t])
                metric[:,i] = [s1,s2,gof]
                i = i+1
            end
        end
    elseif piezo_rate == 1.35
        speeds_1 = [16,8,4,2,1]./1.65.*1.35
        speeds_2 = [16,8,4,2,1]./1.65.*1.35
        metric = zeros(3,18)
        i = 1
        for s1 = speeds_1
            for s2 = speeds_2
                if s1<16 /1.65*1.35 && s2<2 /1.65*1.35
                    continue
                elseif s1<2 /1.65*1.35 && s2<16/1.65*1.35
                    continue
                end                
                kernel, len1, len2 = make_kernel(0,s1,16/1.65*1.35,s2,0)

                nsm_convolved, start_t, stop_t = conv_nsm(nsm, post_encounter_times, kernel, len1, len2)
                gof = corspearman(nsm_convolved, fit_target[start_t:stop_t])
                metric[:,i] = [s1,s2,gof]
                i = i+1
            end
        end        
    end
    
    return metric
end



function test_ash_kernels_centered(ash, fit_target, post_encounter_times, piezo_rate)
    if piezo_rate == 1.65
        speeds_1 = [32,16,8,4,2,1]
        speeds_2 = [32,16,8,4,2,1]
        metric = zeros(3,27)
        i = 1
        for s1 = speeds_1
            for s2 = speeds_2
                if s1<32 && s2<2
                    continue
                elseif s1<2 && s2<32
                    continue
                end
                kernel, len1, len2 = make_kernel(0,s1,32,s2,0)

                ash_convolved, start_t, stop_t = conv_nsm(ash, post_encounter_times, kernel, len1, len2)
                gof = corspearman(ash_convolved, fit_target[start_t:stop_t])
                metric[:,i] = [s1,s2,gof]
                i = i+1
            end
        end       
    end
    
    return metric
end



function find_winner_param(metric)
    max_gof = maximum(abs.(metric[end, :]))
    winner_idx = findall(x->abs.(x)==max_gof, metric[end,:]) # if I ever want to select the top 1% of kernels, change this line 
    winner_param = metric[1:end-1, winner_idx]
    best_gof = metric[end, winner_idx]

    return winner_param, best_gof
end



function swap_test_nsm_kernels_centered(nsm, smooth_traces_array, noisy_idx, post_encounter_times, NSM_like_garbage_library; fdr=0.1, tail=2, piezo_rate=1.65)
    # segment NSM_like_garbage of the same length as nsm
    NSM_like_garbage_borrowed = zeros(length(NSM_like_garbage_library), post_encounter_times[end])
    j = 1
    for i = collect(keys(NSM_like_garbage_library))
        t_start = 101
        t_end = 101+length(post_encounter_times)-1
        if abs.(corspearman(NSM_like_garbage_library[i][t_start:t_end], nsm[post_encounter_times])) < 0.5
            NSM_like_garbage_borrowed[j,:] = vcat(zeros(post_encounter_times[1]-1), vec(NSM_like_garbage_library[i][t_start:t_end])) # pad the pre-encounter times with zeros
            j = j+1
        end
    end
    n_swaps = j-1
    print(n_swaps, "\n")
    
    all_neuron_idx = 1:size(smooth_traces_array, 1)
    good_neuron_idx = setdiff(all_neuron_idx, noisy_idx)
    num_good_neurons = length(good_neuron_idx)
    
    winner_param_all_neurons = zeros(3, num_good_neurons)
    gof_winner_all_shifts = zeros(n_swaps)
    p_val_uncorr = ones(num_good_neurons)
    gof_distribution = Dict()
    
    for n = 1:num_good_neurons
        neuron_idx = good_neuron_idx[n]
        fit_target = smooth_traces_array[neuron_idx, :]
        
        # find best fitting kernel of this neuron to unshifted nsm
        metric_no_shift = test_nsm_kernels_centered(nsm, fit_target, post_encounter_times, piezo_rate)
        winner_param, best_gof = find_winner_param(metric_no_shift)
        gof_winner_no_shift = best_gof[1]
        winner_param_all_neurons[1:2,n] = winner_param[1:2]
        winner_param_all_neurons[end,n] = neuron_idx
        
        # find best fitting kernel of this neuron to each shifted nsm
        gof_winner_all_shifts = zeros(n_swaps)
        for sh = 1:n_swaps
            metric_this_shift = test_nsm_kernels_centered(NSM_like_garbage_borrowed[sh,:], fit_target, post_encounter_times, piezo_rate)
            winner_param, best_gof = find_winner_param(metric_this_shift)
            gof_winner_all_shifts[sh] = best_gof[1]
        end
        
        # compute p-value for this neuron to see if the best fitting kernel for unshifted nsm does better than that for all shifted nsms
        vect = vcat(gof_winner_all_shifts, gof_winner_no_shift)
        s = sortperm(vect)
        tsamples = n_swaps+1

        if gof_winner_no_shift > 0
            rank = maximum(findall(x->x==vect[tsamples], vect[s]))
            p = (tsamples-rank)/tsamples # compute p from rank (rank 1 has the smallest p), if 1% of samples are ranked higher than orig, p=0.01
        else
            rank = minimum(findall(x->x==vect[tsamples], vect[s]))
            p = (rank-1)/tsamples # if rank is very close to 1, the correlational measure is very negative and p is very small
        end
        
        p_val_uncorr[n] = p
        gof_distribution[neuron_idx] = vect
        print("$(neuron_idx)")
    end
    
    p_val_corr = MultipleTesting.adjust(p_val_uncorr, BenjaminiHochberg())
    sig_idx = findall(x->x.<fdr, p_val_corr) ######## need more work
    sig_neuron_idx = winner_param_all_neurons[end,sig_idx]

    return winner_param_all_neurons, p_val_uncorr, p_val_corr, sig_neuron_idx, gof_distribution
end



function swap_test_nsm_kernels_differentiator(nsm, smooth_traces_array, noisy_idx, post_encounter_times, NSM_like_garbage_library; fdr=0.1, tail=2, piezo_rate=1.65)
    # segment NSM_like_garbage of the same length as nsm
    NSM_like_garbage_borrowed = zeros(length(NSM_like_garbage_library), post_encounter_times[end])
    j = 1
    for i = collect(keys(NSM_like_garbage_library))
        t_start = 101
        t_end = 101+length(post_encounter_times)-1
        if abs.(corspearman(NSM_like_garbage_library[i][t_start:t_end], nsm[post_encounter_times])) < 0.25
            NSM_like_garbage_borrowed[j,:] = vcat(zeros(post_encounter_times[1]-1), vec(NSM_like_garbage_library[i][t_start:t_end])) # pad the pre-encounter times with zeros
            j = j+1
        end
    end
    print(j,"\n")
    n_swaps = j-1
    
    all_neuron_idx = 1:size(smooth_traces_array, 1)
    good_neuron_idx = setdiff(all_neuron_idx, noisy_idx)
    num_good_neurons = length(good_neuron_idx)
    
    winner_param_all_neurons = zeros(3, num_good_neurons)
    gof_winner_all_shifts = zeros(n_swaps)
    p_val_uncorr = ones(num_good_neurons)
    gof_distribution = Dict()
    
    for n = 1:num_good_neurons
        neuron_idx = good_neuron_idx[n]
        fit_target = smooth_traces_array[neuron_idx, :]
        
        # find best fitting kernel of this neuron to unshifted nsm
        metric_no_shift = test_nsm_kernels_differentiator(nsm, fit_target, post_encounter_times, piezo_rate)
        winner_param, best_gof = find_winner_param(metric_no_shift)
        gof_winner_no_shift = best_gof[1]
        winner_param_all_neurons[1:2,n] = winner_param[1:2]
        winner_param_all_neurons[end,n] = neuron_idx
        
        # find best fitting kernel of this neuron to each shifted nsm
        gof_winner_all_shifts = zeros(n_swaps)
        for sh = 1:n_swaps
            metric_this_shift = test_nsm_kernels_differentiator(NSM_like_garbage_borrowed[sh,:], fit_target, post_encounter_times, piezo_rate)
            winner_param, best_gof = find_winner_param(metric_this_shift)
            gof_winner_all_shifts[sh] = best_gof[1]
        end
        
        # compute p-value for this neuron to see if the best fitting kernel for unshifted nsm does better than that for all shifted nsms
        vect = vcat(gof_winner_all_shifts, gof_winner_no_shift)
        s = sortperm(vect)
        tsamples = n_swaps+1

        if gof_winner_no_shift > 0
            rank = maximum(findall(x->x==vect[tsamples], vect[s]))
            p = (tsamples-rank)/tsamples # compute p from rank (rank 1 has the smallest p), if 1% of samples are ranked higher than orig, p=0.01
        else
            rank = minimum(findall(x->x==vect[tsamples], vect[s]))
            p = (rank-1)/tsamples # if rank is very close to 1, the correlational measure is very negative and p is very small
        end
        
        p_val_uncorr[n] = p
        gof_distribution[neuron_idx] = vect
        print("$(neuron_idx)")
    end
    
    p_val_corr = MultipleTesting.adjust(p_val_uncorr, BenjaminiHochberg())
    sig_idx = findall(x->x.<fdr, p_val_corr) ######## need more work
    sig_neuron_idx = winner_param_all_neurons[end,sig_idx]

    return winner_param_all_neurons, p_val_uncorr, p_val_corr, sig_neuron_idx, gof_distribution
end



# inputs:
# nsm: a vector of smoothed neural trace of NSM, or average of 2 traces of a pair of NSM
# beh: a vector of behavioral variable, or a vector of neural trace, eg smooth_traces_array[neuron_idx,:]; need to be of the same length as nsm
# times: a series indicating time window of interest
# shift_range: a series indicating the range of shifts
# method: a string, one of the following -- "pearson correction", "spearman correlation", "mutual information", "cosine similarity"
# y_label: a string, can be []
# plot (optional kwarg): if true, plot a distribution of shifts with 95% confidence interval in blue shade
# tail (optional kwarg): default is 2-tailed test, but may want to do one-tailed for nsm=kernel-convolved nsm to pick up negative kernels OR for method="mutual information"
# threshs (optional kwarg): default is two tailed 95% CI, ie [2.5, 97.5]
# samples (optional kwarg): default is random sampling of 5000 shift times from a uniform distribution of time lags where NSM has low autocorrelation
# outputs:
# a plot (optional)
# p: probability that a two-tailed relationship between nsm and beh is significant
# orig: the correlational measure between nsm and beh

function shift_test_nsm(nsm, beh, times, shift_range, method; plot=false, tail=2, threshs=[2.5, 97.5], samples=5000, y_label=[])   
    shifts = Int.(round.(rand(Uniform(shift_range[1],shift_range[end]), samples), digits=0)) # sample from shift_range
    shifted = []
    # first_t = times[1]
    # last_t = times[end]
    
    if method=="pearson correlation"
        orig = cor(nsm[times], beh[times])
        for i = shifts
            nsm_warp = circshift(nsm[times], i)
            temp = cor(nsm_warp, beh[times])
            push!(shifted, temp)
        end
        
    elseif method=="spearman correlation"
        orig = corspearman(nsm[times], beh[times])
        for i = shifts
            nsm_warp = circshift(nsm[times], i)
            temp = corspearman(nsm_warp, beh[times])
            push!(shifted, temp)
        end
        
    elseif method=="mutual information"
        orig = get_mutual_information(nsm[times], beh[times])/get_entropy(nsm[times], beh[times])
        for i = Int64.(shifts)
            nsm_warp = vcat(nsm[first_t+i:last_t], nsm[first_t:first_t+i-1])
            temp = get_mutual_information(nsm_warp, beh[times])/get_entropy(nsm_warp, beh[times])
            push!(back_shifted, temp)
        end
        for i = Int64.(shifts)
            nsm_warp = vcat(nsm[last_t-i+1:last_t], nsm[first_t:last_t-i])
            temp = get_mutual_information(nsm_warp, beh[times])/get_entropy(nsm_warp, beh[times])
            push!(fwd_shifted, temp)
        end
        
    elseif method=="cosine similarity"
        orig = get_cosine_similarity(nsm[times], beh[times])
        for i = Int64.(shifts)
            nsm_warp = vcat(nsm[first_t+i:last_t], nsm[first_t:first_t+i-1])
            temp = get_cosine_similarity(nsm_warp, beh[times])
            push!(back_shifted, temp)
        end
        for i = Int64.(shifts)
            nsm_warp = vcat(nsm[last_t-i+1:last_t], nsm[first_t:last_t-i])
            temp = get_cosine_similarity(nsm_warp, beh[times])
            push!(fwd_shifted, temp)
        end
    end
    
    vect = push!(shifted, orig)
    s = sortperm(vect)
    tsamples = samples+1
    rank = minimum(findall(x->x==vect[tsamples], vect[s]))
    thresh_1 = StatsBase.percentile(shifted, threshs[1])
    thresh_2 = StatsBase.percentile(shifted, threshs[2])
    
    if orig > 0
        p = (tsamples-rank)/tsamples # compute p-val from rank (rank 1 has the smallest value), if 1% of samples are ranked higher than orig, then p=0.01
    elseif orig < 0 && tail == 2
        p = (rank-1)/tsamples # if rank is very close to 1, the correlational measure is very negative and p is very small
    else
        p = 1 # orig is 0, there is no relationship whatsoever
    end
    
    if plot
        figure(figsize=(2,5))
        plt.plot([0.8, 1.2], [orig, orig], color="red", linewidth=3)
        for i = shifted
            plt.plot([1.6, 2.0], [i,i], color="black", alpha=0.01, linewidth=1)
        end

        plt.xticks([1,1.8], ["NSM", "shifted NSM"], rotation=90, fontsize=14)
        plt.axhspan(thresh_1, thresh_2, color=:lightblue)
        ylabel("$method $y_label", fontsize=14)
        title("p = $(round(p; digits=3))", fontsize=14)
    end
    
    return p, orig
end



function swap_test_nsm(nsm, beh, times, NSM_like_garbage_library, method; plot_=false, tail=2, threshs=[2.5, 97.5], y_label="")  
    # segment NSM_like_garbage of the same length as nsm
    NSM_like_garbage_borrowed = zeros(length(NSM_like_garbage_library), times[end])
    j = 1
    for i = collect(keys(NSM_like_garbage_library))
        t_start = 101
        t_end = 101+length(times)-1
        if abs.(corspearman(NSM_like_garbage_library[i][t_start:t_end], nsm[times])) < 0.5
            NSM_like_garbage_borrowed[j,:] = vcat(zeros(times[1]-1), vec(NSM_like_garbage_library[i][t_start:t_end])) # pad the pre-encounter times with zeros
            j = j+1
        end
    end
    n_swaps = j-1
    print(n_swaps, "\n")
    
    nsm_segment = nsm[times]
    beh_segment = beh[times]
    swaps = zeros(n_swaps)
    if method=="pearson correlation"
        orig = cor(nsm_segment, beh_segment)
        for i = 1:n_swaps
            garb = NSM_like_garbage_borrowed[i,times]
            swaps[i] = cor(garb, beh_segment)
        end
        
    elseif method=="spearman correlation"
        orig = corspearman(nsm_segment, beh_segment)
        for i = 1:n_swaps
            garb = NSM_like_garbage_borrowed[i,times]
            swaps[i] = corspearman(garb, beh_segment)
        end
        
    elseif method=="mutual information"
        orig = get_mutual_information(nsm_segment, beh_segment)/get_entropy(nsm_segment, beh_segment)
        for i = 1:n_swaps
            garb = NSM_like_garbage_borrowed[i,times]
            swaps[i] = get_mutual_information(garb, beh_segment)/get_entropy(garb, beh_segment)
        end
        
    elseif method=="cosine similarity"
        orig = get_cosine_similarity(nsm_segment, beh_segment)
        for i = 1:n_swaps
            garb = NSM_like_garbage_borrowed[i,times]
            swaps[i] = get_cosine_similarity(garb, beh_segment)
        end
    end
    vect = push!(swaps, orig)
    s = sortperm(vect)
    tsamples = n_swaps+1
    rank = minimum(findall(x->x==vect[tsamples], vect[s]))
    thresh_1 = StatsBase.percentile(swaps, threshs[1])
    thresh_2 = StatsBase.percentile(swaps, threshs[2])
    
    if orig > 0
        p = (tsamples-rank)/tsamples # compute p-val from rank (rank 1 has the smallest value), if 1% of samples are ranked higher than orig, then p=0.01
    elseif orig < 0 && tail == 2
        p = (rank-1)/tsamples # if rank is very close to 1, the correlational measure is very negative and p is very small
    else
        p = 1 # orig is 0, there is no relationship whatsoever
    end
    
    if plot_
        figure(figsize=(2,5))
        plt.plot([0.8, 1.2], [orig, orig], color="red", linewidth=3)
        for i = swaps
            plt.plot([1.6, 2.0], [i,i], color="black", alpha=0.01, linewidth=1)
        end

        plt.xticks([1,1.8], ["NSM", "NSM-like garbage"], rotation=90, fontsize=14)
        plt.axhspan(thresh_1, thresh_2, color=:lightblue)
        ylabel("$method $y_label", fontsize=14)
        title("p = $(round(p; digits=3))", fontsize=14)
    end    
    return p, orig, swaps
end



# inputs:
# same as inputs of shift_test_nsm(), except that beh can be an array instead of a vector
# fdr (optional kwarg): false detection rate, default 0.05
# output:
# p_val_uncorr: a vector of uncorrected p-values for the shifting test on all neurons
# p_val_corr: a vector of BH corrected p-values for the shifting test on all neurons
# sig_neuron_idx: a vector of idx of neurons that pass the shifting test; smooth_traces_array[sig_neuron_idx,:] returns the neural traces of all nsm-modulated neurons
# orig_all: a vector of correlational measure with nsm over timepoints of interest

function find_sig_neurons(nsm, smooth_traces_array, times, shift_range, method, tail, threshs; fdr=0.05)
    num_neurons = size(smooth_traces_array,1)
    p_val_uncorr = ones(num_neurons)
    orig_all = ones(num_neurons)

    for n = 1:num_neurons
        p, orig = shift_test_nsm(nsm, smooth_traces_array[n,:], times, shift_range, method; plot=false, tail=tail, threshs=threshs, samples=5000)
        p_val_uncorr[n] = p
        orig_all[n] = orig
    end 

    p_val_corr = MultipleTesting.adjust(p_val_uncorr, BenjaminiHochberg())
    sig_neuron_idx = findall(x->x.<=fdr, p_val_corr)
    
    p_val_uncorr, p_val_corr, sig_neuron_idx, orig_all
end



function plot_sig_neurons(p_val_uncorr, p_val_corr, sig_neuron_idx, orig_all; method=method, fdr=0.1)
    message = "$(length(sig_neuron_idx)) out of $(length(p_val_corr)) neurons are NSM-related \n by $(method)"
    
    p1 = Plots.scatter(p_val_uncorr, color=:white, markerstrokecolor=:green, label="uncorrected", markersize = 4, markerstrokewidth=1)
    scatter!(p_val_corr, color=:orange, label="corrected for multiple testing", legend=:topright,
        markersize = 2, markerstrokewidth=0, ylims=(-0.05,1.05), foreground_color_border=:lightgrey,
        ylabel = "p-value", xlabel = "neuron", fontsize=14)
    hline!([fdr], color=:black, linestyle=:dot, label=nothing)
    title!(message)
    
    p2 = Plots.scatter(orig_all, p_val_corr, label=nothing, color=:blue, markersize = 3, markerstrokewidth=0, 
        ylims=(-0.05,1.05), foreground_color_border=:lightgrey, 
        ylabel = "corrected p-value", xlabel = method, fontsize=14)
    hline!([fdr], label=nothing, color=:black, linestyle=:dot)
    
    Plots.plot(p1, p2, layout=(2,1), show=true, size=(600,600))
end



# inputs:
# data_uid: a string
# nsm: a vector of NSM activity
# smooth_traces_array: imported from reload_data()
# time_encounter: imported from reload_data()
# times (optional_kwarg): a series that cover all useful timepoints, eg 1:799
# cs (optional kwarg): default is empty
# output:
# a plot

function plot_3d_pcspace(data_uid, nsm, smooth_traces_array, time_encounter; times = times, time_exc=66, cs = [])
    PCproj_all = neural_pca(nsm, smooth_traces_array; times=times, plot_var_exp=false, plot_pcs=false)
    PCproj_all = PCproj_all[1]
    
    if isempty(cs)
        fig = plt.figure(figsize=(15,4))
        ax = fig.add_subplot(projection="3d")
        f = ax.scatter(PCproj_all[1,:], PCproj_all[2,:], PCproj_all[3,:], c=times, s=4, cmap="plasma")
        ax.scatter(PCproj_all[1,time_encounter], PCproj_all[2,time_encounter], PCproj_all[3,time_encounter], marker="d", c="black", s=20)
        xlabel("PC 1")
        ylabel("PC 2")
        zlabel("PC 3")
        title(data_uid, fontsize=14)

        cbar = plt.colorbar(f, label="time (min)")
        cbar.set_ticks([times[1], mean(times), times[end]])
        cbar.set_ticklabels(["$(Int.(round((times[1]-1)/100)))", "$(Int.(round(mean(times)/100)))", "$(Int.(round(times[end]/100)))"])
        
    elseif cs=="pre_post"
        C = Vector{String}(undef, length(times))
        for t = 1:length(times)
            if t < time_encounter-time_exc
                C[t] = "red"
            elseif t > time_encounter-time_exc && t < time_encounter
                C[t] = "orange"
            else
                C[t] = "blue"
            end
        end
        Plots.scatter(PCproj_all[1,:], PCproj_all[2,:], PCproj_all[3,:], markersize=3, color=C, markerstrokewidth=0,
            xlabel="PC 1", ylabel="PC 2", zlabel="PC 3", fontsize=14, alpha=0.5, label=nothing, title=data_uid)
        
    elseif cs == "manual"
        C = Vector{String}(undef, length(times))
        for t = 1:length(times)
            if t < time_encounter
                C[t] = "orange"
            elseif t > 1300
                C[t] = "purple"
            else
                C[t] = "green"
            end
        end
        Plots.scatter(PCproj_all[1,:], PCproj_all[2,:], PCproj_all[3,:], markersize=3, color=C, markerstrokewidth=0,
            xlabel="PC 1", ylabel="PC 2", zlabel="PC 3", fontsize=14, alpha=0.5, label=nothing, title=data_uid)    
        
    elseif cs == "zoomin"
        C = Vector{String}(undef, length(times))
        for t = 1:length(times)
            if t > time_encounter
                C[t] = "purple"
            else
                C[t] = "green"
            end
        end
        Plots.scatter(PCproj_all[1,:], PCproj_all[2,:], PCproj_all[3,:], markersize=3, color=C, markerstrokewidth=0,
            xlabel="PC 1", ylabel="PC 2", zlabel="PC 3", fontsize=14, alpha=0.5, label=nothing, title=data_uid)   
    end
end



# inputs:
# nsm: a vector of NSM activity 
# smooth_traces_array: an array 
# times (optional kwarg): a series of timepoints of interest, default 1:1600
# plot_var_exp (optional kwarg): if true, the variance explained will be plotted
# plot_pc123 (optional kwarg): if true, the 3D principal component space will be plotted
# plot_pcs (optional kwarg): if true, pca traces and loadings will be plotted
# outputs:
# PCproj: an array with each column being a PC
# PCvars: a vector of % variance explained by each PC

function neural_pca(nsm, smooth_traces_array; times=times, plot_var_exp=false, plot_pcs=true, sign_vector = [])
    data_centered = zeros(size(smooth_traces_array[:, times]))
    
    for n = 1:size(smooth_traces_array,1)
        data_centered[n,:] = zscore_trace(smooth_traces_array[n, times])
    end
    
    M, X, Yt = multivar_fit(data_centered, PCA)
    PC = Yt
    var_exp = M.prinvars / M.tprinvar
    loading = M.proj
    
    if plot_var_exp
        figure(figsize=(6,4))
        subplot(2,1,1)
        plt.plot(var_exp, c="red", linewidth=2)
        plt.axhline(0.05, c="grey", linewidth=1, linestyle="dotted")
        xlim([-0.5,80])
        ylim([-0.05, 0.45])
        ylabel("% var exp", fontsize=12)
        
        subplot(2,1,2)
        plt.plot(cumsum(var_exp), c="purple", linewidth=2)
        plt.axhline(0.8, c="grey", linewidth=1, linestyle="dotted")
        xlim([-0.5,80])
        ylim([0,1.05])
        ylabel("cumulative var exp", fontsize=12)
        xlabel("principal component", fontsize=12)
    end
     
    if plot_pcs
        figure(figsize=(8,5))
        for i = 1:10
            if isempty(sign_vector)
                subplot(10,1,i)
                plt.plot(zscore_trace(PC[i,:]), color="blue", linewidth=2)
                plt.plot(zscore_trace(nsm[times]), color="black", linewidth=1)
                xlim([0,length(times)])
                axis("off")
                # if i==1
                #     title("principal component 1-8")
                # end

                # subplot(8,2,i*2)
                # plt.plot(sort(loading[:,i]), color="green", linewidth=2)
                # plt.axhline(0, color="grey", linewidth=1, linestyle="dotted")
                # axis("off")
                # if i==1
                #     title("loading on PC")
                # end
            else
                subplot(10,1,i)
                plt.plot(zscore_trace(PC[i,:] .* sign_vector[i]), color="blue", linewidth=2)
                plt.plot(zscore_trace(nsm[times]), color="black", linewidth=1)
                xlim([0,length(times)])
                axis("off")
                # if i==1
                #     title("principal component 1-8")
                # end

                # subplot(8,2,i*2)
                # plt.plot(sort(loading[:,i] .* sign_vector[i]), color="green", linewidth=2)
                # plt.axhline(0, color="grey", linewidth=1, linestyle="dotted")
                # axis("off")
                # if i==1
                #     title("loading on PC")
                # end                
            end
        end
    end
    
    return PC, var_exp, loading
end



# input:
# trace: a vector of smoothed neural trace or PC over post-encounter timepoints
# output:
# zscoredtrace: a vector with mean=0 and variance=1

function zscore_trace(trace)
    dt = fit(ZScoreTransform, trace)
    zscoredtrace = StatsBase.transform(dt, trace)
    
    return zscoredtrace
end



# inputs:
# zscored_nsm: a vector of zscoredtrace of a single nsm neuron, or a vector of average zscoredtraces of a pair of nsm neurons
# fit_target: a vector of PC or neural trace
# kernel_range: a series indicating timepoints of interest, eg all post-encounter timepoints
# max_rise (optional kwarg): high rise rate possible, default is 1
# max_fall (optional kwarg): high decay rate possible, default is 1
# max_time_window (optional kwarg): longest time window possible, default is 25 timepoints
# plot_kernels, plot_convolved_nsms (optional kwarg): cannot be true at the same time
# outputs:
# a kernel plot (optional)
# a convolved nsm plot (optional)
# metric: an array with 5 rows, the first 4 rows indicating the rise rate, decay rate, time window and sign of the kernel, 
# and the last row recording the goodness of it measure with fit_target; number of columns represents the number of observations

function test_nsm_kernels(zscored_nsm, fit_target, kernel_range; max_rise=1, max_fall=1, max_time_window=25, plot_kernels=false, plot_convolved_nsms=false)
    metric = [0,0,0,0,0]
    Ts = 1:max_time_window

    for r = 0.1:0.1:max_rise
        for f = 0.1:0.1:max_fall
            for i = Ts
                for s = [-1,1]
                    trend = [] # the ups and downs of a kernel
                    for t = 1:i
                        trend = vcat(trend, r*t)
                    end
                    peak = trend[end]
                    for t = 1:i
                        trend = vcat(trend, peak-f*t)
                    end
                    trend = trend .* s

                    if plot_kernels
                        plt.plot(trend, foreground_color_border=:lightgrey)
                        title("kernels")
                    end

                    convolved_nsm = conv(Float64.(trend), zscored_nsm[kernel_range[1]-i:kernel_range[end]]) 
                    convolved_nsm = zscore_trace(convolved_nsm)

                    if plot_convolved_nsms
                        plt.plot(convolved_nsm)
                        title("kernel-convolved NSMs", foreground_color_border=:lightgrey)
                    end

                    x = convolved_nsm[i+1:(length(convolved_nsm)-2*i+1)]
                    y = fit_target

                    if length(x)==length(y)
                        gof = get_cosine_similarity(x,y)
                        vec = [r,f,i,s,gof]
                        metric = hcat(metric, vec)
                    else
                        print("convolved_nsm and fit_target are not of the same length!")
                    end
                end
            end
        end
    end
    
    metric = metric[:, 2:end]
    return metric
end
    
    
    
# inputs:
# metric: output from test_nsm_kernels()
# plot (optional kwarg): default is false; if true, will plot a lattice with color scale representing goodness-of-fit by cosine similarity (extremely red or blue are both good)
# outputs:
# a scatter plot: redder means better fit
# winner_param: 4-element kernel parameters that fits the fit_target best

function find_winner_kernel(metric; plot=false)
    if plot
        cs = ColorScheme([colorant"blue", colorant"red"]);
        colors = get.(Ref(cs), metric[5,:])./ maximum(metric[5,:])
        Plots.scatter(metric[3,:], metric[1,:]./metric[2,:], metric[4,:], label=nothing, markersize=3, markerstrokewidth=0, fontsize=14,
            color=colors, xlabel="time window", ylabel="rise:decay ratio", zlabel="sign", title="kernel fit", foreground_color_border=:lightgrey)
    end
    
    winner_idx = findall(x->x==maximum(metric[5,:]), metric[5,:])
    winner_param = metric[:,winner_idx][:,1] # multiple kernels can perform equally well (because rise:decay is a ratio), in which case I take the first of those equally good kernels
        
    return winner_param
end
    
    

# inputs:
# zscored_nsm: a vector of zscoredtrace of a single nsm neuron, or a vector of average zscoredtraces of a pair of nsm neurons
# fit_target: a vector of PC or neural trace, need to be of the same length as zscored_nsm[kernel_range]
# kernel_range: a series indicating timepoints of interest, eg all post-encounter timepoints
# metric: output from test_nsm_kernel()
# outputs:
# winner_trend: best-fitting kernel
# winner_convolved_nsm: best-fitting nsm convolution

function find_winner_kernel(zscored_nsm, fit_target, kernel_range, metric)
    winner_idx = findall(x->x==maximum(metric[5,:]), metric[5,:])
    winner_param = metric[:,winner_idx][:,1] # multiple kernels can perform equally well, in which case I take the first of those equally good kernels
    r = winner_param[1]
    f = winner_param[2]
    i = Int.(winner_param[3])
    s = winner_param[4]
    
    trend = [] # the ups and downs of a kernel
    for t = 1:i
        trend = vcat(trend, r*t)
    end
    peak = trend[end]
    for t = 1:i
        trend = vcat(trend, peak-f*t)
    end
    winner_trend = trend .* s
    
    convolved_nsm = conv(Float64.(winner_trend), zscored_nsm[kernel_range[1]-i:kernel_range[end]]) 
    convolved_nsm = zscore_trace(convolved_nsm)
    winner_convolved_nsm = convolved_nsm[i+1:(length(convolved_nsm)-2*i+1)]
    
    return winner_trend, winner_convolved_nsm, r, f, i, s
end



function compute_kernel_dist(metric)
    a = metric[1,:]
    differentiator = Vector{String}(undef, length(a))
    
    for i = 1:length(a)
        if a[i]==0
            a[i] = 1
            differentiator[i] = "green"
        else
            a[i] = 16
            differentiator[i] = "purple"
        end
    end
    x = log2.(a) # starting pt of all kernels
    y = log2.(metric[2,:]) # rise rate of all kernels
    z = log2.(metric[4,:]) # decay rate of all kernels
    
    winner_param, best_gof = find_winner_param(metric)
    
    if winner_param[1]==0
        b = 1
    else
        b = 16
    end
    x0 = log2(b)
    y0 = log2(winner_param[2])
    z0 = log2(winner_param[4])
    
    kernel_dist = sqrt.((x.-x0).^2 + (y.-y0).^2 + (z.-z0).^2)
    kernel_gof = metric[6,:]
    
    return kernel_dist, kernel_gof, differentiator
end



function plot_kernel_dist(kernel_dist, kernel_gof, differentiator)
    Plots.scatter(kernel_dist, kernel_gof, markerstrokewidth=0, color=differentiator, markersize=5, alpha=0.5, label=nothing, 
        xlabel="distance from winner NSM kernel", ylabel="spearman correlation\nwith target neuron", legend=:none, guidefontsize=15, tickfontsize=12,
        xlims=(-0.05,8.05), ylims=(-1,1), foreground_color_border=:lightgrey)
    hline!([0], color=:grey, linewidth=4, linestyle=:dot)
end



# inputs:
# array: vcat(x',y') or vcat(x',y',z'), where x,y,z are vectors of the same length; every column is one significant neuron
# num_cluster (optional kwarg): default is 0; if know which one is best, enter the best scalar here
# plot (optional kwarg): if true, make a colored scatter plot of the clusters
# outputs:
# a plot (optional)
# R: a (dictionary of) KmeansResult object; assignments(r) gives data point cluster assignment; counts(r) gives cluster size; r.centers gives cluster centers

function cluster_by_param(array; plot=true, num_cluster=0)
    include("copy_kmeans.jl")
    
    if size(array,1)>3
        print("dimension of array is too high; plz reduce it to 2 or 3 before clustering")
    else
        if num_cluster==0
            R = Dict()
            cost = zeros(8)
            figure(figsize=(8,8))

            for num_cluster = 1:8
                r = real_kmeans(array, num_cluster)                
                R["$(num_cluster)"] = r
                cost[num_cluster] = r.totalcost
                subplot(3,3,num_cluster)
                plt.scatter(array[1,:], array[2,:], c=r.assignments, s=2)
                axis("off")
                title("ncluster = $num_cluster")
            end
            
            subplot(3,3,9)
            plt.plot(1:8, cost, c="black")
            xlabel("ncluster")
            ylabel("error")
            xlim([1,8])
        else
            R = real_kmeans(array, num_cluster)
            
            p1 = Plots.scatter(array[1,:], array[2,:], marker_z=R.assignments, markerstrokewidth=0, color=:rainbow, markersize=2,
                label=nothing, xlabel="time window", ylabel="rise:decay ratio", legend=:none, fontize=13, foreground_color_border=:lightgrey)
            p2 = Plots.scatter(array[1,:], array[3,:], marker_z=R.assignments, markerstrokewidth=0, color=:rainbow, markersize=2,
                label=nothing, xlabel="time window", ylabel="sign", legend=:none, fontize=13, foreground_color_border=:lightgrey)
            p3 = Plots.scatter(array[2,:], array[3,:], marker_z=R.assignments, markerstrokewidth=0, color=:rainbow, markersize=2,
                label=nothing, xlabel="rise:decay ratio", ylabel="sign", legend=:none, fontize=13, foreground_color_border=:lightgrey)
            p4 = Plots.scatter(array[1,:], array[2,:], array[3,:], marker_z=R.assignments, markerstrokewidth=0, color=:rainbow, markersize=4,
                label=nothing, xlabel="time window", ylabel="rise:decay ratio", zlabel="sign", legend=:none, fontize=16, foreground_color_border=:lightgrey)
            l = @layout [a{0.8h} 
                Plots.grid(1,3)]
            Plots.plot(p4, p1, p2, p3, layout=l, show=true, size=(600,600))
        end
    end
end



# inputs:
# winner_param_all_neurons, sig_neuron_idx, gof_all_neurons: outputs from shift_test_winner()
# outputs:
# x, y, z: vectors of the same length for clustering

function sig_neurons_winner_param_space(winner_param_all_neurons, sig_neuron_idx, gof_all_neurons)
    winner_param_sig_neurons = winner_param_all_neurons[:, sig_neuron_idx]
    x = winner_param_sig_neurons[1,:] # starting point of kernel
    y = winner_param_sig_neurons[2,:] # rise rate
    z = winner_param_sig_neurons[4,:] # decay rate
    
    s = Vector{String}(undef, length(x))
    for i = 1:length(x)
        if (x[i]==0) && (gof_all_neurons[i]>0)
            s[i] = "red" # a neuron positively correlated with NSM
        elseif (x[i]==0) && (gof_all_neurons[i]<0)
            s[i] = "green" # a neuron negatively correlated with NSM
        elseif x[i]<0
            s[i] = "purple" # a differenatiator neuron        
        end
    end
#     c = gof_all_neurons[sig_neuron_idx] # amt of correlation with winner_convolved_nsm
    
#     cs = ColorScheme([colorant"grey", colorant"magenta"])
#     colors = get.(Ref(cs), c)
    
#     Plots.scatter(x,y,z, label=nothing, markersize=3, markerstrokewidth=0, fontsize=14, foreground_color_border=:lightgrey,
#         color=colors, xlabel="kernel time window", ylabel="kernel rise:decay ratio", zlabel="kernel sign", 
#         title="$(length(sig_neuron_idx)) neurons passed the shifting test\n with kernel-convolved NSM")
    
    return x, y, z, s
end
    

    
# inputs:
# array: an array of neural activity across time, eg smooth_traces_array, or that of a subset of neurons
# method: a string, options are "pearson", "spearman"
# clustered (optional kwarg): a KmeansResult object output from cluster_by_param(array; plot=true, num_cluster=best_ncluster)
# neuron_id (optional kwarg): a vector of neuron #, need to be the same length as number of rows in array
# output:
# a plot

function plot_correlation_matrix(array, method; clustered=[], neuron_id=[])
    if isempty(clustered)
        if method=="pearson"
            neuron_correlation_matrix = cor(array, dims=2)
        elseif method=="spearman"
            neuron_correlation_matrix = corspearman(array, dims=2)
        end
        
        if isempty(neuron_id)
            Plots.heatmap(neuron_correlation_matrix, aspect=1, color=:bluesreds, xlabel="neuron", ylabel="neuron", size=(550,500),
                xticks=(1:size(array,1), neuron_id), yticks=(1:size(array,1), neuron_id))
        else
            Plots.heatmap(neuron_correlation_matrix, aspect=1, color=:bluesreds, xlabel="neuron", ylabel="neuron", size=(550,500))
        end    
    else
        kmns_order = sortperm(clustered.assignments)
        if method=="pearson"
            neuron_correlation_matrix = cor(array[kmns_order,:], dims=2)
        elseif method=="spearman"
            neuron_correlation_matrix = corspearman(array[kmns_order,:], dims=2)
        end
        
        if isempty(neuron_id)
            Plots.heatmap(neuron_correlation_matrix, aspect=1, color=:bluesreds, xlabel="neuron", ylabel="neuron", size=(550,500))
        else
            Plots.heatmap(neuron_correlation_matrix, aspect=1, color=:bluesreds, xlabel="neuron", ylabel="neuron", size=(550,500), 
                xticks=(1:size(array,1), neuron_id[kmns_order]), yticks=(1:size(array,1), neuron_id[kmns_order]))
        end
    end
end



# inputs:
# nsm: a vector of nsm activity, of an average of activities of 2 nsms
# smooth_window (optional kwarg): default is 5, larger means more smoothing
# search_window (optional kwarg): default is 5, larger means fewer peaks/troughs identified
# outputs:
# nsm_peaks: a vector of timepoints where nsm activity is at its local maximum
# nsm_troughs: a vector of timepoints where nsm activity is at its local minimum
# a plot(optional)

function find_nsm_peak_trough(nsm; smooth_window=5, search_window=5, plot=false)
    # smooth the trace out
    w = smooth_window
    moving_avg = zeros(length(nsm)-w*2)
    for i = w+1:length(nsm)-w
        moving_avg[i-w] = sum(nsm[i-w:i+w])/(w*2+1)
    end
    
    # find peaks and troughs
    v = search_window
    nsm_peaks = Vector{Float64}([])
    nsm_troughs = Vector{Float64}([])
    for t = v+1:length(moving_avg)-v
        if moving_avg[t] > maximum(moving_avg[t-v:t-1]) && moving_avg[t] > maximum(moving_avg[t+1:t+v])
            push!(nsm_peaks, t+w)
        elseif moving_avg[t] < minimum(moving_avg[t-v:t-1]) && moving_avg[t] < minimum(moving_avg[t+1:t+v])
            push!(nsm_troughs, t+w)
        end
    end
    
    if plot
        Plots.plot(nsm, xlabel="time (loops)", ylabel="NSM activity", color=:black, linewidth=3, 
            fontsize=14, label=nothing, foreground_color_border=:lightgrey)
        for i = nsm_peaks
            vline!([i], color=:red, alpha=0.7, label=nothing)
        end
        for i = nsm_troughs
            vline!([i], color=:blue, alpha=0.7, label=nothing)
        end
    end
    
    nsm_peaks, nsm_troughs
end



# inputs:
# nsm: a vector of zscored smoothed nsm activity, of an average of activities of 2 nsms
# activity_thresh (optional kwarg): default is 0, can change between -1 and 1
# outputs:
# nsm_high_range: a 2-column array, where nsm_high_range[i,1]:nsm_high_range[i,2] are timepoints where nsm activity is constantly above activity_thresh
# nsm_low_range: a 2-column array, where nsm_low_range[i,1]:nsm_low_range[i,2] are timepoints where nsm activity is constantly below activity_thresh
# a plot (optional)

function find_nsm_high_low_range(nsm; activity_thresh=0, plot=false)
    # binarize all timepoints into high and low
    high_idx = findall(x->x.>activity_thresh, nsm)
    low_idx = setdiff(1:length(nsm), high_idx)
    
    # find stretches of time where nsm acitivty is constantly high for >5 loops
    nsm_high_start = Vector{Float64}([])
    nsm_high_end = Vector{Float64}([])
    for i = 5:length(nsm)-4
        if in(high_idx).(i-1)==false && in(high_idx).(i) && in(high_idx).(i+1) && in(high_idx).(i+2) && in(high_idx).(i+3) && in(high_idx).(i+4)
            push!(nsm_high_start, i)
        elseif in(high_idx).(i-4) && in(high_idx).(i-3) && in(high_idx).(i-2) && in(high_idx).(i-1) && in(high_idx).(i) && in(high_idx).(i+1)==false
            push!(nsm_high_end, i)
        end
    end
    
    nsm_high_range = hcat(nsm_high_start, nsm_high_end)

    # find stretches of time where nsm acitivty is constantly low for >5 loops
    nsm_low_start = Vector{Float64}([])
    nsm_low_end = Vector{Float64}([])
    for i = 5:length(nsm)-4
        if in(low_idx).(i-1)==false && in(low_idx).(i) && in(low_idx).(i+1) && in(low_idx).(i+2) && in(low_idx).(i+3) && in(low_idx).(i+4)
            push!(nsm_low_start, i)
        elseif in(low_idx).(i-4) && in(low_idx).(i-3) && in(low_idx).(i-2) && in(low_idx).(i-1) && in(low_idx).(i) && in(low_idx).(i+1)==false
            push!(nsm_low_end, i)
        end
    end 
    
    nsm_low_range = hcat(nsm_low_start, nsm_low_end)
    
    if plot
        Plots.plot(nsm, xlabel="time (loops)", ylabel="NSM activity", color=:black, linewidth=3, 
            label=nothing, fontsize=14, foreground_color_border=:lightgrey)
        for i = 1:size(nsm_high_range,1)
            vspan!([nsm_high_range[i,1], nsm_high_range[i,2]], color=:red, alpha=0.4, label=nothing)
        end
        for i = 1:size(nsm_low_range,1)
            vspan!([nsm_low_range[i,1], nsm_low_range[i,2]], color=:blue, alpha=0.4, label=nothing)
        end
    end
    
    nsm_high_range, nsm_low_range
end
    


# inputs:
# trend: a vector of neural trace or behavioral variable, e.g. trend = velocity; or trend = traces_array[30,:]
# events: a vector of confocal timepoints at which event happened, e.g. nsm_peaks, egg laying events; if event only happened once, put the single timepoint in square brackets, [105]
# window_size: a scalar indicating the width of window of interest, e.g. window_size = 3 means you are looking at 3 timepoints before and 3 timepoints after each event
# y_label: a string to indicate what trend is varying with the event, e.g. "neuron #50", "velocity (mm/s)"
# outputs:
# a plot with grey lines indicating the +/-1 standard error
# a vector of mean value of trend within window of interest
# a vector of standard error of trend within window of interest; undefined if event only happened once

function eta(trend, events, window_size, y_label)
    w = window_size
    values = zeros(1,2*w+1)
    for t = events
        t_selected = trend[t-w:t+w]
        values = vcat(values, t_selected')
    end
    val = values[2:end,:]
    val_mean = mean(val, dims=1)
    val_std = std(val, dims=1)
    val_sem = val_std/sqrt(length(events))
    upper_bound = val_mean .+ val_sem
    lower_bound = val_mean .- val_sem

    figure(figsize=(10,5))
    plt.plot(-w:w, val_mean', color="black", linewidth=3)
    plt.plot(-w:w, lower_bound', color="black", alpha=0.2)
    plt.plot(-w:w, upper_bound', color="black", alpha=0.2)
    plt.axvline(0, color="red")
    xlabel("time relative to event (loops)", fontsize=14)
    ylabel(y_label, fontsize=14)
    title("event triggered average", fontsize=14)
    
    return val_mean, val_sem
end



function invert_dict(dict, warning::Bool = false)
    vals = collect(values(dict))
    dict_length = length(unique(vals))

    if dict_length < length(dict)
        if warning
            warn("Keys/Vals are not one-to-one")
        end 

        linked_list = Array[]

        for i in vals 
            push!(linked_list,[])
        end 

        new_dict = Dict(zip(vals, linked_list))

        for (key,val) in dict 
            push!(new_dict[val],key)
        end
    else
        key = collect(keys(dict))

        counter = 0
        for (k,v) in dict 
            counter += 1
            vals[counter] = v
            key[counter] = k
        end
        new_dict = Dict(zip(vals, key))
    end 

    return new_dict
end


function findPresynapticPartners(list_neurons, data)
    # n_synapse_chemical = 0
    # n_synapse_electrical = 0
    list_edges = filter(x->x["post"] in list_neurons, data)
    presynaptic_partner_dict = Dict() # a single dictionary for a neuron class (D and V count as two different neuron classes)
    
    if length(list_edges) > 0
        for edge = list_edges
            #0: chemical synapse 2: electrical synapse (gap junction)
            if edge["typ"] == 0
                n_synapse_chemical = sum(edge["syn"])
                n_synapse_electrical = 0
            elseif edge["typ"] == 2
                n_synapse_chemical = 0
                n_synapse_electrical = sum(edge["syn"])
            else
                print("$(list_neurons[1]) has an unknown edge type")
            end
            
            if haskey(presynaptic_partner_dict, edge["pre"]) # happens if neuron X projects to both the left and right neurons in neuron class Y
                presynaptic_partner_dict[edge["pre"]] = presynaptic_partner_dict[edge["pre"]] + n_synapse_chemical + n_synapse_electrical
            else
                presynaptic_partner_dict[edge["pre"]] = n_synapse_chemical + n_synapse_electrical
            end
        end
    else
        print("$(list_neurons[1]) has no presynaptic partners\n")
    end # if length(list_edges)
    
    return presynaptic_partner_dict
end

    

function quantifySynapticInputModulated(presynaptic_partner_dict, modulated_neuron_id)
    modulated_input = 0
    all_input = 0
    
    for key in keys(presynaptic_partner_dict)
        all_input = all_input + presynaptic_partner_dict[key]
        if string(key) in modulated_neuron_id # for standalone neuron classes
            modulated_input = modulated_input + presynaptic_partner_dict[key]
        elseif string(key[end]) in ["L", "R"] && string(key[1:end-1]) in modulated_neuron_id # for neuron classes with L/R pairs
            modulated_input = modulated_input + presynaptic_partner_dict[key]
        end
    end
    
    return modulated_input, all_input
end

    

function quantifySynapticInputReceptor(presynaptic_partner_dict, receptor_neuron_id)
    modulated_input = 0
    all_input = 0
    
    for key in keys(presynaptic_partner_dict)
        all_input = all_input + presynaptic_partner_dict[key]
        if string(key) in receptor_neuron_id # for standalone neuron classes
            modulated_input = modulated_input + presynaptic_partner_dict[key]
        elseif string(key[end]) in ["L", "R"] && string(key[1:end-1]) in receptor_neuron_id # for neuron classes with L/R pairs
            modulated_input = modulated_input + presynaptic_partner_dict[key]
        end
    end
    
    return modulated_input, all_input
end

    

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)
