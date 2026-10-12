function best_value = best_at_time_budget(candidate_times, best_history, budget)
%BEST_AT_TIME_BUDGET Best-so-far value available by an empirical time budget.
% candidate_times(i) is the cumulative time when candidate i becomes
% available; best_history(i) is the best objective seen through i.

    candidate_times = candidate_times(:);
    best_history = best_history(:);
    if numel(candidate_times) ~= numel(best_history)
        error('candidate_times and best_history must have the same length.');
    end

    idx = find(candidate_times <= budget & isfinite(best_history), 1, 'last');
    if isempty(idx)
        best_value = NaN;
    else
        best_value = best_history(idx);
    end
end
