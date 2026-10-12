function plot_matrix_admm_history( ...
    r_history,...
    dxy_history,...
    round_y_collect,...
    y_non_ratio,...
    beta_history,...
    admm_opts)
%
% Plot convergence history of matrix ADMM algorithm
%
% Inputs:
%   r_history       : KKT residual h^k
%   dxy_history     : feasibility residual p^k
%   round_y_collect : binary gap
%   y_non_ratio     : fractional ratio
%   beta_history    : adaptive stepsize
%   admm_opts       : plotting options
%


%% ======================================================
% Convert vectors
% ======================================================

r_plot      = r_history(:);
dxy_plot    = dxy_history(:);
round_plot  = round_y_collect(:);
ynon_plot   = y_non_ratio(:);
beta_plot   = beta_history(:);



%% ======================================================
% Determine subplot-specific floor values
% ======================================================

floor1 = choose_plot_floor( ...
    [r_plot; dxy_plot]);


floor2 = choose_plot_floor( ...
    [round_plot; ynon_plot]);


floor3 = choose_plot_floor( ...
    beta_plot);



%% ======================================================
% Process semilogy curves
% ======================================================

r_plot = process_log_curve( ...
    r_plot, ...
    floor1);


dxy_plot = process_log_curve( ...
    dxy_plot, ...
    floor1);


round_plot = process_log_curve( ...
    round_plot, ...
    floor2);


ynon_plot = process_log_curve( ...
    ynon_plot, ...
    floor2);


beta_plot = process_log_curve( ...
    beta_plot, ...
    floor3);



%% ======================================================
% X range
% ======================================================

max_iter = max([
    length(r_plot), ...
    length(dxy_plot), ...
    length(round_plot), ...
    length(ynon_plot), ...
    length(beta_plot)
    ]);

max_iter = 619;
x_range = [
    1, ...
    max_iter + 0.1 * max_iter
    ];



%% ======================================================
% Global font
% ======================================================

set(groot, ...
    'defaultAxesFontName', ...
    'Times New Roman', ...
    'defaultTextFontName', ...
    'Times New Roman');



%% ======================================================
% Style
% ======================================================

blue   = [0.0000 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
green  = [0.4660 0.6740 0.1880];
purple = [0.4940 0.1840 0.5560];


% Curve width
lw = 2.2;


% Font sizes
axis_font_size   = 13;
label_font_size  = 14;
title_font_size  = 14;
legend_font_size = 13;


% Axis width
axis_line_width = 1.2;



%% ======================================================
% Legend position control
%
% false:
%   use ordinary northeast position
%
% true:
%   manually place legend in the blank region
% ======================================================

manual_legend = false;


% Manual legend position parameters
%
% Larger legend_x -> move right
% Smaller legend_y -> move down
legend_x = 0.62;
legend_y = 0.34;



%% ======================================================
% Fixed-beta plotting control
%
% true:
%   beta_k is fixed.
%   Put the constant beta value in the middle of panel (c).
%
% false:
%   beta_k is adaptive.
%   Use the original logarithmic-axis setting.
% ======================================================

fixbeta = true;



%% ======================================================
% Figure
% ======================================================

fig = figure( ...
    'Units', ...
    'centimeters', ...
    'Position', ...
    [2 2 15.0 13.0], ...
    'Color', ...
    'w', ...
    'Renderer', ...
    'painters');


layout = tiledlayout( ...
    fig, ...
    3, ...
    1, ...
    'TileSpacing', ...
    'compact', ...
    'Padding', ...
    'compact');


ax = gobjects(3,1);



%% ======================================================
% (a) convergence
% ======================================================

ax(1) = nexttile(layout);


h1 = semilogy( ...
    ax(1), ...
    r_plot, ...
    '-', ...
    'Color', ...
    blue, ...
    'LineWidth', ...
    lw);


hold(ax(1), 'on');


h2 = semilogy( ...
    ax(1), ...
    dxy_plot, ...
    '--', ...
    'Color', ...
    orange, ...
    'LineWidth', ...
    lw);


hold(ax(1), 'off');


ylabel( ...
    ax(1), ...
    'Error', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    label_font_size, ...
    'FontWeight', ...
    'normal');


title( ...
    ax(1), ...
    '(a) Convergence metrics', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    title_font_size, ...
    'FontWeight', ...
    'normal');



%% ======================================================
% Legend for panel (a)
% ======================================================

lgd1 = legend( ...
    ax(1), ...
    [h1 h2], ...
    {'$h^k$', '$p^k$'}, ...
    'Interpreter', ...
    'latex', ...
    'Box', ...
    'off');


set( ...
    lgd1, ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    legend_font_size, ...
    'ItemTokenSize', ...
    [28 16]);


lgd1.AutoUpdate = 'off';



%% ======================================================
% Legend position
% ======================================================

if manual_legend

    % --------------------------------------------------
    % Manual position
    % --------------------------------------------------

    drawnow;


    lgd1.Units = ...
        'normalized';


    ax_pos = ...
        ax(1).Position;


    lgd_pos = ...
        lgd1.Position;


    % Horizontal position
    lgd_pos(1) = ...
        ax_pos(1) ...
        + legend_x * ax_pos(3);


    % Vertical position
    lgd_pos(2) = ...
        ax_pos(2) ...
        + legend_y * ax_pos(4);


    lgd1.Position = ...
        lgd_pos;


else

    % --------------------------------------------------
    % Ordinary legend position
    % --------------------------------------------------

    lgd1.Location = ...
        'southwest';

end



%% ======================================================
% (b) identification
% ======================================================

ax(2) = nexttile(layout);


h3 = semilogy( ...
    ax(2), ...
    round_plot, ...
    '-.', ...
    'Color', ...
    green, ...
    'LineWidth', ...
    lw);


hold(ax(2), 'on');


h4 = semilogy( ...
    ax(2), ...
    ynon_plot, ...
    '-', ...
    'Color', ...
    purple, ...
    'LineWidth', ...
    lw);


hold(ax(2), 'off');


ylabel( ...
    ax(2), ...
    'Error', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    label_font_size, ...
    'FontWeight', ...
    'normal');


title( ...
    ax(2), ...
    '(b) Support identification', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    title_font_size, ...
    'FontWeight', ...
    'normal');


lgd2 = legend( ...
    ax(2), ...
    [h3 h4], ...
    {'Binary gap', 'Fractional ratio'}, ...
    'Location', ...
    'southwest', ...
    'Box', ...
    'off');


set( ...
    lgd2, ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    legend_font_size, ...
    'ItemTokenSize', ...
    [28 16]);


lgd2.AutoUpdate = 'off';



%% ======================================================
% (c) beta
% ======================================================

ax(3) = nexttile(layout);


semilogy( ...
    ax(3), ...
    beta_plot, ...
    '-', ...
    'Color', ...
    blue, ...
    'LineWidth', ...
    lw);


xlabel( ...
    ax(3), ...
    'Iteration', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    label_font_size, ...
    'FontWeight', ...
    'normal');


ylabel( ...
    ax(3), ...
    'Stepsize', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    label_font_size, ...
    'FontWeight', ...
    'normal');


title( ...
    ax(3), ...
    '(c) Dual stepsize', ...
    'FontName', ...
    'Times New Roman', ...
    'FontSize', ...
    title_font_size, ...
    'FontWeight', ...
    'normal');



%% ======================================================
% Axis setting
% ======================================================

for i = 1:3

    set( ...
        ax(i), ...
        'FontName', ...
        'Times New Roman', ...
        'FontSize', ...
        axis_font_size, ...
        'FontWeight', ...
        'normal', ...
        'LineWidth', ...
        axis_line_width, ...
        'TickDir', ...
        'out', ...
        'TickLength', ...
        [0.018 0.018], ...
        'Box', ...
        'off', ...
        'XGrid', ...
        'on', ...
        'YGrid', ...
        'on', ...
        'XMinorGrid', ...
        'off', ...
        'YMinorGrid', ...
        'off', ...
        'GridAlpha', ...
        0.10, ...
        'MinorGridAlpha', ...
        0.05);


    ax(i).XAxisLocation = ...
        'bottom';


    ax(i).YAxisLocation = ...
        'left';


    ax(i).YMinorTick = ...
        'off';

end



%% ======================================================
% Y-axis limits
% ======================================================

% ------------------------------------------------------
% Panel (a)
% ------------------------------------------------------

yl1 = ylim(ax(1));


ylim( ...
    ax(1), ...
    [floor1, yl1(2)]);



% ------------------------------------------------------
% Panel (b)
% ------------------------------------------------------

yl2 = ylim(ax(2));


ylim( ...
    ax(2), ...
    [floor2, yl2(2)]);



% ------------------------------------------------------
% Panel (c)
% ------------------------------------------------------

if fixbeta

    % ==================================================
    % Fixed beta:
    % place the constant beta value at the center
    % of the logarithmic axis.
    %
    % For example:
    % beta = 1000
    %
    % Y limits:
    % 500 -- 2000
    %
    % Y ticks:
    % 500, 1000, 2000
    % ==================================================

    beta_valid = beta_plot( ...
        isfinite(beta_plot) ...
        & beta_plot > 0);


    if ~isempty(beta_valid)

        beta0 = beta_valid(1);


        ylim( ...
            ax(3), ...
            [ ...
            beta0 / 2, ...
            beta0 * 2 ...
            ]);

    end


else

    % ==================================================
    % Adaptive beta:
    % original plotting rule
    % ==================================================

    yl3 = ylim(ax(3));


    ylim( ...
        ax(3), ...
        [floor3, yl3(2)]);

end



%% ======================================================
% Y-axis ticks
% ======================================================

% ------------------------------------------------------
% Panel (a)
% ------------------------------------------------------

set_three_visible_log_ticks( ...
    ax(1));


% ------------------------------------------------------
% Panel (b)
% ------------------------------------------------------

set_three_nice_log_ticks( ...
    ax(2));


% ------------------------------------------------------
% Panel (c)
% ------------------------------------------------------

if fixbeta

    beta_valid = beta_plot( ...
        isfinite(beta_plot) ...
        & beta_plot > 0);


    if ~isempty(beta_valid)

        beta0 = beta_valid(1);


        beta_ticks = [
            beta0 / 2, ...
            beta0, ...
            beta0 * 2
            ];


        set( ...
            ax(3), ...
            'YTick', ...
            beta_ticks);


        beta_tick_labels = {
            sprintf('%.4g', beta0 / 2), ...
            sprintf('%.4g', beta0), ...
            sprintf('%.4g', beta0 * 2)
            };


        set( ...
            ax(3), ...
            'YTickLabel', ...
            beta_tick_labels);


        ax(3).YMinorTick = ...
            'off';

    end


else

    set_three_visible_log_ticks( ...
        ax(3));

end



%% ======================================================
% Shared x-axis
% ======================================================

linkaxes( ...
    ax, ...
    'x');


for i = 1:3

    xlim( ...
        ax(i), ...
        x_range);

end


% Only show x tick labels for the last subplot
ax(1).XTickLabel = [];

ax(2).XTickLabel = [];



%% ======================================================
% Save
% ======================================================

if isfield(admm_opts, 'save_plot') ...
        && admm_opts.save_plot


    if isfield(admm_opts, 'figure_dir')

        folder = ...
            admm_opts.figure_dir;

    else

        folder = ...
            'figures';

    end


    if ~exist(folder, 'dir')

        mkdir(folder);

    end


    exportgraphics( ...
        fig, ...
        fullfile( ...
            folder, ...
            'matrix_admm_history.pdf'), ...
        'ContentType', ...
        'vector', ...
        'BackgroundColor', ...
        'white');

end


end



%% ======================================================
% Helper function
%
% Choose subplot-specific plotting floor
% ======================================================

function floor_value = choose_plot_floor(data)


data = data(:);


% Remove NaN and Inf
data = data(isfinite(data));


if isempty(data)

    floor_value = ...
        1e-16;

    return;

end


data_min = ...
    min(data);


if data_min < 5e-16

    floor_value = ...
        5e-16;

else

    floor_value = ...
        data_min;

end


end



%% ======================================================
% Helper function
%
% Process curve for logarithmic plotting
% ======================================================

function y = process_log_curve( ...
    y, ...
    floor_value)


y = y(:);


y(~isfinite(y)) = ...
    NaN;


idx = find( ...
    y <= 0, ...
    1, ...
    'first');


if ~isempty(idx)

    y(idx) = ...
        floor_value;


    if idx < length(y)

        y(idx+1:end) = ...
            NaN;

    end

end


end



%% ======================================================
% Helper function
%
% Set exactly THREE logarithmic Y ticks
% ======================================================

function set_three_visible_log_ticks(ax)


yl = ...
    ylim(ax);


if any(~isfinite(yl)) || ...
        yl(1) <= 0 || ...
        yl(2) <= 0

    return;

end


log_lower = ...
    log10(yl(1));


log_upper = ...
    log10(yl(2));


if abs(log_lower - round(log_lower)) < 1e-10

    log_lower = ...
        round(log_lower);

end


if abs(log_upper - round(log_upper)) < 1e-10

    log_upper = ...
        round(log_upper);

end


lower_exp = ...
    ceil(log_lower);


upper_exp = ...
    floor(log_upper);


exponents = ...
    lower_exp:upper_exp;



if length(exponents) >= 3

    idx = round( ...
        linspace( ...
            1, ...
            length(exponents), ...
            3));


    selected_exp = ...
        exponents(idx);


    selected_exp = ...
        unique( ...
            selected_exp, ...
            'stable');


    if length(selected_exp) < 3

        selected_exp = [
            exponents(1), ...
            exponents(ceil(end/2)), ...
            exponents(end)
            ];

    end


    ticks = ...
        10 .^ selected_exp;


elseif length(exponents) == 2

    tick_low = ...
        10^exponents(1);


    tick_high = ...
        10^exponents(2);


    tick_mid = ...
        sqrt( ...
            tick_low * tick_high);


    ticks = [
        tick_low, ...
        tick_mid, ...
        tick_high
        ];


else

    ticks = ...
        logspace( ...
            log_lower, ...
            log_upper, ...
            3);

end


set( ...
    ax, ...
    'YTick', ...
    ticks);


ax.YMinorTick = ...
    'off';


end



%% ======================================================
% Helper function
%
% Set THREE nice logarithmic ticks for panel (b)
% ======================================================

function set_three_nice_log_ticks(ax)


yl = ...
    ylim(ax);


if any(~isfinite(yl)) || ...
        yl(1) <= 0 || ...
        yl(2) <= 0

    return;

end



%% Lower tick

tick_low = ...
    yl(1);



%% Logarithmic midpoint

raw_mid = ...
    sqrt( ...
        yl(1) * yl(2));



%% Nice midpoint

tick_mid = ...
    nearest_nice_number( ...
        raw_mid);



%% Nice upper tick

tick_high = ...
    nice_number_below( ...
        yl(2));



%% Safety adjustment

if tick_mid <= tick_low

    tick_mid = ...
        sqrt( ...
            tick_low * tick_high);


    tick_mid = ...
        nearest_nice_number( ...
            tick_mid);

end


if tick_mid >= tick_high

    tick_mid = ...
        sqrt( ...
            tick_low * tick_high);

end



%% Apply ticks

ticks = [
    tick_low, ...
    tick_mid, ...
    tick_high
    ];


set( ...
    ax, ...
    'YTick', ...
    ticks);



%% Clean numeric labels

labels = cell( ...
    1, ...
    length(ticks));


for j = 1:length(ticks)

    labels{j} = ...
        sprintf( ...
            '%.3g', ...
            ticks(j));

end


set( ...
    ax, ...
    'YTickLabel', ...
    labels);


ax.YMinorTick = ...
    'off';


end



%% ======================================================
% Helper function
%
% Return nearest visually nice number
% ======================================================

function y = nearest_nice_number(x)


if x <= 0 || ~isfinite(x)

    y = x;

    return;

end


exponent = ...
    floor(log10(x));


scale = ...
    10^exponent;


mantissa = ...
    x / scale;


nice_set = [
    1, ...
    1.2, ...
    1.5, ...
    2, ...
    2.5, ...
    5, ...
    10
    ];


[~, idx] = ...
    min( ...
        abs( ...
            nice_set - mantissa));


y = ...
    nice_set(idx) * scale;


end



%% ======================================================
% Helper function
%
% Largest nice number <= x
% ======================================================

function y = nice_number_below(x)


if x <= 0 || ~isfinite(x)

    y = x;

    return;

end


exponent = ...
    floor(log10(x));


scale = ...
    10^exponent;


mantissa = ...
    x / scale;


nice_set = [
    1, ...
    1.2, ...
    1.5, ...
    2, ...
    2.5, ...
    5, ...
    10
    ];


candidates = ...
    nice_set( ...
        nice_set <= mantissa);


if isempty(candidates)

    exponent = ...
        exponent - 1;


    scale = ...
        10^exponent;


    y = ...
        10 * scale;

else

    y = ...
        max(candidates) * scale;

end


end