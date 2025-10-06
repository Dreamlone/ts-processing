def cell_to_generate_simple_autocorr_plot():
    lag = 3

    cmap = plt.colormaps["coolwarm"]
    colors = [cmap(i / (n - lag)) for i in range(n - lag)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, values, color="gray", lw=1.5, label="time series")
    ax.scatter(t, values, color="gray", s=20, zorder=3)

    for i, color in enumerate(colors, start=lag):
        if (i - lag) % 10 != 0:
            continue
        ax.plot([t[i - lag], t[i]], [values[i - lag], values[i]], color=color, lw=2, alpha=0.8)
        ax.scatter([t[i - lag], t[i]], [values[i - lag], values[i]], color=color, s=50, zorder=3)

    ax.set_title(f"Autocorrelation illustration for lag={lag} (every 10th pair is shown)", fontsize=13)
    ax.set_xlabel("Datetime index (t)")
    ax.set_ylabel("y(t)")
    ax.grid(alpha=0.2)
    plt.show()
