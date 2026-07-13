import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

"Seed for random checkerboard"
#rng = np.random.default_rng(5473644335)

"Seed for constant diffusion"
rng = np.random.default_rng(8839394)


def get_environment(l_lambda: float, u_lambda: float, N: int) -> np.array:
    """Create a random checkerboard environment a(x) of length N in each direction"""
    
    return rng.choice([l_lambda, u_lambda], N**2).reshape((N,N))



def time_step(x: float, y: float, dt: float, environment: np.array, N: int) -> tuple:
    """Update position (x,y) by time dt, and return (x+dx,y+dy)"""

    "shift so (x=-N-0.5,y=-N-0.5) is the (0,0) entry of the environment matrix, increasing up/right"
    "sigma^2 = a"
    sigma = np.sqrt(environment[int(np.floor(x+ N + 0.5)), int(np.floor(y+N+0.5))])
    x += sigma * rng.normal() * np.sqrt(dt)
    y += sigma * rng.normal() * np.sqrt(dt)

    return (x,y)


def make_plot(x_path: np.array, y_path: np.array, environment: np.array, N: int):
    """Make a scatter plot of x_path, y_path on the background of environment.
    The entries of environment, indexed in 2d space from bottom left to top right, should
    be coloured black or white depending on the two values of the array, spanning space from
    -N to N in each direction with 2N+1 squares."""

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    fig.subplots_adjust(top=0.88, left=0.1, right=0.96, bottom=0.1)
    
    title_font = {
        "family": "DejaVu Sans",
        "size": 14,
        "weight": "bold",
        "color": "0.12",
    }
    label_font = {
        "family": "DejaVu Sans",
        "size": 10,
        "color": "0.2",
    }
    ax.set_title("Particle Path", loc="center", fontdict=title_font, pad=16)

    values = np.unique(environment)
    if len(values) == 1:
        ax.set_facecolor("0.67")
    else:
        black_value = values[0]
        ax.set_facecolor("white")
        for i in range(environment.shape[0]):
            for j in range(environment.shape[1]):
                if environment[i, j] == black_value:
                    ax.add_patch(Rectangle(
                        (i - N - 0.5, j - N - 0.5),
                        1,
                        1,
                        facecolor="0.33",
                        edgecolor="none",
                        linewidth=0,
                        zorder=0,
                    ))

    for grid_line in np.linspace(-N - 0.5, N + 0.5, 2 * N + 2):
        ax.axvline(grid_line, color="0.15", linestyle=(0, (3, 3)),
                   linewidth=0.5, alpha=0.45, zorder=1)
        ax.axhline(grid_line, color="0.15", linestyle=(0, (3, 3)),
                   linewidth=0.5, alpha=0.45, zorder=1)

    ax.plot(x_path, y_path, c="#d62728", linewidth=1.1, zorder=2)
    ax.scatter(x_path, y_path, s=1, c="#d62728", zorder=3)
    ax.set_xlim(-N - 0.5, N + 0.5)
    ax.set_ylim(-N - 0.5, N + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.2")
        spine.set_linewidth(0.8)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig("homogenization_diffusion.png", bbox_inches="tight")
    plt.show()


def make_animation(x_path: np.array, y_path: np.array, environment: np.array, N: int, save: bool,
                   frames_per_step: int = 5, interval: int = 10, dt: float = 1):
    """Animate the particle path on the background of environment.
    """

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    fig.subplots_adjust(top=0.88, left=0.1, right=0.96, bottom=0.1)
    
    title_font = {
        "family": "DejaVu Sans",
        "size": 14,
        "weight": "bold",
        "color": "0.12",
    }
    
    values = np.unique(environment)
    if len(values) == 1:
        ax.set_facecolor("0.67")
        ax.set_title("Constant coefficient diffusion", loc="center", fontdict=title_font, pad=16)
    else:
        ax.set_title("Diffusion in a random checkerboard environment", loc="center", fontdict=title_font, pad=16)
        black_value = values[0]
        ax.set_facecolor("white")
        for i in range(environment.shape[0]):
            for j in range(environment.shape[1]):
                if environment[i, j] == black_value:
                    ax.add_patch(Rectangle(
                        (i - N - 0.5, j - N - 0.5),
                        1,
                        1,
                        facecolor="0.33",
                        edgecolor="none",
                        linewidth=0,
                        zorder=0,
                    ))

    for grid_line in np.linspace(-N - 0.5, N + 0.5, 2 * N + 2):
        ax.axvline(grid_line, color="0.15", linestyle=(0, (3, 3)),
                   linewidth=0.5, alpha=0.45, zorder=1)
        ax.axhline(grid_line, color="0.15", linestyle=(0, (3, 3)),
                   linewidth=0.5, alpha=0.45, zorder=1)

    path_line, = ax.plot([], [], c="#d62728", linewidth=1.1, zorder=2)
    particle, = ax.plot([], [], marker="o", c="#d62728", markersize=2.5, zorder=3)
    time_text = ax.text(1, 1.02, "", transform=ax.transAxes,
                        fontsize=11, fontfamily="DejaVu Sans", color="0.15",
                        horizontalalignment="right", verticalalignment="bottom",
                        clip_on=False)
    ax.set_xlim(-N - 0.5, N + 0.5)
    ax.set_ylim(-N - 0.5, N + 0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks(np.linspace(-N,N,2*N+1))
    ax.set_yticks(np.linspace(-N,N,2*N+1))
    for spine in ax.spines.values():
        spine.set_color("0.2")
        spine.set_linewidth(0.8)
    ax.set_aspect("equal", adjustable="box")

    total_frames = max(1, (len(x_path) - 1) * frames_per_step + 1)

    def update(frame):
        if len(x_path) == 1:
            current_x = x_path[0]
            current_y = y_path[0]
            path_line.set_data([current_x], [current_y])
            particle.set_data([current_x], [current_y])
            #time_text.set_text("t = 0.000")
            return path_line, particle, time_text

        step = min(frame // frames_per_step, len(x_path) - 2)
        progress = (frame % frames_per_step) / frames_per_step
        if frame == total_frames - 1:
            step = len(x_path) - 2
            progress = 1

        current_x = (1 - progress) * x_path[step] + progress * x_path[step + 1]
        current_y = (1 - progress) * y_path[step] + progress * y_path[step + 1]
        path_line.set_data(
            np.append(x_path[:step + 1], current_x),
            np.append(y_path[:step + 1], current_y),
        )
        particle.set_data([current_x], [current_y])
        #time_text.set_text(f"t = {((step + progress) * dt):.3f}")
        return path_line, particle, time_text

    animation = FuncAnimation(fig, update, frames=total_frames, interval=interval,
                              blit=True, repeat=False)
    if save:
        if len(values) == 1:
            animation.save("constant_diffusion.mp4")
        else:
            animation.save("particle_path.mp4")
    plt.show()
    return animation


def run_simulation(dt: float, steps: int, N: int, l_lambda: float, u_lambda: float, save: bool = False):
    """Run the simulation using steps of size dt, confined to an environment
    of dimensions (2N+1) x (2N+1), centered at 0.

    - for dx << 1 we should have sqrt{lambda dt} << 1
    - for us to exit the center box in a reasonable number of steps we should also
        have something like 20 * sqrt{lambda dt} > 1
    - this implies that we should choose dt ~ 1/ 400 lambda
    """
    
    x,y = 0,0
    x_path = [0]
    y_path = [0]
    environment = get_environment(l_lambda,u_lambda,2*N+1)
    for i in range(steps):
        if x <= -N-0.5 or x >= N+0.5 or y <= -N-0.5 or y >= N+0.5:
            break
        (x,y) = time_step(x,y,dt, environment, N)
        x_path.append(x)
        y_path.append(y)
    
    #make_plot(np.array(x_path),np.array(y_path), environment, N)
    make_animation(np.array(x_path), np.array(y_path), environment, N, dt = dt, save = save)


"For the random checkerboard"
#run_simulation(dt = 0.001, steps = 1000, N = 5, l_lambda = 5, u_lambda = 20, save = True)


"For the constant coefficient"
run_simulation(dt = 0.001, steps = 1000, N = 5, l_lambda = 10, u_lambda = 10, save = True)

"""
TO ADD:
    - possibly have time step size change with the ellipticity to make the path look smoother
    - run for long time and zoom out/do rescaling
"""
