using Markdown
using InteractiveUtils

# ╔═╡ f17103ea-06bf-11f1-a2b0-79e68ed152eb
md"""
# Project_01 - Spinning Pendulum and the Lagrange equations

In this project, a pendulum is attached to a spinning frame. The frame has dimensions,

$h_1 = 0.2~m$

$w_1 = 0.1~m$

and the pendulum length is $L = 0.15~m$ with a $m = 0.1~kg$ point mass at the end of the system.  
The pendulum swings in the $x'-z'$ plane as it rotates at a constant speed, $\Omega$.

![Spinning pendulum with rotating and fixed coordinate systems.](https://raw.githubusercontent.com/cooperrc/me5180-project_01/refs/heads/main/spinning_pendulum.svg)

## Goals
- Build the equations of motion using Lagrange's equation with $L = T - V$
- Solve for the motion for a slow rotation speed and a fast rotation speed
- Visualize the solution with plots and animations
"""

# ╔═╡ 0d9be664-d7c5-4084-add2-25e5418742d6
begin
    using Symbolics
    using Plots
    using DifferentialEquations
    using Latexify

    # -----------------------------
    # Symbolic setup
    # -----------------------------
    @variables t Ω L g w1 h1 m
    @variables θ(t)

    D = Differential(t)
    θdot = D(θ)

    x_prime = w1 + L * sin(θ)
    z_prime = h1 - L * cos(θ)

    T = 0.5 * m * (L^2 * θdot^2 + Ω^2 * x_prime^2)
    V = m * g * z_prime
    Lag = T - V

    dL_dθ = expand_derivatives(Symbolics.derivative(Lag, θ))
    dL_dθdot = expand_derivatives(Symbolics.derivative(Lag, θdot))
    EL_equation = expand_derivatives(D(dL_dθdot) - dL_dθ)

    θdd_expr = simplify(first(solve_for([EL_equation], D(D(θ)))))
    θ_double_dot = eval(build_function(θdd_expr, θ, D(θ), g, L, Ω, w1))

    # -----------------------------
    # Numerical ODE function
    # -----------------------------
    function pendulum_ode!(dstate, state, p, t)
        angle = state[1]
        angular_velocity = state[2]

        g_val, L_val, Ω_val, w1_val, h1_val = p

        dstate[1] = angular_velocity
        dstate[2] = θ_double_dot(angle, angular_velocity, g_val, L_val, Ω_val, w1_val)
    end

    # -----------------------------
    # Parameters and simulation
    # -----------------------------
    initial_state = [30 * pi / 180, 0.0]
    time_span = (0.0, 3.0)

    parameters_slow = (9.81, 0.15, 0.1, 0.1, 0.2)
    parameters_fast = (9.81, 0.15, 10.0, 0.1, 0.2)

    prob_slow = ODEProblem(pendulum_ode!, initial_state, time_span, parameters_slow)
    prob_fast = ODEProblem(pendulum_ode!, initial_state, time_span, parameters_fast)

    sol_slow = solve(prob_slow, Tsit5(), reltol=1e-6, abstol=1e-6)
    sol_fast = solve(prob_fast, Tsit5(), reltol=1e-6, abstol=1e-6)

    # -----------------------------
    # Static plots
    # -----------------------------
    p1 = plot(sol_slow.t, sol_slow[1, :],
        xlabel = "time (s)",
        ylabel = "θ (rad)",
        title = "Slow Rotation: Angle vs Time",
        legend = false
    )

    p2 = plot(sol_slow.t, sol_slow[2, :],
        xlabel = "time (s)",
        ylabel = "ω (rad/s)",
        title = "Slow Rotation: Angular Speed vs Time",
        legend = false
    )

    p3 = plot(sol_fast.t, sol_fast[1, :],
        xlabel = "time (s)",
        ylabel = "θ (rad)",
        title = "Fast Rotation: Angle vs Time",
        legend = false
    )

    p4 = plot(sol_fast.t, sol_fast[2, :],
        xlabel = "time (s)",
        ylabel = "ω (rad/s)",
        title = "Fast Rotation: Angular Speed vs Time",
        legend = false
    )

    plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 700))
end

# ╔═╡ a4d6c8be-06bf-11f1-1c2d-6f4fbb4d0001
begin
    function animate_pendulum(sol, parameters; number_frames = 150, filename = "pendulum.gif")
        g_val, L_val, Ω_val, w1_val, h1_val = parameters
        pivot = (w1_val, h1_val)

        time_values = range(sol.t[1], sol.t[end], length = number_frames)

        anim = @animate for current_time in time_values
            current_state = sol(current_time)
            current_angle = current_state[1]
            current_speed = current_state[2]

            angle_plot = plot(sol.t, sol[1, :],
                xlabel = "time (s)",
                ylabel = "θ (rad)",
                title = "Angle vs Time",
                legend = false
            )
            scatter!(angle_plot, [current_time], [current_angle])

            speed_plot = plot(sol.t, sol[2, :],
                xlabel = "time (s)",
                ylabel = "ω (rad/s)",
                title = "Angular Speed vs Time",
                legend = false
            )
            scatter!(speed_plot, [current_time], [current_speed])

            bob_x = w1_val + L_val * sin(current_angle)
            bob_z = h1_val - L_val * cos(current_angle)

            pendulum_plot = plot(
                xlim = (w1_val - 1.2 * L_val, w1_val + 1.2 * L_val),
                ylim = (h1_val - 1.2 * L_val, h1_val + 0.2 * L_val),
                xlabel = "x' (m)",
                ylabel = "z' (m)",
                title = "Pendulum Motion",
                legend = false,
                aspect_ratio = :equal
            )

            plot!(pendulum_plot, [pivot[1], bob_x], [pivot[2], bob_z], lw = 3)
            scatter!(pendulum_plot, [bob_x], [bob_z], ms = 8)

            plot(angle_plot, speed_plot, pendulum_plot,
                layout = @layout([a; b; c]),
                size = (700, 900)
            )
        end

        gif(anim, filename, fps = 30)
    end

    slow_gif = animate_pendulum(sol_slow, parameters_slow, filename = "pendulum_slow.gif")
    fast_gif = animate_pendulum(sol_fast, parameters_fast, filename = "pendulum_fast.gif")

    md"""
    ## Pendulum Motion Animations
    The slow and fast rotation cases were animated below.
    """
end

# ╔═╡ b7e8f9ca-06bf-11f1-2d3e-7a5fcc5e0002
begin
    function animate_fixed_frame_trajectory(sol_slow, sol_fast, parameters_slow, parameters_fast;
        number_frames = 150, filename = "fixed_frame_trajectory.gif")

        g_slow, L_slow, Ω_slow, w1_slow, h1_slow = parameters_slow
        g_fast, L_fast, Ω_fast, w1_fast, h1_fast = parameters_fast

        time_slow = range(sol_slow.t[1], sol_slow.t[end], length = number_frames)
        time_fast = range(sol_fast.t[1], sol_fast.t[end], length = number_frames)

        anim = @animate for i in 1:number_frames
            t_slow = time_slow[i]
            t_fast = time_fast[i]

            state_slow = sol_slow(t_slow)
            state_fast = sol_fast(t_fast)

            θ_slow = state_slow[1]
            θ_fast = state_fast[1]

            x_slow = (w1_slow + L_slow * sin(θ_slow)) * cos(Ω_slow * t_slow)
            y_slow = (w1_slow + L_slow * sin(θ_slow)) * sin(Ω_slow * t_slow)

            x_fast = (w1_fast + L_fast * sin(θ_fast)) * cos(Ω_fast * t_fast)
            y_fast = (w1_fast + L_fast * sin(θ_fast)) * sin(Ω_fast * t_fast)

            slow_x_path = [(w1_slow + L_slow * sin(sol_slow(t)[1])) * cos(Ω_slow * t) for t in time_slow[1:i]]
            slow_y_path = [(w1_slow + L_slow * sin(sol_slow(t)[1])) * sin(Ω_slow * t) for t in time_slow[1:i]]

            fast_x_path = [(w1_fast + L_fast * sin(sol_fast(t)[1])) * cos(Ω_fast * t) for t in time_fast[1:i]]
            fast_y_path = [(w1_fast + L_fast * sin(sol_fast(t)[1])) * sin(Ω_fast * t) for t in time_fast[1:i]]

            pslow = plot(
                slow_x_path, slow_y_path,
                xlabel = "x (m)",
                ylabel = "y (m)",
                title = "Slow Rotation: Fixed-Frame Path",
                legend = false,
                aspect_ratio = :equal,
                xlim = (-0.3, 0.3),
                ylim = (-0.3, 0.3)
            )
            scatter!(pslow, [x_slow], [y_slow], ms = 7)

            pfast = plot(
                fast_x_path, fast_y_path,
                xlabel = "x (m)",
                ylabel = "y (m)",
                title = "Fast Rotation: Fixed-Frame Path",
                legend = false,
                aspect_ratio = :equal,
                xlim = (-0.3, 0.3),
                ylim = (-0.3, 0.3)
            )
            scatter!(pfast, [x_fast], [y_fast], ms = 7)

            plot(pslow, pfast, layout = (1, 2), size = (900, 400))
        end

        gif(anim, filename, fps = 30)
    end

    fixed_frame_gif = animate_fixed_frame_trajectory(
        sol_slow, sol_fast, parameters_slow, parameters_fast,
        filename = "fixed_frame_trajectory.gif"
    )

    md"""
    ## Fixed-Frame Trajectory Animation
    This animation shows the pendulum bob path in the fixed $x$-$y$ frame for both the slow and fast rotation cases.
    """
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
DifferentialEquations = "~7.17.0"
Latexify = "~0.16.10"
Plots = "~1.41.6"
Symbolics = "~6.58.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╟─f17103ea-06bf-11f1-a2b0-79e68ed152eb
# ╠═0d9be664-d7c5-4084-add2-25e5418742d6
# ╠═a4d6c8be-06bf-11f1-1c2d-6f4fbb4d0001
# ╠═b7e8f9ca-06bf-11f1-2d3e-7a5fcc5e0002
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
