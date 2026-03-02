### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f17103ea-06bf-11f1-a2b0-79e68ed152eb
md"""# Project_01 - Spinning Pendulum and the Lagrange equations

In this project, a pendulum is attached to a spinning frame. The frame has dimensions, 

$h_1 = 0.2~m$

$w_1 = 0.1~m$

and the pendulum length is $L=0.15~m$ with a $m=0.1~kg$ point mass at the end of the system. The pendulum swings in the $x'-z'-plane$ as it rotates at a constant speed, $\Omega$

![Spinning pendulum with rotating  and fixed coordinate systems.](https://raw.githubusercontent.com/cooperrc/me5180-project_01/refs/heads/main/spinning_pendulum.svg)

Your team's goal is to 

- build the equations of motion using Lagrange and least action $L=T-V$
- solve for the motion for a slow rotation speed and a fast rotation speed
- visualize the solution with plots and animations

"""

# ╔═╡ 0d9be664-d7c5-4084-add2-25e5418742d6
using Symbolics, Plots, DifferentialEquations, Latexify

begin
    @variables t Ω L g w1 h1 m
    @variables θ(t)
    D = Differential(t)

    θdot = D(θ)

    x_prime = w1 + L*sin(θ)
    z_prime = h1 - L*cos(θ)

    T = 0.5*m*(L^2*θdot^2 + Ω^2*x_prime^2)
    V = m*g*z_prime

    Lag = T - V
end

begin
    dL_dθ    = expand_derivatives(Symbolics.derivative(Lag, θ))
    dL_dθdot = expand_derivatives(Symbolics.derivative(Lag, θdot))
    EL_equation = expand_derivatives(D(dL_dθdot) - dL_dθ)
end

ode_symbolic = solve_for(EL_equation, D(D(θ)))

ode_symbolic_simplified = simplify(ode_symbolic)

begin
    θ_double_dot = simplify(build_function(ode_symbolic_simplified, θ, D(θ), g, L, Ω, w1; expression=Val(false)))
end

function pendulum_ode!(dstate, state, p, t)
    position_angle = state[1]
    velocity_angle = state[2]

    g, L, Ω, w1, h1 = p
    dstate[1] = velocity_angle
    dstate[2] = θ_double_dot(position_angle, velocity_angle, g, L, Ω, w1)
end

begin
    initial_state = [30*pi/180, 0.0]
    parameters_slow = (9.81, 0.15, 0.1, 0.1, 0.2)
    parameters_fast = (9.81, 0.15, 10.0, 0.1, 0.2)
    time_span = (0.0, 3.0)

    prob_slow = ODEProblem(pendulum_ode!, initial_state, time_span, parameters_slow)
    prob_fast = ODEProblem(pendulum_ode!, initial_state, time_span, parameters_fast)

    sol_slow = solve(prob_slow, reltol=1e-6)
    sol_fast = solve(prob_fast, reltol=1e-6)
end

function animate_pendulum(sol, parameters; number_frames=150, filename="pendulum.gif")
    gravity_value, length_pendulum, Omega, w1, h1 = parameters
    origin_pivot = (w1, h1)

    time_values = range(sol.t[1], sol.t[end], length=number_frames)

    anim = @animate for time_value in time_values
        current_state = sol(time_value)
        angle_current = current_state[1]
        speed_current = current_state[2]

        angle_plot = plot(sol.t, sol[1, :],
            xlabel="time (s)", ylabel="θ (rad)",
            title="Angle vs Time", legend=false)
        scatter!(angle_plot, [time_value], [angle_current])

        speed_plot = plot(sol.t, sol[2, :],
            xlabel="time (s)", ylabel="ω (rad/s)",
            title="Angular Speed vs Time", legend=false)
        scatter!(speed_plot, [time_value], [speed_current])

        bob_x = length_pendulum * sin(angle_current) + w1
        bob_y = h1 - length_pendulum * cos(angle_current)

        pendulum_plot = plot(
            xlim=(w1 - 1.2*length_pendulum, w1 + 1.2*length_pendulum),
            ylim=(h1 - 1.2*length_pendulum, h1 + 0.2*length_pendulum),
            aspect_ratio=:equal,
            xlabel="x' (m)", ylabel="z' (m)",
            title="Pendulum Motion", legend=false)

        plot!(pendulum_plot,
            [origin_pivot[1], bob_x],
            [origin_pivot[2], bob_y],
            lw=3)

        scatter!(pendulum_plot, [bob_x], [bob_y], ms=8)

        plot(angle_plot, speed_plot, pendulum_plot,
            layout=@layout([a; b; c]), size=(700, 900))
    end

    gif(anim, filename, fps=30)
end

animate_pendulum(sol_slow, parameters_slow, number_frames=150, filename="pendulum_slow.gif")

animate_pendulum(sol_fast, parameters_fast, number_frames=150, filename="pendulum_fast.gif")

md"""
## Fixed-Frame Trajectory Animation

This animation shows the pendulum bob path in the fixed \(x\)-\(y\) frame for both the slow and fast rotation cases. This gives another way to visualize how increasing the rotation speed changes the motion of the system.
"""

function animate_fixed_frame_trajectory(sol_slow, sol_fast, parameters_slow, parameters_fast; number_frames=150, filename="fixed_frame_trajectory.gif")
    g_slow, L_slow, Ωslow, w1_slow, h1_slow = parameters_slow
    g_fast, L_fast, Ωfast, w1_fast, h1_fast = parameters_fast

    time_slow = range(sol_slow.t[1], sol_slow.t[end], length=number_frames)
    time_fast = range(sol_fast.t[1], sol_fast.t[end], length=number_frames)

    anim = @animate for i in 1:number_frames
        current_time_slow = time_slow[i]
        current_time_fast = time_fast[i]

        state_slow = sol_slow(current_time_slow)
        state_fast = sol_fast(current_time_fast)

        θ_slow = state_slow[1]
        θ_fast = state_fast[1]

        x_slow = (w1_slow + L_slow*sin(θ_slow)) * cos(Ωslow*current_time_slow)
        y_slow = (w1_slow + L_slow*sin(θ_slow)) * sin(Ωslow*current_time_slow)

        x_fast = (w1_fast + L_fast*sin(θ_fast)) * cos(Ωfast*current_time_fast)
        y_fast = (w1_fast + L_fast*sin(θ_fast)) * sin(Ωfast*current_time_fast)

        slow_x_path = [(w1_slow + L_slow*sin(sol_slow(t)[1])) * cos(Ωslow*t) for t in time_slow[1:i]]
        slow_y_path = [(w1_slow + L_slow*sin(sol_slow(t)[1])) * sin(Ωslow*t) for t in time_slow[1:i]]

        fast_x_path = [(w1_fast + L_fast*sin(sol_fast(t)[1])) * cos(Ωfast*t) for t in time_fast[1:i]]
        fast_y_path = [(w1_fast + L_fast*sin(sol_fast(t)[1])) * sin(Ωfast*t) for t in time_fast[1:i]]

        p1 = plot(
            slow_x_path, slow_y_path,
            xlabel="x (m)", ylabel="y (m)",
            title="Slow Rotation: Fixed-Frame Path",
            legend=false,
            aspect_ratio=:equal,
            xlim=(-0.3, 0.3), ylim=(-0.3, 0.3)
        )
        scatter!(p1, [x_slow], [y_slow], ms=7)

        p2 = plot(
            fast_x_path, fast_y_path,
            xlabel="x (m)", ylabel="y (m)",
            title="Fast Rotation: Fixed-Frame Path",
            legend=false,
            aspect_ratio=:equal,
            xlim=(-0.3, 0.3), ylim=(-0.3, 0.3)
        )
        scatter!(p2, [x_fast], [y_fast], ms=7)

        plot(p1, p2, layout=(1,2), size=(900,400))
    end

    gif(anim, filename, fps=30)
end

animate_fixed_frame_trajectory(sol_slow, sol_fast, parameters_slow, parameters_fast, number_frames=150, filename="fixed_frame_trajectory.gif")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

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
manifest_format = "2.0"
project_hash = "71853c6197a6a7f222db0f1978c7cb232b87c5ee"

[deps]
"""

# ╔═╡ Cell order:
# ╟─f17103ea-06bf-11f1-a2b0-79e68ed152eb
# ╠═0d9be664-d7c5-4084-add2-25e5418742d6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
