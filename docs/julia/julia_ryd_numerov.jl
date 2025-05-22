using PyCall

ryd_numerov = pyimport("ryd_numerov");

state = ryd_numerov.RydbergState("Rb", n = 50, l = 0, j = 0.5)
println(state)

state.create_wavefunction()
println(state.wavefunction.w_list[1:10])
println(state.wavefunction.grid.z_list[1:10])

using Plots

p = plot(
    state.wavefunction.grid.z_list,
    state.wavefunction.w_list,
    xlabel = "z",
    ylabel = "w(z)",
    label = state.get_label("ket"),
)

display(p)
