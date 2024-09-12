import streamlit as st
import plotly.graph_objects as go
from cfd_simulation import simulate_airflow

# Sidebar inputs for user settings
st.sidebar.title("Fan and AC Settings")

# Better defaults for fan settings
fan_cmm = st.sidebar.number_input("Fan CMM (Cubic Meters per Minute)", min_value=10.0, value=150.0, step=10.0)
fan_rpm = st.sidebar.number_input("Fan RPM (Rotations per Minute)", min_value=500, value=2500, step=100)
fan_diameter = st.sidebar.number_input("Fan Diameter (meters)", min_value=0.1, value=0.7, step=0.1)
fan_x = st.sidebar.number_input("Fan X Position (meters)", min_value=0.0, value=4.0, step=0.1)
fan_y = st.sidebar.number_input("Fan Y Position (meters)", min_value=0.0, value=2.0, step=0.1)
fan_z = st.sidebar.number_input("Fan Z Position (meters)", min_value=0.0, value=1.5, step=0.1)

# Better defaults for AC settings
ac_x = st.sidebar.number_input("AC X Position (meters)", min_value=0.0, value=0.0, step=0.1)
ac_y = st.sidebar.number_input("AC Y Position (meters)", min_value=0.0, value=1.5, step=0.1)
ac_z = st.sidebar.number_input("AC Z Position (meters)", min_value=0.0, value=1.0, step=0.1)

# Room dimensions (better default size)
room_length = st.sidebar.number_input("Room Length (meters)", min_value=1.0, value=8.0, step=1.0)
room_width = st.sidebar.number_input("Room Width (meters)", min_value=1.0, value=4.0, step=1.0)
room_height = st.sidebar.number_input("Room Height (meters)", min_value=1.0, value=3.0, step=1.0)

# Simulation time (keeping it short)
time_duration = st.sidebar.number_input("Simulation Time (seconds)", min_value=0.01, value=0.03, step=0.01)

# Button to generate plot
if st.sidebar.button("Generate 3D Plot"):
    # Run the CFD simulation with updated defaults
    X, Y, Z, U, V, W = simulate_airflow(
        fan_cmm=fan_cmm,
        fan_rpm=fan_rpm,
        fan_diameter=fan_diameter,
        fan_position=(fan_x, fan_y, fan_z),
        ac_position=(ac_x, ac_y, ac_z),
        room_dimensions=(room_length, room_width, room_height),
        time_duration=time_duration
    )

    # Create 3D interactive plot
    fig = go.Figure()

    # Add airflow visualization using quiver-style plot
    fig.add_trace(go.Cone(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),  # Coordinates
        u=U.ravel(), v=V.ravel(), w=W.ravel(),  # Velocities
        colorscale='Viridis',                   # Color scale based on velocity magnitude
        colorbar=dict(title="Velocity (m/s)"),  # Color bar label
        sizemode="absolute",                    # Keep the size of arrows consistent
        sizeref=0.1,                            # Scaling factor for arrow size
        showscale=True                          # Show the color scale
    ))

    # Add scatter points for AC and Fan positions
    fig.add_trace(go.Scatter3d(
        x=[ac_x], y=[ac_y], z=[ac_z],  # AC position
        mode='markers',
        marker=dict(size=5, color='red'),
        name="AC Inlet"
    ))

    fig.add_trace(go.Scatter3d(
        x=[fan_x], y=[fan_y], z=[fan_z],  # Fan position
        mode='markers',
        marker=dict(size=5, color='green'),
        name="Fan"
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (meters)"),
            yaxis=dict(title="Y (meters)"),
            zaxis=dict(title="Z (meters)")
        ),
        title="Interactive 3D Airflow with AC and Fan",
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
