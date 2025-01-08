import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the data
df = pd.read_csv('experiments/outputs/parm.csv', index_col=False)
datasets_metrics = df['datametric']

unique_datasets = datasets_metrics.unique()
num_plots = len(unique_datasets)

# Set the number of columns and rows in the subplot grid
cols = 4
rows = 3

fig = plt.figure(figsize=(24, 14))

for i, dataset_metric in enumerate(unique_datasets):
    # Create a 3D subplot
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

    # Filter the data
    ds_data = df[datasets_metrics == dataset_metric].reset_index(drop=True)

    # Pivot to create a matrix of NS indexed by (Lambda, Beta)
    pivoted = ds_data.pivot(index='Lambda', columns='Beta', values='NS')
    pivoted = pivoted.sort_index(axis=0).sort_index(axis=1)

    # Extract X, Y vectors and the Z matrix
    X_vals = pivoted.columns.values   # Beta
    Y_vals = pivoted.index.values    # Lambda
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = pivoted.values

    # Plot the 3D surface with a color gradient from blue to red
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.jet,  # Choose the blue-to-red gradient
        edgecolor='none',
        linewidth=0,
        antialiased=True
    )

    # Determine the position of the minimum
    min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    min_x = X[min_idx]
    min_y = Y[min_idx]
    min_z = Z[min_idx]

    # Place a red dot at the minimum
    ax.scatter(
        min_x, min_y, min_z,
        s=50, c='red', marker='o', depthshade=True
    )

    # Adjust the axes
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel(r'$\lambda$', fontsize=12)
    ax.set_zlabel('Non-Specificity', labelpad=2, fontsize=12, rotation=90)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='z', labelsize=9)

    # 3D view (azimuth, elevation)
    ax.view_init(elev=30, azim=45)

    # Subplot title
    ax.text2D(
        0.5, -0.12,
        f"({chr(97 + i)}) {dataset_metric}",
        transform=ax.transAxes,
        ha='center',
        fontsize=16
    )

    # Color bar with the new gradient
    m = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z)))
    m.set_array([])
    cbar = plt.colorbar(m, ax=ax, pad=0.04, fraction=0.05, format='%.2f', shrink=0.6)
    cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("surface_3D.pdf", format='pdf', dpi=300)
plt.savefig("surface_3D.png", format='png', dpi=300)
plt.show()
