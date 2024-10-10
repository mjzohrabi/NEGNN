import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define the data
data = {
    "Cora": {
        "alpha_27": [(1, 82.5), (2, 79.4), (3, 81.7), (4, 78.2), (5, 82.3), (6, 81.2), (7, 81)],
        "beta_1": [(24, 80.2), (25, 82.5), (26, 77.3), (27, 82.5), (28, 79.5), (29, 80.8), (30, 79.7)],
    },
    "Citeseer": {
        "alpha_10": [(1, 68.1), (2, 69.4), (3, 68.2), (4, 67.8), (5, 68.6), (6, 66.8), (7, 67.8)],
        "beta_2": [(7, 69.3), (8, 69.3), (9, 68.9), (10, 69.4), (11, 69.2), (12, 70.8), (13, 67)],
    },
    "Pubmed": {
        "alpha_100": [(1, 78.2), (2, 77.8), (3, 78.3), (4, 77.6), (5, 78.4), (6, 77.7), (7, 78.1)],
    },
}

# Create a subplot figure
fig = make_subplots(rows=5, cols=2, vertical_spacing=0.1, horizontal_spacing=0.05)

# Plot each subplot
datasets = list(data.keys())
params = {
    "Cora": ["alpha_27", "beta_1"],
    "Citeseer": ["alpha_10", "beta_2"],
    "Pubmed": ["alpha_100"]
}

row_idx = 1
for i, dataset in enumerate(datasets):
    for j, param in enumerate(params[dataset]):
        coords = data[dataset][param]
        x_vals, y_vals = zip(*coords)
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, mode='markers+lines', name=f'{param}',
                marker=dict(color='black'), line=dict(color='gray', dash='dash')
            ),
            row=row_idx, col=j+1 if dataset != "Pubmed" else 1
        )
        fig.update_xaxes(title_text="X-axis", row=row_idx, col=j+1 if dataset != "Pubmed" else 1)
        fig.update_yaxes(title_text="Y-axis", row=row_idx, col=j+1 if dataset != "Pubmed" else 1)
        fig.update_yaxes(showticklabels=False, row=row_idx, col=j+1)
        fig.update_layout(
            legend=dict(
                title="Legend",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    row_idx += 1 if dataset != "Pubmed" else 0.5

# Show the plot
fig.show()
