"""
Graph Representation & Topological Defect Characterization for PECT-JEPA.

Transforms 3D volumetric defect voxels into a spatial adjacency network G = (V, E)
using NetworkX and Scipy, providing quantitative engineering NDT metrics:
  - Fatigue crack propagation length L_crack (mm) via geodesic graph diameter.
  - Volumetric metal loss V_loss (mm^3) via 3D component integration.
  - Defect morphology classification (linear crack vs dendritic SCC vs pitting).
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


Z_LAYER_DEPTHS = np.array([0.25, 0.85, 1.60, 2.50], dtype=np.float32)


def build_defect_graph(
    volume_3d: np.ndarray,             # [sY, sX, 4]
    threshold_percentile: float = 95.0,
    r_conn: float = 2.5,               # Max physical connection distance in mm
    min_component_size: int = 8,
    z_layer_depths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Constructs a 3D Defect Graph G = (V, E) from volumetric anomaly energy.
    Returns the graph, connected components, and node coordinate arrays.
    """
    sY, sX, n_depth = volume_3d.shape
    z_coords = z_layer_depths if z_layer_depths is not None else Z_LAYER_DEPTHS

    # 1. Active Defect Voxel Detection
    thresh = float(np.percentile(volume_3d, threshold_percentile))
    active_y, active_x, active_z_idx = np.where(volume_3d >= thresh)

    if len(active_y) == 0:
        thresh = float(np.percentile(volume_3d, 90.0))
        active_y, active_x, active_z_idx = np.where(volume_3d >= thresh)

    n_nodes = len(active_y)
    if n_nodes == 0:
        return {"error": "No active defect voxels detected", "num_nodes": 0}

    # If too many voxels (e.g. wide corrosion sheet), keep top 4000 strongest flaw voxels
    if n_nodes > 4000:
        top_k_idx = np.argsort(volume_3d[active_y, active_x, active_z_idx])[-4000:]
        active_y = active_y[top_k_idx]
        active_x = active_x[top_k_idx]
        active_z_idx = active_z_idx[top_k_idx]
        n_nodes = len(active_y)

    # 2. Node Coordinates in physical mm
    pts_x = active_x.astype(np.float32)
    pts_y = active_y.astype(np.float32)
    pts_z = z_coords[active_z_idx]
    energies = volume_3d[active_y, active_x, active_z_idx]
    points_3d = np.stack([pts_x, pts_y, pts_z], axis=1)  # [N, 3]

    # 3. Fast Graph Construction via Vectorized KD-Tree Query
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(
            i,
            x=float(pts_x[i]),
            y=float(pts_y[i]),
            z=float(pts_z[i]),
            energy=float(energies[i]),
            layer=int(active_z_idx[i]),
        )

    tree = cKDTree(points_3d)
    pairs = list(tree.query_pairs(r=r_conn))
    if len(pairs) > 0:
        pairs_arr = np.array(pairs, dtype=np.int64)
        diffs = points_3d[pairs_arr[:, 0]] - points_3d[pairs_arr[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)
        for (u, v), d in zip(pairs_arr, dists):
            G.add_edge(int(u), int(v), weight=float(d))

    # 4. Connected Components
    raw_components = [list(c) for c in nx.connected_components(G)]
    # Filter small noise clusters
    valid_components = [c for c in raw_components if len(c) >= min_component_size]
    valid_components.sort(key=len, reverse=True)

    return {
        "graph": G,
        "points_3d": points_3d,
        "energies": energies,
        "num_nodes": n_nodes,
        "num_edges": G.number_of_edges(),
        "threshold": thresh,
        "connected_components": valid_components,
        "num_valid_components": len(valid_components),
    }


def compute_crack_metrics(
    graph_data: Dict[str, Any],
    fastener_center: Optional[Tuple[float, float]] = None,
    fastener_radius: float = 3.5,
) -> Dict[str, Any]:
    """
    Calculates Quantitative Aerospace NDT Crack Metrics:
      - Physical crack propagation length L_crack (mm) via Dijkstra geodesic path.
      - Crack propagation orientation angle theta (degrees).
      - Depth penetration profile across layers.
      - Linearity and morphology classification.
    """
    G = graph_data.get("graph")
    components = graph_data.get("connected_components", [])
    if G is None or not components:
        return {"error": "No valid defect components in graph"}

    # Focus on primary (largest) defect component
    primary_comp = components[0]
    subG = G.subgraph(primary_comp).copy()

    nodes_in_comp = list(primary_comp)
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]] for n in nodes_in_comp])
    xy_coords = coords[:, :2]

    # Detect or use fastener center
    if fastener_center is None:
        c_x, c_y = float(np.mean(xy_coords[:, 0])), float(np.mean(xy_coords[:, 1]))
    else:
        c_x, c_y = fastener_center

    dist_from_center = np.linalg.norm(xy_coords - np.array([c_x, c_y]), axis=1)

    # Root node: closest to fastener hole boundary
    root_idx = int(np.argmin(dist_from_center))
    root_node = nodes_in_comp[root_idx]

    # Tip node: farthest from fastener center
    tip_idx = int(np.argmax(dist_from_center))
    tip_node = nodes_in_comp[tip_idx]

    # Geodesic Crack Length via Dijkstra Shortest Path
    try:
        path = nx.shortest_path(subG, source=root_node, target=tip_node, weight="weight")
        crack_length_mm = float(sum(subG[path[k]][path[k+1]]["weight"] for k in range(len(path) - 1)))
    except nx.NetworkXNoPath:
        # Fallback to straight-line Euclidean span if graph is weakly disconnected
        crack_length_mm = float(np.linalg.norm(coords[tip_idx] - coords[root_idx]))
        path = [root_node, tip_node]

    # Orientation Angle in degrees [-180, 180]
    dx = G.nodes[tip_node]["x"] - G.nodes[root_node]["x"]
    dy = G.nodes[tip_node]["y"] - G.nodes[root_node]["y"]
    theta_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Depth Penetration Profile (percentage of flaw voxels in each depth layer)
    layers = [G.nodes[n]["layer"] for n in nodes_in_comp]
    layer_counts = {k: int(np.sum(np.array(layers) == k)) for k in range(4)}
    max_depth_layer = int(max(layers))
    max_depth_mm = float(Z_LAYER_DEPTHS[max_depth_layer])

    # Morphology Linearity (Principal Component Analysis of node coordinates)
    cov = np.cov(coords, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    linearity = float(eigvals[0] / (eigvals[1] + eigvals[2] + 1e-6))
    planarity = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-6))

    if linearity > 3.0:
        morphology = "Linear Fatigue Crack"
    elif planarity > 0.4:
        morphology = "Planar Inter-layer Flaw"
    else:
        morphology = "Volumetric / Pitting Cluster"

    return {
        "crack_length_mm": round(crack_length_mm, 2),
        "orientation_deg": round(theta_deg, 1),
        "root_coord_mm": (round(float(G.nodes[root_node]["x"]), 2), round(float(G.nodes[root_node]["y"]), 2)),
        "tip_coord_mm": (round(float(G.nodes[tip_node]["x"]), 2), round(float(G.nodes[tip_node]["y"]), 2)),
        "max_depth_mm": round(max_depth_mm, 2),
        "layer_distribution": layer_counts,
        "morphology": morphology,
        "linearity_score": round(linearity, 2),
        "path_nodes": path,
        "total_flaw_voxels": len(nodes_in_comp),
    }


def compute_corrosion_volume(
    graph_data: Dict[str, Any],
    voxel_dx_mm: float = 1.0,
    voxel_dy_mm: float = 1.0,
    voxel_dz_mm: float = 0.75,
) -> Dict[str, Any]:
    """
    Computes volumetric metal loss and surface thinning metrics for corrosion scans.
    """
    G = graph_data.get("graph")
    components = graph_data.get("connected_components", [])
    if G is None or not components:
        return {"error": "No valid corrosion components in graph"}

    voxel_vol = voxel_dx_mm * voxel_dy_mm * voxel_dz_mm
    total_voxels = sum(len(c) for c in components)
    v_loss_mm3 = float(total_voxels * voxel_vol)

    all_nodes = [n for c in components for n in c]
    all_xy = set((int(G.nodes[n]["x"]), int(G.nodes[n]["y"])) for n in all_nodes)
    area_mm2 = float(len(all_xy) * voxel_dx_mm * voxel_dy_mm)

    max_depth_mm = float(max(G.nodes[n]["z"] for n in all_nodes))

    return {
        "volumetric_metal_loss_mm3": round(v_loss_mm3, 2),
        "surface_area_mm2": round(area_mm2, 2),
        "max_pit_depth_mm": round(max_depth_mm, 2),
        "num_corrosion_pits": len(components),
        "total_flaw_voxels": total_voxels,
    }


def plot_defect_graph_3d(
    graph_data: Dict[str, Any],
    crack_metrics: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Plots a publication-quality 3D Graph Network visualization:
      - Flaw nodes colored by physical depth Z (mm).
      - Adjacency edges showing topological connectivity.
      - Primary crack propagation path highlighted in bold red.
    """
    G = graph_data.get("graph")
    components = graph_data.get("connected_components", [])
    if G is None or not components:
        return ""

    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Vectorized Line3DCollection for fast rendering
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    segments = []
    for comp in components[:5]:
        subG = G.subgraph(comp)
        edges = list(subG.edges())
        if len(edges) > 3000:
            sub_idx = np.random.choice(len(edges), size=3000, replace=False)
            edges = [edges[i] for i in sub_idx]
        for u, v in edges:
            segments.append([
                [G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u]["z"]],
                [G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v]["z"]],
            ])
    if segments:
        lc = Line3DCollection(segments, colors="gray", alpha=0.35, linewidths=0.8)
        ax.add_collection3d(lc)

    # Plot nodes colored by depth
    comp_nodes = [n for c in components[:5] for n in c]
    if len(comp_nodes) > 5000:
        sub_n = np.random.choice(len(comp_nodes), size=5000, replace=False)
        comp_nodes = [comp_nodes[i] for i in sub_n]
    xs = [G.nodes[n]["x"] for n in comp_nodes]
    ys = [G.nodes[n]["y"] for n in comp_nodes]
    zs = [G.nodes[n]["z"] for n in comp_nodes]
    energies = [G.nodes[n]["energy"] for n in comp_nodes]

    p = ax.scatter(
        xs, ys, zs,
        c=zs,
        cmap="plasma",
        s=18,
        alpha=0.85,
        edgecolors="none",
        label="Flaw Voxels"
    )
    cb = fig.colorbar(p, ax=ax, fraction=0.035, pad=0.08, shrink=0.7)
    cb.set_label("Physical Depth Z (mm)", fontsize=10, fontweight="bold")

    # Highlight crack path if available
    if crack_metrics and "path_nodes" in crack_metrics:
        path = crack_metrics["path_nodes"]
        px = [G.nodes[n]["x"] for n in path]
        py = [G.nodes[n]["y"] for n in path]
        pz = [G.nodes[n]["z"] for n in path]
        ax.plot(px, py, pz, color="red", linewidth=3.5, label=f"Crack Path (L={crack_metrics.get('crack_length_mm', 0):.1f} mm)")
        ax.scatter([px[0]], [py[0]], [pz[0]], color="lime", s=70, marker="o", label="Crack Root (Fastener Edge)")
        ax.scatter([px[-1]], [py[-1]], [pz[-1]], color="yellow", s=70, marker="^", label="Crack Tip")

    ax.set_xlabel("X Scan (mm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y Scan (mm)", fontsize=10, fontweight="bold")
    ax.set_zlabel("Depth Z (mm)", fontsize=10, fontweight="bold")
    ax.set_zlim(3.0, 0.0)  # Inverted so 0.0 is top surface
    ax.set_title(title or "3D PECT-JEPA Topological Defect Graph", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=9)
    ax.view_init(elev=28, azim=-60)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return save_path or ""


def export_graph_interactive_html(
    graph_data: Dict[str, Any],
    crack_metrics: Optional[Dict[str, Any]] = None,
    save_path: str = "defect_graph_3d.html",
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Exports an interactive 3D Graph Network visualization using Plotly.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    G = graph_data.get("graph")
    components = graph_data.get("connected_components", [])
    if G is None or not components:
        return None

    edge_x, edge_y, edge_z = [], [], []
    for comp in components[:5]:
        subG = G.subgraph(comp)
        edges = list(subG.edges())
        if len(edges) > 3000:
            sub_idx = np.random.choice(len(edges), size=3000, replace=False)
            edges = [edges[i] for i in sub_idx]
        for u, v in edges:
            edge_x.extend([G.nodes[u]["x"], G.nodes[v]["x"], None])
            edge_y.extend([G.nodes[u]["y"], G.nodes[v]["y"], None])
            edge_z.extend([G.nodes[u]["z"], G.nodes[v]["z"], None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(120, 120, 120, 0.4)", width=1.5),
        hoverinfo="none",
        name="Topological Adjacency",
    )

    comp_nodes = [n for c in components[:5] for n in c]
    if len(comp_nodes) > 5000:
        sub_n = np.random.choice(len(comp_nodes), size=5000, replace=False)
        comp_nodes = [comp_nodes[i] for i in sub_n]
    node_x = [G.nodes[n]["x"] for n in comp_nodes]
    node_y = [G.nodes[n]["y"] for n in comp_nodes]
    node_z = [G.nodes[n]["z"] for n in comp_nodes]
    node_energy = [G.nodes[n]["energy"] for n in comp_nodes]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers",
        marker=dict(
            size=4.0,
            color=node_z,
            colorscale="Plasma",
            opacity=0.85,
            colorbar=dict(title="Depth Z (mm)", thickness=15, len=0.7),
        ),
        text=[f"X: {x:.1f}mm<br>Y: {y:.1f}mm<br>Depth Z: {z:.2f}mm<br>Energy: {e:.4f}"
              for x, y, z, e in zip(node_x, node_y, node_z, node_energy)],
        hoverinfo="text",
        name="Defect Voxels",
    )

    data = [edge_trace, node_trace]

    if crack_metrics and "path_nodes" in crack_metrics:
        path = crack_metrics["path_nodes"]
        px = [G.nodes[n]["x"] for n in path]
        py = [G.nodes[n]["y"] for n in path]
        pz = [G.nodes[n]["z"] for n in path]
        path_trace = go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines+markers",
            line=dict(color="red", width=5.0),
            marker=dict(size=5.0, color="red"),
            name=f"Crack Path (L={crack_metrics.get('crack_length_mm', 0):.1f}mm)",
        )
        data.append(path_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=title or "3D PECT-JEPA Defect Graph Network", font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="X Scan (mm)"),
            yaxis=dict(title="Y Scan (mm)"),
            zaxis=dict(title="Depth Z (mm)", range=[3.0, 0.0]),
            aspectratio=dict(x=1.0, y=1.0, z=0.4),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.write_html(save_path, include_plotlyjs="cdn")
    return save_path
