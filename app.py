from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io
import base64
from matplotlib.figure import Figure
import networkx as nx
import os
from flask import send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path),
                               'icon.png', mimetype='image/vnd.microsoft.icon')

@app.route('/compute', methods=['POST'])
def compute():
    try:
        # Get input data from the request
        data = request.json
        
        D = data['D']   # Demand
        K = data['K']   # Ordering cost
        c = data['c']   # Unit cost
        h = data['h']   # Holding cost

        n = len(D)
        
        # Initialize dynamic programming array
        F = np.full(n + 1, float('inf'))
        F[0] = 0
        decisions = [0] * (n + 1)

        # Forward DP calculation of minimum cost
        for t in range(1, n + 1):
            for j in range(0, t):
                # Calculate holding cost
                holding_cost = 0
                for m in range(j + 1, t):
                    future_demand = sum(D[m:t])
                    holding_cost += h[m-1] * future_demand

                unit_cost = sum(D[j:t]) * c[j]
                total_cost = F[j] + K[j] + unit_cost + holding_cost
                
                if total_cost < F[t]:
                    F[t] = total_cost
                    decisions[t] = j
        
        # Compute the optimal order plan
        order_plan = []
        order_periods = []
        order_quantities = []
        
        t = n
        while t > 0:
            j = decisions[t]
            order_period = j + 1
            order_quantity = sum(D[j:t])
            
            order_plan.insert(0, (order_period, t, order_quantity))
            order_periods.insert(0, j)
            order_quantities.insert(0, order_quantity)
            
            t = j

        # Calculate inventory levels for each month
        inventory_levels = [0] * n

        for period, end_period, quantity in order_plan:
            period_idx = period - 1  # Convert to 0-indexed
            remaining = quantity
            
            for m in range(period_idx, end_period):
                if m > period_idx:
                    inventory_levels[m] = remaining
                remaining -= D[m]

        # Create visualization
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Line graph showing inventory levels
        plt.subplot(2, 1, 1)
        months = list(range(1, n+1))

        # Create points for inventory levels with jumps at order points
        x_values = []
        y_values = []

        for m in range(n):
            # Add point for beginning of month inventory
            x_values.append(m + 1)
            y_values.append(inventory_levels[m])
            
            # Add point for immediate after ordering (if this is an order month)
            if m in order_periods:
                idx = order_periods.index(m)
                x_values.append(m + 1 + 0.01)  # Small offset to show the jump
                y_values.append(inventory_levels[m] + order_quantities[idx])
            
            # Add point for end of month inventory
            x_values.append(m + 1.99)  # Just before next month
            if m < n-1:
                y_values.append(inventory_levels[m+1])
            else:
                y_values.append(0)  # End with zero inventory

        plt.plot(x_values, y_values, 'b-', linewidth=2, label='Inventory Level')
        plt.scatter(months, inventory_levels, color='blue', s=50, zorder=3)

        # Show points with orders
        for period, quantity in zip(order_periods, order_quantities):
            period_month = period + 1  # Convert to 1-indexed
            plt.annotate(f'Order {quantity}', 
                        xy=(period_month, inventory_levels[period] + order_quantities[order_periods.index(period)]),
                        xytext=(period_month, (inventory_levels[period] + order_quantities[order_periods.index(period)] + max(D) // 4)),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=9,
                        fontweight='bold',
                        ha='center')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (Start of Month)')
        plt.ylabel('Inventory Level')
        plt.title('Inventory Level Tracking')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()

        # 2. Bar graph showing monthly demand
        plt.subplot(2, 1, 2)
        bars = plt.bar(months, D, color='green', alpha=0.7, label='Monthly Demand')

        # Show demand values on the bars
        for bar, demand in zip(bars, D):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(demand), ha='center', va='bottom')

        # Show ordering with vertical lines
        for period in order_periods:
            period_month = period + 1  # Convert to 1-indexed
            plt.axvline(x=period_month, color='red', linestyle='--', alpha=0.7)
            idx = order_periods.index(period)
            plt.annotate(f'Order {order_quantities[idx]}', 
                        xy=(period_month, D[period] + max(D) // 10),
                        ha='center',
                        fontsize=9,
                        color='red')

        # Also plot the cumulative cost by period
        ax2 = plt.twinx()
        cumulative_costs = [F[i] for i in range(1, n+1)]
        ax2.plot(months, cumulative_costs, 'ro-', label='Cumulative Min Cost')
        ax2.set_ylabel('Cumulative Cost')
        ax2.legend(loc='upper right')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (months)')
        plt.ylabel('Demand (units)')
        plt.title('Monthly Demand, Order Points and Cumulative Costs')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='upper left')

        plt.tight_layout()
        
        # Convert plot to base64 string to embed in HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Create optimal path for network graph visualization
        optimal_path = [0]  # Start at node 0
        optimal_edges = []
        t = n
        while t > 0:
            j = decisions[t]
            optimal_path.insert(1, t)  # Insert at beginning to maintain order
            optimal_edges.append((j, t))
            t = j

        # Create a directed graph
        G = nx.DiGraph()

        # Adjust node positions with better spacing
        pos = {i: (i, -0.1 * i) for i in range(n + 1)}

        # Add nodes
        for i in range(n + 1):
            G.add_node(i, pos=pos[i])

        # Add all possible edges
        for t in range(1, n + 1):
            for j in range(0, t):
                holding_cost = sum(h[m-1] * sum(D[m:t]) for m in range(j + 1, t))
                edge_cost = K[j] + holding_cost
                G.add_edge(j, t, weight=edge_cost)

        # Create figure for network visualization
        fig2, ax = plt.subplots(figsize=(15, 9))
        ax.set_title('Wagner-Whitin Network Graph with Optimal Path Highlighted', fontsize=16)

        # Draw non-optimal edges first with low alpha
        non_optimal_edges = [e for e in G.edges() if e not in optimal_edges]
        nx.draw_networkx_edges(G, pos, edgelist=non_optimal_edges, width=0.3, alpha=0.15, edge_color='gray', connectionstyle='arc3,rad=0.15')

        # Draw nodes with different colors
        node_colors = ['green' if node == 0 else 'red' if node == n else 'gold' if node in optimal_path else 'lightgray' for node in G.nodes()]
        node_sizes = [3000 if node in [0, n] else 2000 if node in optimal_path else 1000 for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

        # Draw optimal edges with bold red lines and custom arcs
        for u, v in optimal_edges:
            rad = 0.25 if abs(u - v) > 1 else 0.1
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3.5, edge_color='red', connectionstyle=f'arc3,rad={rad}', arrowsize=20)

        # Node Labels
        node_labels = {i: f"Period {i}\nD={D[i-1]}\nCost={F[i]:.1f}" if i > 0 else "Start\nPeriod 0" for i in range(n+1)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

        # Adjust legend positioning inside the figure
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=3, label='Optimal Path'),
            plt.Line2D([0], [0], color='gray', lw=0.5, alpha=0.3, label='Other Possible Paths'),
            plt.scatter([], [], s=200, color='green', label='Start Node'),
            plt.scatter([], [], s=200, color='red', label='End Node'),
            plt.scatter([], [], s=200, color='gold', label='Order Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

        plt.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)  # Fine-tune layout

        # Convert network graph to base64
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        network_plot_data = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        return jsonify({
            "min_cost": F[n], 
            "order_plan": order_plan,
            "plot": plot_data,
            "network_plot": network_plot_data
        })
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)