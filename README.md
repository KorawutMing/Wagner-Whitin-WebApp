# Inventory Optimization Web Application

This web application is designed to solve an inventory optimization problem using dynamic programming. The goal is to minimize costs associated with ordering and holding inventory over time while meeting the demand. The application allows users to input data about demand, ordering costs, unit costs, and holding costs, and it computes the optimal ordering plan, along with relevant visualizations.

## Features

- **Inventory Optimization Calculation**: Computes the optimal ordering strategy based on demand, ordering costs, unit costs, and holding costs using dynamic programming.
- **Visualizations**:
  - **Inventory Level Tracking**: Displays a line graph showing the inventory level over time, with indications of when orders are placed.
  - **Monthly Demand and Cumulative Costs**: A bar chart showing the monthly demand, with vertical lines marking the ordering points, and a cumulative cost line plot.
  - **Network Graph Visualization**: A directed network graph that visualizes the decision process, including optimal paths and edges with associated costs.

## How to Run

### Prerequisites

- Python 3.x
- Flask
- Matplotlib
- NetworkX
- Numpy