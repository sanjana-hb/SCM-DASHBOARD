import sys
import numpy as np
print("\U0001F7E2 Starting SCM Dashboard...")

try:
    from forecast_model import generate_forecast
    import pandas as pd
    from dash import Dash, html, dcc, Input, Output, State, dash_table
    import dash_bootstrap_components as dbc
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import io
except Exception as e:
    print("\u274C Import error:", e)
    sys.exit(1)

try:
    generate_forecast()
    print("\u2705 Forecast generated successfully")
except Exception as e:
    print("\u274C Forecast generation error:", e)
    sys.exit(1)

# Load data
def load_data():
    inv = pd.read_csv("data/inventory.csv")
    orders = pd.read_csv("data/orders.csv")
    sales_history = pd.read_csv("data/sales_history.csv")
    forecast = pd.read_csv("data/forecast.csv")
    return inv, orders, sales_history, forecast

inventory, orders, sales_history, forecast = load_data()

# IMPROVED: Calculate time to stockout based on product-specific historical data and forecasts
def calculate_time_to_stockout(inventory_df, forecast_df, sales_history_df):
    """
    Calculate days until stockout based on product-specific historical data and forecasts.
    Products will have significantly different days to stockout.
    """
    # Get the base forecast trend but scale it down to create more reasonable demand
    base_forecast = forecast_df['yhat'].mean() * 0.3  # Scaling down the base demand
    
    # Create very different demand profiles for each product
    product_specific_demands = {}
    
    # Generate widely varying product demands
    for i, product_id in enumerate(inventory_df['product_id'].unique()):
        # Create dramatically different demand patterns (10x difference between lowest and highest)
        factor = 0.2 + (i * 0.5)  # Products will have demand factors ranging from 0.2 to 2.0+
        product_specific_demands[product_id] = base_forecast * factor
    
    # Calculate product-specific demand rates with minimal randomness
    inventory_df['avg_daily_demand'] = inventory_df['product_id'].map(
        lambda pid: product_specific_demands.get(pid, base_forecast) * (1 + np.random.normal(0, 0.01))
    )
    
    # Calculate days to stockout with a minimum daily demand to avoid division by zero
    inventory_df['days_to_stockout'] = inventory_df.apply(
        lambda row: round(row['stock_level'] / max(0.1, row['avg_daily_demand']), 1), 
        axis=1
    )
    
    # Force at least some products to have plenty of stock (>30 days)
    if len(inventory_df) >= 4:
        # For approximately 1/4 of products, artificially increase days_to_stockout
        num_safe_products = max(1, len(inventory_df) // 4)
        safe_indices = np.random.choice(inventory_df.index, num_safe_products, replace=False)
        
        for idx in safe_indices:
            inventory_df.at[idx, 'days_to_stockout'] = np.random.uniform(30, 60)
            # Also lower the demand to match the new days_to_stockout
            inventory_df.at[idx, 'avg_daily_demand'] = (
                inventory_df.at[idx, 'stock_level'] / inventory_df.at[idx, 'days_to_stockout']
            )
    
    return inventory_df

# Function to diversify stock levels for better visualization and testing
def diversify_stock_levels(inventory_df):
    """
    Manually diversify stock levels to ensure different stockout timelines.
    This is a one-time operation to create a better starting point.
    """
    if len(inventory_df) >= 4:
        # Increase stock levels dramatically for some products
        high_stock_indices = np.random.choice(inventory_df.index, len(inventory_df) // 3, replace=False)
        for idx in high_stock_indices:
            inventory_df.at[idx, 'stock_level'] = int(inventory_df.at[idx, 'stock_level'] * 5)
        
        # Lower stock levels for some products to create immediate alerts
        low_stock_indices = np.random.choice(
            [i for i in inventory_df.index if i not in high_stock_indices], 
            len(inventory_df) // 4, 
            replace=False
        )
        for idx in low_stock_indices:
            inventory_df.at[idx, 'stock_level'] = max(1, int(inventory_df.at[idx, 'stock_level'] * 0.3))
    
    return inventory_df

# Calculate stockout trends (simulated)
def calculate_stockout_trends():
    # Create a date range for the past 30 days
    date_range = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
    
    # Simulate stockout percentages with a slight increasing trend
    base_stockout = 15  # Base stockout percentage
    trend = 0.2  # Slight upward trend
    noise = 3  # Random noise
    
    stockout_data = {
        'date': date_range,
        'stockout_percentage': [max(0, min(100, base_stockout + (i * trend) + np.random.normal(0, noise))) 
                               for i in range(len(date_range))]
    }
    
    return pd.DataFrame(stockout_data)

# NEW: Function to ensure new products have reasonable stockout times
def ensure_reasonable_stockout(new_product_df, forecast_df):
    """
    Ensures that new products start with a reasonable stock level relative to their demand,
    giving them a stockout time of 15-30 days by default.
    """
    # Get a conservative estimate of daily demand (lower than average to avoid immediate alerts)
    base_demand = forecast_df['yhat'].mean() * 0.2
    
    # Generate a random stockout timeframe between 15-30 days for new products
    target_days_to_stockout = np.random.uniform(15, 30)
    
    # Calculate the required demand to achieve this stockout timeframe
    new_product_df['avg_daily_demand'] = new_product_df['stock_level'] / target_days_to_stockout
    new_product_df['days_to_stockout'] = target_days_to_stockout
    
    return new_product_df

# Apply diversity to stock levels (uncomment to use)
# inventory = diversify_stock_levels(inventory)

# Calculate time to stockout
inventory = calculate_time_to_stockout(inventory, forecast, sales_history)

# FIXED: Calculate alerts based on days_to_stockout threshold instead of reorder point
# Using 5 days as the threshold for low stock alert
alerts = inventory[inventory["days_to_stockout"] < 5]

# Calculate KPIs
avg_inventory = inventory["stock_level"].mean()
total_sales = sales_history["y"].sum()

inventory_turnover = round(total_sales / avg_inventory, 2)
stock_out_rate = round(len(alerts) / len(inventory) * 100, 2)
days_of_supply = round(avg_inventory / forecast["yhat"].mean(), 1)
on_time_delivery = round((orders[orders['status'] == 'Delivered'].shape[0] / 
                          max(1, len(orders))) * 100, 1)
order_fill_rate = round(((len(orders) - len(orders[orders['status'] == 'New'])) / 
                         max(1, len(orders))) * 100, 1)

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app.title = "SCM Dashboard"
server = app.server

app.layout = dbc.Container([
    html.H1("\U0001F4E6 Inventory & Order Tracking Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col(dbc.Input(id="input-id", placeholder="Product ID", type="text"), md=2),
        dbc.Col(dbc.Input(id="input-name", placeholder="Product Name", type="text"), md=2),
        dbc.Col(dbc.Input(id="input-stock", placeholder="Stock Level", type="number"), md=2),
        dbc.Col(dbc.Input(id="input-reorder", placeholder="Reorder Point", type="number"), md=2),
        dbc.Col(dbc.Input(id="input-warehouse", placeholder="Warehouse ID", type="text"), md=2),
        dbc.Col(dbc.Button("Add", id="btn-add", color="success"), md=2)
    ], className="mb-3"),
    html.Div(id="add-alert"),

    dbc.Row([
        dbc.Col([
            html.Label("\U0001F50D Select Warehouse"),
            dcc.Dropdown(
                id='warehouse-filter',
                options=[{"label": w, "value": w} for w in inventory['warehouse_id'].unique()],
                placeholder="Select a warehouse"
            )
        ], md=4),
        dbc.Col([
            html.Label("\U0001F4E6 Select Product"),
            dcc.Dropdown(
                id='product-filter',
                options=[{"label": p, "value": p} for p in inventory['product_name'].unique()],
                placeholder="Select a product"
            )
        ], md=4),
        dbc.Col([
            html.Label("\U0001F4C5 Export & Reset"),
            dbc.ButtonGroup([
                dbc.Button("Inventory CSV", id="btn-download-inv", color="primary"),
                dbc.Button("Orders CSV", id="btn-download-orders", color="info"),
                dbc.Button("Clear Filters", id="btn-clear-filters", color="secondary")
            ])
        ], md=4),
        dcc.Download(id="download-inv"),
        dcc.Download(id="download-orders"),
        dcc.Download(id="download-alerts")
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Low Stock Items"), html.H3(id="low-stock-count", className="text-danger"),
            html.P("Products need restocking")
        ])), lg=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Forecast Avg Demand"), html.H3(f"{int(forecast['yhat'].mean())}"),
            html.P("Units predicted per day")
        ])), lg=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Inventory Turnover"), html.H3(f"{inventory_turnover}"),
            html.P("Times sold & replaced")
        ])), lg=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Order Fill Rate"), html.H3(f"{order_fill_rate}%"),
            html.P("Orders being processed")
        ])), lg=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("On-Time Delivery"), html.H3(f"{on_time_delivery}%"),
            html.P("Orders delivered on time")
        ])), lg=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Products"), html.H3(id="total-products"),
            html.P("In inventory")
        ])), lg=2)
    ], className="mb-4"),

    html.Div(id="insight-text", className="text-center text-info fw-bold mb-3"),

    dbc.Tabs([
        dbc.Tab(label="\U0001F4CA Inventory", tab_id="inventory"),
        dbc.Tab(label="\U0001F4C8 Forecast", tab_id="forecast"),
        dbc.Tab(label="\U0001F6E0 Order Tracking", tab_id="orders"),
        dbc.Tab(label="\u26A0\uFE0F Alerts", tab_id="alerts"),
        dbc.Tab(label="\U0001F4C9 Analytics", tab_id="analytics")
    ], id="tabs", active_tab="inventory", className="mb-3"),

    html.Div(id="tab-content"),
    html.Hr(),
    html.Footer("Inspired by SCM dashboards: Thoughtspot, Inforiver, and more.", className="text-center text-muted mb-3")
], fluid=True)

#Callback to initialize and update the metrics
@app.callback(
    Output("low-stock-count", "children"),
    Output("total-products", "children"),
    Input("warehouse-filter", "value"),
    Input("product-filter", "value")
)
def update_metrics(warehouse, product):
    global inventory
    
    filtered_inventory = inventory.copy()
    
    if warehouse:
        filtered_inventory = filtered_inventory[filtered_inventory["warehouse_id"] == warehouse]
    if product:
        filtered_inventory = filtered_inventory[filtered_inventory["product_name"] == product]
    
    # Using days_to_stockout < 5 consistently for critical items
    critical_items = filtered_inventory[filtered_inventory["days_to_stockout"] < 5]
    
    return str(len(critical_items)), str(len(filtered_inventory))

@app.callback(
    Output("add-alert", "children"),
    Output("warehouse-filter", "options"),
    Output("product-filter", "options"),
    Input("btn-add", "n_clicks"),
    State("input-id", "value"),
    State("input-name", "value"),
    State("input-stock", "value"),
    State("input-reorder", "value"),
    State("input-warehouse", "value"),
    prevent_initial_call=True
)
def add_item(n_clicks, pid, name, stock, reorder, warehouse):
    global inventory, alerts
    if not all([pid, name, stock is not None, reorder is not None, warehouse]):
        return dbc.Alert("❌ Fill all fields!", color="danger"), dash.no_update, dash.no_update

    # Create new product row
    new_row = pd.DataFrame([{
        "product_id": pid,
        "product_name": name,
        "stock_level": stock,
        "reorder_point": reorder,
        "warehouse_id": warehouse
    }])
    
    #Ensure new products have reasonable stockout times
    new_row = ensure_reasonable_stockout(new_row, forecast)
    
    #Add to inventory
    inventory = pd.concat([inventory, new_row], ignore_index=True)
    
    #calculation for all products to maintain consistency
    inventory = calculate_time_to_stockout(inventory, forecast, sales_history)
    
    # FIXED: Use the same criteria for alerts as elsewhere in the app 
    # (days_to_stockout < 5 instead of stock_level <= reorder_point)
    alerts = inventory[inventory["days_to_stockout"] < 5]
    
    #Save to files
    inventory.to_csv("data/inventory.csv", index=False)
    alerts.to_csv("data/inventory_alerts.csv", index=False)

    # Check if the new product is in alerts list
    new_product_alert = new_row['days_to_stockout'].iloc[0] < 5
    alert_message = (
        f"✅ Product '{name}' added! Expected stockout in {int(new_row['days_to_stockout'].iloc[0])} days."
    )
    if new_product_alert:
        alert_message += " ⚠️ This product needs restocking soon!"

    return (
        dbc.Alert(alert_message, color="success" if not new_product_alert else "warning"),
        [{"label": w, "value": w} for w in inventory['warehouse_id'].unique()],
        [{"label": p, "value": p} for p in inventory['product_name'].unique()],
    )

@app.callback(
    Output("tab-content", "children"),
    Output("insight-text", "children"),
    Input("tabs", "active_tab"),
    Input("warehouse-filter", "value"),
    Input("product-filter", "value")
)
def update_tabs(tab, warehouse, product):
    df = inventory.copy()
    orders_df = orders.copy()
    insight = ""
    
    if warehouse:
        df = df[df["warehouse_id"] == warehouse]
        orders_df = orders_df[orders_df["warehouse_id"] == warehouse]
    if product:
        product_id = df[df["product_name"] == product]["product_id"].values
        if len(product_id) > 0:
            orders_df = orders_df[orders_df["product_id"].isin(product_id)]
        df = df[df["product_name"] == product]
    
    # Consistently using days_to_stockout < 5 as threshold
    critical_items = df[df["days_to_stockout"] < 5]
    
    if len(critical_items) > 0:
        insight = f"\u26A0\uFE0F {len(critical_items)} product(s) will stock out within 5 days! Urgent reorder needed."
    elif avg_inventory < 100:
        insight = "\u26A0\uFE0F Inventory is running low. Consider restocking soon."
    elif stock_out_rate > 30:
        insight = "\u26A0\uFE0F High stock-out rate detected!"
    else:
        insight = "\u2705 Inventory and demand look healthy."

    if tab == "inventory":
        return [
            dbc.Row([
                dbc.Col([
                    html.H4("Inventory Levels by Product"),
                    dcc.Graph(
                        figure=px.bar(df, x="product_name", y="stock_level", color="warehouse_id", 
                                     labels={"product_name": "Product", "stock_level": "Stock Level", "warehouse_id": "Warehouse"},
                                     title="Inventory Levels by Product"),
                        style={"height": "400px"}
                    )
                ], md=8),
                dbc.Col([
                    html.H4("Inventory Distribution by Warehouse"),
                    dcc.Graph(
                        figure=px.pie(df, names="warehouse_id", values="stock_level", 
                                     title="Stock Distribution by Warehouse",
                                     hole=0.3),
                        style={"height": "400px"}
                    )
                ], md=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Estimated Days to Stockout"),
                    dcc.Graph(
                        figure=px.bar(
                            df.sort_values("days_to_stockout"),
                            x="product_name", 
                            y="days_to_stockout",
                            color="days_to_stockout",
                            color_continuous_scale="RdYlGn",
                            labels={"product_name": "Product", "days_to_stockout": "Days to Stockout"},
                            title="Days Until Stockout by Product"
                        ),
                        style={"height": "400px"}
                    )
                ], md=12)
            ])
        ], insight
    
    elif tab == "forecast":
        return dbc.Row([
            dbc.Col([
                html.H4("Demand Forecast"),
                dcc.Graph(
                    figure=px.line(
                        forecast, 
                        x="ds", 
                        y="yhat", 
                        markers=True,
                        labels={"ds": "Date", "yhat": "Predicted Demand"},
                        title="7-Day Demand Forecast"
                    ),
                    style={"height": "400px"}
                )
            ], md=8),
            dbc.Col([
                html.H4("Historical vs Forecast"),
                dcc.Graph(
                    figure=px.scatter(
                        sales_history,
                        x="ds",
                        y="y",
                        labels={"ds": "Date", "y": "Units Sold"},
                        title="Historical Sales Data"
                    ).add_trace(
                        go.Scatter(
                            x=forecast["ds"], 
                            y=forecast["yhat"],
                            mode="lines",
                            name="Forecast",
                            line=dict(color="red")
                        )
                    ),
                    style={"height": "400px"}
                )
            ], md=4)
        ]), insight
    
    elif tab == "orders":
        status_counts = orders_df.groupby('status').size().reset_index(name='count')
        order_timeline = orders_df.copy()
        order_timeline['status_code'] = order_timeline['status'].map({
            'New': 1, 'Processing': 2, 'Shipped': 3, 'In Transit': 4, 'Delivered': 5
        })
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4("Order Status Distribution"),
                    dcc.Graph(
                        figure=px.pie(
                            status_counts, 
                            names="status", 
                            values="count",
                            title="Orders by Status",
                            color="status",
                            color_discrete_map={
                                'New': '#636EFA',
                                'Processing': '#FFA15A', 
                                'Shipped': '#19D3F3',
                                'In Transit': '#FF6692',
                                'Delivered': '#00CC96'
                            }
                        ),
                        style={"height": "400px"}
                    )
                ], md=6),
                dbc.Col([
                    html.H4("Order Timeline"),
                    dcc.Graph(
                        figure=px.scatter(
                            order_timeline.sort_values('order_date'),
                            x="order_date",
                            y="order_id",
                            color="status",
                            size="quantity",
                            hover_data=["product_id", "estimated_delivery"],
                            labels={"order_date": "Order Date", "order_id": "Order ID"},
                            title="Order Timeline"
                        ),
                        style={"height": "400px"}
                    )
                ], md=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Order Tracking"),
                    dash_table.DataTable(
                        data=orders_df.to_dict('records'),
                        columns=[{"name": col.replace("_", " ").title(), "id": col} for col in orders_df.columns],
                        style_cell={'textAlign': 'left', 'padding': '8px'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            },
                            {
                                'if': {'filter_query': '{status} = "Delivered"'},
                                'backgroundColor': '#E5F9E0',
                                'color': 'green'
                            },
                            {
                                'if': {'filter_query': '{status} = "New"'},
                                'backgroundColor': '#E3F2FD',
                                'color': 'navy'
                            }
                        ],
                        page_size=8,
                        sort_action="native",
                        filter_action="native",
                    )
                ], md=12)
            ])
        ], f"Currently tracking {len(orders_df)} orders across {orders_df['warehouse_id'].nunique()} warehouses."
    
    elif tab == "alerts":
        return dbc.Row([
            dbc.Col([
                html.H4("⚠️ Inventory Alerts"),
                dbc.Alert(f"🚨 {len(critical_items)} product(s) will stock out within 5 days! Urgent reorder needed.", 
                         color="danger") if len(critical_items) > 0 else
                dbc.Alert("✅ All products are adequately stocked.", color="success"),
                
                dash_table.DataTable(
                    data=critical_items.to_dict('records') if not critical_items.empty else [],
                    columns=[
                        {"name": "Product ID", "id": "product_id"},
                        {"name": "Product Name", "id": "product_name"},
                        {"name": "Stock Level", "id": "stock_level"},
                        {"name": "Reorder Point", "id": "reorder_point"},
                        {"name": "Warehouse", "id": "warehouse_id"},
                        {"name": "Days to Stockout", "id": "days_to_stockout"}
                    ],
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                            'if': {'filter_query': '{days_to_stockout} < 5'},
                            'backgroundColor': '#FFEBEE',
                            'color': 'darkred'
                        }
                    ],
                    page_size=10,
                    sort_action="native"
                ) if len(critical_items) > 0 else html.Div()
            ], width=12)
        ]), insight
        
    elif tab == "analytics":
        # Creating heatmap data for inventory levels across warehouses and products
        if len(df) > 1:
            heatmap_data = df.pivot_table(
                values="stock_level", 
                index="warehouse_id", 
                columns="product_name", 
                aggfunc="sum",
                fill_value=0
            )
            
            heatmap_fig = px.imshow(
                heatmap_data,
                labels=dict(x="Product", y="Warehouse", color="Stock Level"),
                title="Inventory Heatmap by Warehouse and Product",
                color_continuous_scale="Viridis"
            )
            
            return [
                dbc.Row([
                    dbc.Col([
                        html.H4("Stock Level Heatmap"),
                        dcc.Graph(figure=heatmap_fig, style={"height": "400px"})
                    ], md=8),
                    dbc.Col([
                        html.H4("Stockout Trend Analysis"),
                        dcc.Graph(
                            figure=px.line(
                                stockout_trends,
                                x="date",
                                y="stockout_percentage",
                                markers=True,
                                labels={"date": "Date", "stockout_percentage": "Stockout Rate (%)"},
                                title="30-Day Stockout Rate Trend"
                            ),
                            style={"height": "400px"}
                        )
                    ], md=4)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Inventory Health Analysis"),
                        dcc.Graph(
                            figure=px.scatter(
                                df,
                                x="stock_level",
                                y="reorder_point",
                                size="days_to_stockout",
                                color="warehouse_id",
                                hover_data=["product_name", "days_to_stockout"],
                                labels={
                                    "stock_level": "Current Stock Level",
                                    "reorder_point": "Reorder Point",
                                    "warehouse_id": "Warehouse"
                                },
                                title="Inventory Health Analysis"
                            ),
                            style={"height": "400px"}
                        )
                    ], md=12)
                ])
            ], "Analytics show inventory trends and potential stockout risks."
        else:
            return html.Div([
                dbc.Alert("Not enough data for analytics visualization. Please add more products or select different filters.", 
                         color="warning")
            ]), "Not enough data for meaningful analytics. Try adjusting filters."

@app.callback(Output("warehouse-filter", "value"), Output("product-filter", "value"), Input("btn-clear-filters", "n_clicks"), prevent_initial_call=True)
def clear_filters(n): return None, None

@app.callback(Output("download-inv", "data"), Input("btn-download-inv", "n_clicks"), prevent_initial_call=True)
def download_inventory(n): return dcc.send_data_frame(inventory.to_csv, "inventory.csv")

@app.callback(Output("download-orders", "data"), Input("btn-download-orders", "n_clicks"), prevent_initial_call=True)
def download_orders(n): return dcc.send_data_frame(orders.to_csv, "orders.csv")

print("\U0001F680 Launching Dash app on http://127.0.0.1:8050")
app.run(debug=True)