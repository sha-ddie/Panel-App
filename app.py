import panel as pn
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Initialize Panel extensions for Plotly
pn.extension('plotly')

# ---------------------------------------------------------
# 1. Define the Widgets
# ---------------------------------------------------------
dist_choice = pn.widgets.Select(
    name="Distributions Simulator", 
    options=['Binomial', 'Poisson', 'Normal', 'Standard Normal', 'Exponential']
)

# Distribution Parameters
mu = pn.widgets.FloatSlider(name='Mean (μ)', start=-10, end=10, value=5, step=0.1)
sigma = pn.widgets.FloatSlider(name='Std Dev (σ)', start=0.1, end=5, value=2, step=0.1)
lam = pn.widgets.FloatSlider(name='Lambda (λ)', start=0.1, end=20, value=5, step=0.1)
n_trials = pn.widgets.IntSlider(name='Trials (n)', start=1, end=100, value=20)
p_prob = pn.widgets.FloatSlider(name='Probability (p)', start=0.01, end=0.99, value=0.5, step=0.01)
scale_exp = pn.widgets.FloatSlider(name='Scale (Mean: β)', start=0.1, end=5, value=1, step=0.1)

# Area Under the Curve (AUC) Range Sliders
x_range = pn.widgets.RangeSlider(
    name='Calculate P(a < X < b)', 
    start=-4, end=4, value=(-1, 1), step=0.05
)
x_range_discrete = pn.widgets.RangeSlider(
    name='Calculate P(a ≤ X ≤ b)', 
    start=0, end=4, value=(1, 2), step=1
)

# ---------------------------------------------------------
# 2. Dynamically show/hide sliders AND update Range limits
# ---------------------------------------------------------
def get_relevant_sliders(dist):
    """Shows only relevant sliders and adjusts the AUC range limits."""
    if dist == 'Normal':
        x_range.param.update(start=-10, end=10, value=(-4, 5))
        params = "Parameters: μ, σ"
        return pn.Column(mu, sigma, x_range, pn.pane.Markdown(params), width=300)
        
    elif dist == 'Standard Normal':
        x_range.param.update(start=-4, end=4, value=(0, 4))
        params = "Parameters: μ=0, σ=1 (Fixed)"
        return pn.Column(pn.pane.Markdown(params), x_range, width=300)
        
    elif dist == 'Exponential':
        x_range.param.update(start=0, end=25, value=(0, 2))
        params = "Parameters: β"
        return pn.Column(scale_exp, x_range, pn.pane.Markdown(params), width=300)
        
    elif dist == 'Poisson':
        x_range_discrete.param.update(start=0, end=15, value=(int(lam.value), int(lam.value)))
        params = "Parameters: λ"
        return pn.Column(lam, x_range_discrete, pn.pane.Markdown(params), width=300)
        
    elif dist == 'Binomial':
        x_range_discrete.param.update(start=0, end=n_trials.value, value=(5, 5))
        params = "Parameters: n, p"
        return pn.Column(n_trials, p_prob, x_range_discrete, pn.pane.Markdown(params), width=300)

dynamic_sliders = pn.bind(get_relevant_sliders, dist_choice)

# ---------------------------------------------------------
# 3. The Plotting Function (Reactive with AUC)
# ---------------------------------------------------------
@pn.depends(dist_choice, mu, sigma, lam, n_trials, p_prob, scale_exp, x_range, x_range_discrete)
def plot_distribution(dist, mu_val, sigma_val, lam_val, n_val, p_val, scale_val, bounds, bounds_discrete):
    fig = go.Figure()
    prob_text = ""
    prob_calculation = ""
    param_symbols = []
    parm_values = []
   
    # Handle Continuous Distributions (PDF + Shading)
    if dist in ['Normal', 'Standard Normal', 'Exponential']:
        
        if dist == 'Standard Normal':
            mu_val, sigma_val = 0, 1
            param_symbols, parm_values = ["μ", "σ"], [mu_val, sigma_val]
            
        if dist == 'Exponential':
            param_symbols, parm_values = ["β"], [scale_val]
            x = np.linspace(0, scale_val * 5, 500)
            y = stats.expon.pdf(x, scale=scale_val)
        else:
            param_symbols, parm_values = ["μ", "σ"], [mu_val, sigma_val]
            x = np.linspace(mu_val - 4*sigma_val, mu_val + 4*sigma_val, 500)
            y = stats.norm.pdf(x, mu_val, sigma_val)
            
        # Plot the full line
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='PDF', line=dict(width=3, color='black')))
        
        # Calculate and Plot the Shaded Area
        a, b = bounds
        mask = (x >= a) & (x <= b)
        x_shade, y_shade = x[mask], y[mask]
        
        fig.add_trace(go.Scatter(
            x=x_shade, y=y_shade, 
            fill='tozeroy', 
            name=f'P({a:.2f} < X < {b:.2f})', 
            fillcolor='rgba(0, 150, 255, 0.5)', 
            line=dict(color='rgba(0, 150, 255, 0.8)')
        ))
        
        # Do the actual math for the probability using CDF
        if dist == 'Exponential':
            prob = stats.expon.cdf(b, scale=scale_val) - stats.expon.cdf(a, scale=scale_val)
            prob_calculation = f"""\
            ## Calculation Steps            
            ```python 
            from scipy import stats
            # Define Parameters: Scale = {scale_val:.2f} 
            p_b = stats.expon.cdf({b:.2f}, scale={scale_val:.2f}) 
            p_a = stats.expon.cdf({a:.2f}, scale={scale_val:.2f})
            area = p_b - p_a = {prob:.4f}
            ```"""
        else:
            prob = stats.norm.cdf(b, mu_val, sigma_val) - stats.norm.cdf(a, mu_val, sigma_val)
            prob_calculation = f"""\
            ## Calculation Steps            
            ```python 
            from scipy import stats
            # Define Parameters: Mean = {mu_val:.2f}, Std = {sigma_val:.2f}
            p_b = stats.norm.cdf({b:.2f}, {mu_val:.2f}, {sigma_val:.2f}) 
            p_a = stats.norm.cdf({a:.2f}, {mu_val:.2f}, {sigma_val:.2f})
            area = p_b - p_a = {prob:.4f}
            ```"""
            
        prob_text = f"### 🎯 **P({a:.2f} < X < {b:.2f}) = {prob:.4f}**"
        
    # Handle Discrete Distributions (PMF + Shading)
    elif dist == 'Poisson':
        param_symbols, parm_values = ["λ"], [lam_val]
        x = np.arange(0, int(lam_val * 3) + 1)
        y = stats.poisson.pmf(x, lam_val)
        
        a, b = int(bounds_discrete[0]), int(bounds_discrete[1])
        mask = (x >= a) & (x <= b)
        
        # Default color for out-of-range bars
        colors = ['rgba(111, 84, 174, 0.5)'] * len(x)
        # Highlight color for in-range bars
        for i in np.where(mask)[0]:
            colors[i] = 'rgba(0, 150, 255, 0.8)'
            
        fig.add_trace(go.Bar(x=x, y=y, name='Poisson PMF', marker_color=colors, showlegend=False))
        
        # Calculations
        points = list(range(a, b + 1))
        probs = [stats.poisson.pmf(point, lam_val) for point in points]
        text = " + ".join([f"P(X={p})" for p in points])
        probs_text = " + ".join([f"{prob:.4f}" for prob in probs])
        
        prob_calculation = f"""\
        ## Calculation Steps            
        ```python                     
        from scipy import stats
        # Define Parameters: Lambda (λ) = {lam_val:.2f}
        # {text}
        probs = [{", ".join([f"stats.poisson.pmf({p}, {lam_val})" for p in points])}]
        sum(probs) = {probs_text}
                  = {sum(probs):.4f}
        ```"""
        prob_text = f"### 🎯 **P({a} ≤ X ≤ {b}) = {sum(probs):.4f}**"
        
    elif dist == 'Binomial':
        param_symbols, parm_values = ["n", "p"], [n_val, p_val]
        x = np.arange(0, n_val + 1)
        y = stats.binom.pmf(x, n_val, p_val)
        
        a, b = int(bounds_discrete[0]), int(bounds_discrete[1])
        mask = (x >= a) & (x <= b)
        
        # Default color for out-of-range bars
        colors = ['rgba(46, 139, 87, 0.5)'] * len(x)
        # Highlight color for in-range bars
        for i in np.where(mask)[0]:
            colors[i] = 'rgba(0, 150, 255, 0.8)'
            
        fig.add_trace(go.Bar(x=x, y=y, name='Binomial PMF', marker_color=colors, showlegend=False))
        
        # Calculations
        points = list(range(a, b + 1))
        probs = [stats.binom.pmf(point, n_val, p_val) for point in points]
        text = " + ".join([f"P(X={p})" for p in points])
        probs_text = " + ".join([f"{prob:.4f}" for prob in probs])
        
        prob_calculation = f"""\
        ## Calculation Steps            
        ```python                     
        from scipy import stats
        # Define Parameters: Trials (n) = {n_val}, Prob (p) = {p_val:.2f}
        # {text}
        probs = [{", ".join([f"stats.binom.pmf({p}, {n_val}, {p_val})" for p in points])}]
        sum(probs) = {probs_text}
                  = {sum(probs):.4f}
        ```"""
        prob_text = f"### 🎯 **P({a} ≤ X ≤ {b}) = {sum(probs):.4f}**"

    # Format Layout
    parm_values = [ round(value,2) for value in parm_values]
    params_dict = {symbol: parm_values[index] for index, symbol in enumerate(param_symbols)}
    fig.update_layout(
        title=f"{dist} Distribution  ({params_dict})",
        xaxis_title="Value (x)",
        yaxis_title="Probability Density" if dist not in ['Poisson', 'Binomial'] else "Probability Mass",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=30, t=40, b=20),
        showlegend=False
    )
    
    # Return BOTH the plotly figure and the text pane containing the math
    return pn.Column(
        pn.Column( fig,  pn.pane.Markdown(prob_text, align='center', margin=(10, 0, 0, 0)) ),
        pn.pane.Markdown(prob_calculation, styles={  'background-color': '#f8f9fa', 'padding': '10px','border-radius': '5px', 'font-size': '14px'  } ,
                         margin=(-5, 15, 0, -320))   )
# ---------------------------------------------------------
# 4. Build the Dashboard Layout
# ---------------------------------------------------------

# Sidebar
sidebar = pn.Column(
    pn.pane.Markdown("## Distribution Simulator", styles={'font-size': '15px', 'font-weight': 'bold'}),  dist_choice,  dynamic_sliders,  width=350)
sidebar.styles = {'background': '#f1f3f5', 'padding': '15px', 'border-right': '1px solid #dee2e6'}

# Main Dashboard Assembly
dashboard = pn.Row(sidebar, plot_distribution)
dashboard.styles = {'background': '#f0ebdb'} 

dashboard.servable()