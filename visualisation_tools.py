import numpy as np
import matplotlib.pyplot as plt

def plot_function(model, func_name, x_min=0.01, x_max=5.0, num_points=100, 
                 title=None, x_label=None, y_label=None, ax=None, **kwargs):
    """
    Simplified function to plot any one-dimensional model function.
    
    Parameters:
    -----------
    model : object
        Instance of your model class
    func_name : str
        Name of the function to plot (e.g., "utility", "utility_dm")
    x_min, x_max : float
        Minimum and maximum values for the x-axis
    num_points : int
        Number of points to evaluate
    title, x_label, y_label : str
        Plot labels (optional)
    ax : matplotlib.axes
        Axes to plot on. If None, creates a new figure
    **kwargs : dict
        Additional arguments to pass to the plot function
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get the function from the model
    if not hasattr(model, func_name):
        raise ValueError(f"Function '{func_name}' not found in model")
    
    func = getattr(model, func_name)
    
    # Create the x grid
    x = np.linspace(x_min, x_max, num_points)
    
    # Evaluate the function over the grid
    y = np.zeros(num_points)
    
    for i in range(num_points):
        try:
            y[i] = func(x[i])
        except TypeError:
            # Try with different argument patterns if direct call fails
            try:
                y[i] = func(x=x[i])
            except:
                try:
                    y[i] = func(x[i], **kwargs)
                except:
                    raise ValueError(f"Could not evaluate function {func_name} with input {x[i]}")
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    ax.plot(x, y, **kwargs)
    
    # Set labels and title
    if x_label:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel("Input")
        
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(f"{func_name}(x)")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{func_name} Function")
    
    ax.grid(True)
    
    return fig

def compare_functions(model, func_names, x_min=0.01, x_max=5.0, num_points=100, 
                     title=None, x_label=None, labels=None):
    """
    Compare multiple functions from the same model.
    
    Parameters:
    -----------
    model : object
        Instance of your model class
    func_names : list of str
        Names of functions to compare
    x_min, x_max : float
        Minimum and maximum values for the x-axis
    num_points : int
        Number of points to evaluate
    title : str
        Plot title (optional)
    x_label : str
        x-axis label (optional)
    labels : list of str
        Labels for each function (optional)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is None:
        labels = func_names
    
    for i, func_name in enumerate(func_names):
        plot_function(model, func_name, x_min, x_max, num_points, 
                     ax=ax, label=labels[i])
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Function Comparison")
        
    if x_label:
        ax.set_xlabel(x_label)
        
    ax.legend()
    
    return fig

def compare_parameterizations(model1, model2, func_name, x_min=0.01, x_max=5.0, 
                             num_points=100, title=None, x_label=None, labels=None):
    """
    Compare the same function with different parameterizations.
    
    Parameters:
    -----------
    model1, model2 : objects
        Instances of the model class with different parameters
    func_name : str
        Name of the function to compare
    x_min, x_max : float
        Minimum and maximum values for the x-axis
    num_points : int
        Number of points to evaluate
    title : str
        Plot title (optional)
    x_label : str
        x-axis label (optional)
    labels : list of str
        Labels for the two parameterizations (optional)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is None:
        labels = ["Parameterization 1", "Parameterization 2"]
    
    # Plot first parameterization
    plot_function(model1, func_name, x_min, x_max, num_points, 
                 ax=ax, label=labels[0])
    
    # Plot second parameterization
    plot_function(model2, func_name, x_min, x_max, num_points,
                 ax=ax, label=labels[1], linestyle='--')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Comparison of {func_name} Function")
        
    if x_label:
        ax.set_xlabel(x_label)
        
    ax.legend()
    
    return fig

def plot_2d_policy(model, policy_array, x_grid, y_grid, x_label="Variable 1", 
                  y_label="Variable 2", title=None, cmap='viridis', slice_idx=None):
    """
    Plot a 2D policy function or slice of a higher-dimensional policy.
    
    Parameters:
    -----------
    model : object
        Instance of your model class
    policy_array : ndarray
        Policy function array (2D or higher)
    x_grid, y_grid : ndarray
        Grid values for x and y axes
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    cmap : str
        Colormap name
    slice_idx : tuple
        Indices for slicing higher-dimensional policies (e.g., (0,) for first index)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # If policy is higher dimensional, take a slice
    if slice_idx is not None:
        if isinstance(slice_idx, tuple):
            if len(policy_array.shape) > 2 + len(slice_idx):
                raise ValueError(f"Policy has {len(policy_array.shape)} dimensions, but only {len(slice_idx)} slice indices provided")
            
            # Create the appropriate slice object
            idx = tuple(slice_idx) + (slice(None), slice(None))
            policy_2d = policy_array[idx]
        else:
            # Assume it's a single index
            policy_2d = policy_array[slice_idx]
    else:
        if len(policy_array.shape) > 2:
            raise ValueError("Policy has more than 2 dimensions, but no slice_idx provided")
        policy_2d = policy_array
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create the plot
    c = ax.pcolormesh(X, Y, policy_2d.T, cmap=cmap, shading='auto')
    fig.colorbar(c, ax=ax, label="Policy Value")
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Policy Function")
    
    return fig

# Example usage
if __name__ == "__main__":
    # Define a simple test class with some functions
    class TestModel:
        def __init__(self, gamma=1.5, psi=0.28):
            self.gamma = gamma
            self.psi = psi
            
        def utility(self, c):
            """Standard CRRA utility function"""
            if self.gamma == 1:
                return np.log(c)
            else:
                return (c**(1-self.gamma) - 1) / (1-self.gamma)
            
        def marginal_utility(self, c):
            """Marginal utility"""
            return c**(-self.gamma)
            
        def early_utility(self, y):
            """Early utility function"""
            if self.psi == 1:
                return np.log(y)
            else:
                return (y**(1-self.psi) - 1) / (1-self.psi)
    
    # Create models with different parameters
    model1 = TestModel(gamma=1.5, psi=0.28)
    model2 = TestModel(gamma=2.0, psi=0.3)
    
    # Plot a single function
    plot_function(model1, "utility", x_min=0.1, x_max=3.0, title="CRRA Utility Function", 
                 x_label="Consumption")
    
    # Compare different functions from the same model
    compare_functions(model1, ["utility", "marginal_utility"], x_min=0.1, x_max=3.0,
                     title="Utility and Marginal Utility", x_label="Consumption",
                     labels=["Utility", "Marginal Utility"])
    
    # Compare the same function with different parameters
    compare_parameterizations(model1, model2, "utility", x_min=0.1, x_max=3.0,
                            title="CRRA Utility with Different Risk Aversion",
                            x_label="Consumption", 
                            labels=[f"γ={model1.gamma}", f"γ={model2.gamma}"])
    
    # Create a sample 2D policy array
    a_grid = np.linspace(0, 10, 20)
    m_grid = np.linspace(0, 5, 10)
    policy = np.zeros((20, 10))
    
    # Fill with a sample policy function: c(a,m) = 0.05*a + 0.1*m
    for i, a in enumerate(a_grid):
        for j, m in enumerate(m_grid):
            policy[i, j] = 0.05 * a + 0.1 * m
    
    # Plot the 2D policy
    plot_2d_policy(model1, policy, a_grid, m_grid, 
                  x_label="Assets", y_label="Money", 
                  title="Consumption Policy Function")
    
    plt.show()