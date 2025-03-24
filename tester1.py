import numpy as np
import matplotlib.pyplot as plt

def analyze_dm_utility():
    """Analyze DM utility function behavior with different parameter values"""
    
    # Define the DM utility function
    def utility_dm(y, Psi, psi):
        """DM utility function"""
        if psi == 1:
            return Psi * np.log(y)
        else:
            return Psi * (y ** (1 - psi) - 1) / (1 - psi)
    
    # Create a grid of y values
    y_values = np.linspace(0.01, 5.0, 500)
    
    # Parameter sets to test
    param_sets = [
        {"Psi": 2.19, "psi": 0.279, "label": "Baseline (Psi=2.19, psi=0.279)"},
        {"Psi": 2.1, "psi": 0.2, "label": "Alternative (Psi=2.1, psi=0.2)"},
        {"Psi": 1.0, "psi": 0.5, "label": "Test 1 (Psi=1.0, psi=0.5)"},
        {"Psi": 1.0, "psi": 0.9, "label": "Test 2 (Psi=1.0, psi=0.9)"}
    ]
    
    # Create plots for utility function
    plt.figure(figsize=(12, 8))
    
    # Plot DM utility for each parameter set
    for params in param_sets:
        utility_values = [utility_dm(y, params["Psi"], params["psi"]) for y in y_values]
        plt.plot(y_values, utility_values, label=params["label"])
    
    plt.title("DM Utility Function for Different Parameter Values")
    plt.xlabel("DM Good (y)")
    plt.ylabel("Utility")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add a subplot focusing on y values near 0
    plt.figure(figsize=(12, 8))
    
    # Smaller range of y values to see behavior near origin
    y_small = np.linspace(0.01, 1.0, 500)
    
    # Plot DM utility for each parameter set
    for params in param_sets:
        utility_values = [utility_dm(y, params["Psi"], params["psi"]) for y in y_small]
        plt.plot(y_small, utility_values, label=params["label"])
    
    plt.title("DM Utility Function Near Origin")
    plt.xlabel("DM Good (y)")
    plt.ylabel("Utility")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # ===== Look at marginal utility =====
    plt.figure(figsize=(12, 8))
    
    # Define the marginal DM utility function
    def marginal_utility_dm(y, Psi, psi):
        """Marginal DM utility function"""
        return Psi * y ** (-psi)
    
    # Plot marginal utility for each parameter set
    for params in param_sets:
        marginal_values = [marginal_utility_dm(y, params["Psi"], params["psi"]) for y in y_values]
        plt.plot(y_values, marginal_values, label=params["label"])
    
    plt.title("DM Marginal Utility Function")
    plt.xlabel("DM Good (y)")
    plt.ylabel("Marginal Utility")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 50)  # Limit y-axis to see the function behavior better
    
    # ===== Test alternative DM utility formulations =====
    plt.figure(figsize=(12, 8))
    
    # Alternative utility function with an intercept adjustment
    def utility_dm_adjusted(y, Psi, psi):
        """Adjusted DM utility function to ensure it passes through origin"""
        if psi == 1:
            return Psi * np.log(y)
        else:
            # Subtract the value at y=0 to ensure u(0)=0
            return Psi * ((y ** (1 - psi) - 1) / (1 - psi))
    
    # Plot adjusted utility function
    for params in param_sets:
        utility_values = [utility_dm_adjusted(y, params["Psi"], params["psi"]) for y in y_values]
        plt.plot(y_values, utility_values, label=params["label"])
    
    plt.title("Adjusted DM Utility Function")
    plt.xlabel("DM Good (y)")
    plt.ylabel("Utility")
    plt.grid(True)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Return some data for further analysis
    return {
        "y_values": y_values,
        "utility_values": {
            params["label"]: [utility_dm(y, params["Psi"], params["psi"]) for y in y_values]
            for params in param_sets
        }
    }

# Call the analysis function
if __name__ == "__main__":
    analyze_dm_utility()