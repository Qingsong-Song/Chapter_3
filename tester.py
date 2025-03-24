import numpy as np
import matplotlib.pyplot as plt

def analyze_job_functions():
    """Analyze how the job market functions behave"""
    
    # Function to calculate job finding probability
    def job_finding(θ, zeta):
        """Job finding probability for a worker"""
        prob = 1 / (1 + θ ** (-zeta)) ** (1 / zeta)
        return min(max(prob, 1e-3), 1)
    
    # Function to calculate job filling probability
    def job_filling(θ, zeta):
        """Job filling probability for a firm"""
        prob = 1 / (1 + θ ** zeta) ** (1 / zeta)
        return min(max(prob, 1e-3), 1)
    
    # Create market tightness grid
    θ_values = np.linspace(0.01, 500, 5000)
    
    # Test with different zeta values
    zeta_values = [0.65, 0.75, 0.85]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot job finding probabilities
    plt.subplot(2, 1, 1)
    for zeta in zeta_values:
        finding = [job_finding(θ, zeta) for θ in θ_values]
        plt.plot(θ_values, finding, label=f"Job Finding (zeta={zeta})")
    
    plt.title("Job Finding Probability vs Market Tightness")
    plt.xlabel("Market Tightness (θ)")
    plt.ylabel("Job Finding Probability")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    
    # Plot job filling probabilities
    plt.subplot(2, 1, 2)
    for zeta in zeta_values:
        filling = [job_filling(θ, zeta) for θ in θ_values]
        plt.plot(θ_values, filling, label=f"Job Filling (zeta={zeta})")
    
    plt.title("Job Filling Probability vs Market Tightness")
    plt.xlabel("Market Tightness (θ)")
    plt.ylabel("Job Filling Probability")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Also examine the relationship between the two probabilities
    plt.figure(figsize=(14, 10))
    
    zeta = 0.75  # Use a single zeta value
    
    # Calculate both probabilities
    finding_probs = [job_finding(θ, zeta) for θ in θ_values]
    filling_probs = [job_filling(θ, zeta) for θ in θ_values]
    
    # Plot both on same axes
    plt.subplot(2, 1, 1)
    plt.plot(θ_values, finding_probs, 'b-', label="Job Finding Probability")
    plt.plot(θ_values, filling_probs, 'r-', label="Job Filling Probability")
    plt.title(f"Job Market Probabilities vs Market Tightness (zeta={zeta})")
    plt.xlabel("Market Tightness (θ)")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)
    
    # Plot product of probabilities (efficiency of matching)
    plt.subplot(2, 1, 2)
    plt.plot(θ_values, [f*j for f, j in zip(finding_probs, filling_probs)], 'g-')
    plt.title("Matching Efficiency (Product of Finding & Filling Probabilities)")
    plt.xlabel("Market Tightness (θ)")
    plt.ylabel("Finding × Filling")
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Demonstrate how these probabilities affect steady-state employment
    plt.figure(figsize=(12, 6))
    
    # Parameters
    delta = 0.035  # Job destruction rate
    
    # Calculate steady-state employment for different θ
    employment = []
    for θ in θ_values:
        λ = job_finding(θ, zeta)
        emp = λ / (λ + delta)
        employment.append(emp)
    
    plt.plot(θ_values, employment)
    plt.title(f"Steady-State Employment Rate (delta={delta}, zeta={zeta})")
    plt.xlabel("Market Tightness (θ)")
    plt.ylabel("Employment Rate")
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    analyze_job_functions()