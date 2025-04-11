import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

class LagosWrightAiyagariSolver:
    def __init__(self, params):
        """Initialize solver with model parameters"""
        # Preference parameters
        self.beta = params['beta']      # Discount factor
        self.alpha = params['alpha']    # Probability of DM consumption opportunity
        self.alpha_1 = params['alpha_1']  # Probability of accepting both money and assets
        self.alpha_0 = self.alpha - self.alpha_1  # Probability of accepting only money
        self.gamma = params['gamma']    # Risk aversion parameter
        self.Psi = params['Psi']       # DM utility scaling parameter
        
        # Production and labor market parameters
        self.psi = params['psi']       # Matching function elasticity
        self.zeta = params['zeta']     # Worker bargaining power
        self.nu = params['nu']         # Labor supply elasticity
        self.mu = params['mu']         # Matching efficiency
        self.delta = params['delta']   # Job separation rate
        self.kappa = params['kappa']   # Vacancy posting cost
        self.replace_rate = params['repl_rate']  # Unemployment benefit replacement rate
        
        # Grid specifications
        self.n_a = params['n_a']       # Number of asset grid points
        self.n_m = params['n_m']       # Number of money grid points
        self.n_d = params['n_d']       # Number of deposit grid points
        self.n_l = params['n_l']       # Number of loan grid points
        self.a_min = params['a_min']   # Minimum asset value
        self.a_max = params['a_max']   # Maximum asset value
        self.m_min = params['m_min']   # Minimum money holdings
        self.m_max = params['m_max']   # Maximum money holdings
        self.l_min = params['l_min']   # Minimum deposit/loan value
        self.l_max = params['l_max']   # Maximum deposit/loan value
        self.n_e = 2                   # Employment states: [0=unemployed, 1=employed]
        self.n_z = 3                   # Skill types: [0=low, 1=medium, 2=high]
        self.ny = 300                  # Number of grid points for DM goods
        self.nd = 50                   # Number of grid points for deposit/loan
        
        # Price parameters
        self.prices = np.array([
            params['py'],              # Price of DM goods
            params['Rl'],             # Return on illiquid assets
            params['i']              # Gross Return of banks borrowing
        ])
        
        # Convergence parameters
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        
        # Initialize grids
        self.setup_grids(params)

        
        # Initialize value and policy functions
        self.initialize_functions()

    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Asset grid
        self.a_grid = np.linspace(params['a_min'], params['a_max'], self.n_a)
        
        # Money holdings grid
        self.m_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)
        
        # Deposit grid (real terms)
        self.d_grid = np.linspace(params['m_min'], params['m_max'], self.n_m)

        # loan grid
        self.l_grid = np.linspace(params['l_min'], params['l_max'], self.n_l)
        
        # Employment grid
        self.e_grid = np.array([0, 1])  # 0 = unemployed, 1 = employed
        
        # Productivity/skill grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Low, medium, high skill
        
        # Skill type distribution (fixed - doesn't change over time)
        self.z_dist = np.array([0.62, 0.28, 0.1])  # Distribution of skill types

    
    def initialize_functions(self):
        """Initialize value and policy functions"""
        # Initialize arrays
        self.W = np.zeros((self.n_a, self.n_d, self.n_l, self.n_z, self.n_e))
        self.V = np.zeros((self.n_a, self.n_m, self.n_z, self.n_e))
        
        # Initialize W with reasonable guesses using broadcasting
        a_grid_reshaped = self.a_grid.reshape(-1, 1)  # Shape: (n_a, 1)
        
        
        
        
    def utility(self, c):
        """
        1. Utility function for centralised market
        2. c : endogenously determined by the skill type e, which impacts the budget constraint
        """
        c = np.asarray(c)
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        safe_c = np.maximum(c, eps)
        
        return np.where(c <= 0,
                    -1e5,  # penalty for non-positive consumption
                    (safe_c ** (1 - self.gamma) - 1) / (1 - self.gamma))
    
    def utility_dm(self, y):
        """
        Utility function for decentralized market -- paramters are different from the centralised market
        Generates 26% increase in consumption during preference shocks.
        """
        y = np.asarray(y)
        return np.where(y <= 0,
                    0,  # normalisation in the paper
                    self.Psi * (y ** (1 - self.psi) - 1) / (1 - self.psi))
    
    def κ_prime_inv(self, py):
        """
        Solve the firm's optimal early consumption goods production, given early consumption price.
        Assumed: q_bar = y_bar = 1
        :param py:
        :return: the optimal supply of early consumption goods
        """
        y_temp = 1 / ((1 + py ** (1 / (self.zeta - 1))) ** self.zeta)
        y_tilda = max(min(y_temp, 1), 0)
        return y_tilda

    def κ_fun(self, y):
        """
        κ(y) = q̄ - Q(y), where Q is a PPF and q̄ == 1.
        Opportunity cost of producing y.
        :param y: a vector of early consumption.
        :return:
        """
        cost = np.zeros_like(y)
        positive_mask = (y >= 0)
        cost[positive_mask] = 1 - (1 - y[positive_mask] ** (1 / self.zeta)) ** self.zeta
        return cost

    
    def q_fun(self, py):
        """
        The productivity-adjusted revenue of a job (independent of productivity z).
        """
        y_tilda = self.κ_prime_inv(py)
        q = 1 + py * y_tilda - self.κ_fun(y_tilda)
        return q

    def firm_post_rev(self, prices):
        """
        Calculates the productivity-normalized value of a filled job.
        This represents the value per unit of worker productivity (z).
        Both revenue and costs scale with z, so we normalize by z to get 
        a single market tightness for all productivity types.
        
        Derived from Eq (13): ϕᶠ(z) = zq(pʸ) - zw₁ + (1-δ)ϕᶠ(z)/Rⁱ
        When normalized by z: ϕ̃ᶠ = ϕᶠ(z)/z = q(pʸ) - w₁ + (1-δ)ϕ̃ᶠ/Rⁱ
        
        Returns:
            normalized_rev: The value per unit of productivity (independent of z)
        """
        py = prices[0]
        Rl = prices[1]

        q = self.q_fun(py)
        normalized_rev = (1 - self.mu) * q / (1 - (1 - self.delta) / Rl)
        return normalized_rev
    
    def job_finding(self, θ):
        """
        Job finding probability λ for a worker.
        :param θ: tightness in the labor market:  job vacancy / seekers
        :return: vacancy filling rate between 0 and 1.
        """
        prob = 1 / (1 + θ ** (-self.zeta)) ** (1 / self.zeta)
        return min(max(prob, 1e-3), 1)

    def job_filling(self, θ):
        """
        Job filling rate for a firm.
        :param θ: tightness in the labor market job_opening / seekers
        :return: vacancy filling rate between 0 and 1.
        """
        prob = 1 / (1 + θ ** self.zeta) ** (1 / self.zeta)
        return min(max(prob, 1e-3), 1)
    
    def job_fill_inv(self, x):
        """
        Inverse function of job filling function.
        :param x: probability of job filling
        :return: tightness
        """
        θ = ((1 / x) ** self.zeta - 1) ** (1 / self.zeta)
        return θ

    def solve_θ(self, prices):
        """
        Solves for equilibrium market tightness using the free-entry condition.
        
        The key insight is that both the vacancy creation cost (kappa*z) and the expected 
        benefit of hiring (rev*z) scale with productivity z. Therefore, when comparing 
        costs and benefits, z cancels out, leading to a single market tightness for all 
        productivity types.
        
        Derived from Eq (17): -z*kappa + λ(θ)zϕ̃ᶠ/θRⁱ ≤ 0
        Dividing by z:        -kappa + λ(θ)ϕ̃ᶠ/θRⁱ ≤ 0
        
        Parameters:
            prices: Vector of prices [py, Rl]
        
        Returns:
            θ: Market tightness
            λ: Job-finding probability
            filling: Vacancy-filling probability
        """
        kappa = self.kappa
        Rl = prices[1]
        # Get productivity-normalized value of a filled job
        normalized_rev = self.firm_post_rev(prices=prices)
        
        # Compare expected cost with expected benefit (both normalised by z)
        if kappa >= normalized_rev / Rl:
            θ = 1e-3  # Almost no vacancies if cost exceeds revenue
        else:
            θ = self.job_fill_inv(kappa * Rl / normalized_rev)
        
        # Compute job finding probability & vacancy filling probability
        λ = self.job_finding(θ)
        filling = self.job_filling(θ)
        return θ, λ, filling
    

    def solve_dm_consume(self, W_guess, prices):
        """
        Solve DM problem when household experiences preference shock (ε = 1)
        
        Parameters:
        -----------
        a_m : float
            Money holdings
        a_l = a - a_m
            Illiquid asset holdings
        W : callable
            The CM value function W_e(a, d, l; z)
        phi_m : float
            Real price of money
        p_y : float
            Price of early consumption
        nu : callable
            Utility function for early consumption
        grid_y, grid_l, grid_d : ndarray
            Grids for choice variables
        
        Returns:
        --------
        y_opt, l_opt, d_opt : float
            Optimal early consumption, borrowing, and deposits
        """
        # Unpack prices
        py = prices[0]
        i = prices[2]

        # Unpack grids
        a_grid = self.a_grid
        m_grid = self.m_grid
        d_grid = self.d_grid
        l_grid = self.l_grid
        n_a = self.n_a
        n_m = self.n_m
        v_grid = self.V
        
        # Setup grids for each return
        policy_y0 = np.full(v_grid.shape, 0.0)  # Early consumption grid when omega = 0
        policy_y1 = np.full(v_grid.shape, 0.0)  # Early consumption grid when omega = 1

        
        
        for e_idx in range(self.n_e):
            for z_idx in range(self.n_z):
                for a_idx, a in enumerate(n_a):
                    for m_idx, a_m in enumerate(n_m):
                        a_l = max(min(a - a_m, self.a_max), self.a_min)

                        # case 1: preference shock + omega = 0
                        grid_y0 = np.linspace(0, a_m, self.ny)
                        # case 1.1: no borrow (deposit or not)
                        l_idx = 0
                        v_grid0 = self.utility_dm(grid_y0) + 


                        # case 1.2: borrow


                        # compare the two cases, get the maximum of V & associated policy functions





                        # case 2: preference shock + omega = 1
                        grid_y1 = np.linspace(0, a, self.ny)
                        



    @staticmethod    
    def find_closest_indices(value, grid):
        """
        Find the two closest indices in the grid to the given value.
        
        Parameters:
        -----------
        value : float
            The value to find closest indices for
        grid : numpy.ndarray
            The grid of values
        
        Returns:
        --------
        tuple
            (lower_index, upper_index, lower_weight, upper_weight)
        """
        # Check if value is outside the grid
        if value <= grid[0]:
            return 0, 0, 1.0, 0.0
        elif value >= grid[-1]:
            return len(grid)-1, len(grid)-1, 1.0, 0.0
        
        # Find the index of the grid point just below the value
        lower_idx = np.searchsorted(grid, value, side='right') - 1
        upper_idx = lower_idx + 1
        
        # Calculate weights for interpolation
        grid_range = grid[upper_idx] - grid[lower_idx]
        if grid_range == 0:  # Avoid division by zero
            lower_weight = 1.0
            upper_weight = 0.0
        else:
            lower_weight = (grid[upper_idx] - value) / grid_range
            upper_weight = (value - grid[lower_idx]) / grid_range
        
        return lower_idx, upper_idx, lower_weight, upper_weight


    
    def interpolate_2d(self, x_value, y_value, x_grid, y_grid, grid_values):
        """
        Perform bilinear interpolation for any two dimensions.
        
        Parameters:
        -----------
        x_value : float
            Query value for the first dimension (e.g., asset, loan, etc.)
        y_value : float
            Query value for the second dimension (e.g., deposit, loan, etc.)
        x_grid : numpy.ndarray
            Grid of values for the first dimension
        y_grid : numpy.ndarray
            Grid of values for the second dimension
        grid_values : numpy.ndarray
            Grid values with shape matching x_grid and y_grid dimensions
            This would be a 2D slice of W with fixed values for other dimensions
        
        Returns:
        --------
        float
            Interpolated value
        """
        # Find indices and weights for x dimension
        x_lower, x_upper, x_lower_weight, x_upper_weight = self.find_closest_indices(x_value, x_grid)
        
        # Find indices and weights for y dimension
        y_lower, y_upper, y_lower_weight, y_upper_weight = self.find_closest_indices(y_value, y_grid)
        
        # Perform bilinear interpolation
        interpolated_value = (
            grid_values[x_lower, y_lower] * x_lower_weight * y_lower_weight +
            grid_values[x_upper, y_lower] * x_upper_weight * y_lower_weight +
            grid_values[x_lower, y_upper] * x_lower_weight * y_upper_weight +
            grid_values[x_upper, y_upper] * x_upper_weight * y_upper_weight
        )
        
        return interpolated_value  
        

    
    
    def find_nearest_index(self, grid, value):
        """Find index of the nearest grid point to a given value"""
        idx = np.abs(grid - value).argmin()
        return idx
    
    def find_nearest_lower_index(self, grid, value):
            """Find index of the nearest grid point less than or equal to a given value"""
            indices = np.where(grid <= value)[0]
            if len(indices) == 0:
                return 0
            return indices[-1]

    def market_clearing(self, ):
        """
        Solve for the equilibrium market tightness that clears the labor market.
        """
        # Firm production and revenue calculations
        self.Ys = self.κ_prime_inv(self.prices[0])
        
        # Calculate firm revenue for each skill type
        self.frev = np.zeros(self.n_z)
        for z_idx in range(self.n_z):
            self.frev[z_idx] = self.z_grid[z_idx] * (1.0 + self.prices[0] * self.Ys - self.κ_fun(np.array([self.Ys]))[0])
        
        # Calculate wages and unemployment benefits (base on labor share)
        self.wages_bar = np.zeros((2, self.n_z))
        self.wages_bar[1, :] = self.mu * self.frev  # Employed wage (labor share * revenue)
        self.wages_bar[0, :] = self.replace_rate * self.wages_bar[1, :]  # Unemployed benefits
        
        # Taxes and transfers
        self.Ag0 = params['Ag0']  # Government bond supply
        taulumpsum = ((1.0 / self.prices[1]) - 1.0) * self.Ag0  # Revenue from money creation
        
        # Apply lump-sum transfer to all households using broadcasting
        self.tau = np.full((self.n_a, 2, self.n_z), taulumpsum)
        
        # Full income (wages + transfers) for all states
        self.wages = np.zeros((self.n_a, 2, self.n_z))
        for z_idx in range(self.n_z):
            for e_idx in range(2):
                self.wages[:, e_idx, z_idx] = self.wages_bar[e_idx, z_idx] + self.tau[:, e_idx, z_idx]
        
        # Firm profits and value
        profits = self.frev - self.wages_bar[1, :]
        self.Js = profits / (1.0 - ((1.0 - self.delta) / self.prices[1]))
        
        # Calculate steady state tightness and employment rate
        # First check if the free-entry condition can be satisfied
        entry_condition_max = np.max((self.z_grid * self.kappa * self.prices[1]) / self.Js)
        
        if entry_condition_max < 1:
            # Use the first productivity type to determine tightness (should be the same for all z)
            self.market_tightness = self.job_fill_inv((self.z_grid[0] * self.kappa * self.prices[1]) / self.Js[0])
        else:
            # If entry condition can't be satisfied, set very low tightness
            self.market_tightness = 0.0001
        
        # Calculate job finding probability and employment rate
        self.job_finding_prob = self.job_finding(self.market_tightness)
        self.emp_rate = self.job_finding_prob / (self.delta + self.job_finding_prob)
        
        # Employment transition matrix: rows = current state, cols = next state
        # [0,0] = P(unemployed → unemployed), [0,1] = P(unemployed → employed)
        # [1,0] = P(employed → unemployed), [1,1] = P(employed → employed)
        self.emp_transition = np.array([
            [1 - self.job_finding_prob, self.job_finding_prob],
            [self.delta, 1 - self.delta]
        ])
    


if __name__ == "__main__":
    params = {
        # Preference parameters
        'beta': 0.96,      # Discount factor
        'alpha': 0.075,    # Probability of DM consumption opportunity
        'alpha_1': 0.06,   # Probability of accepting both money and assets
        'gamma': 1.5,      # Risk aversion parameter
        'Psi': 2.2,       # DM utility scaling parameter
        
        # Production and labor market parameters
        'psi': 0.28,      # Matching function elasticity
        'zeta': 0.75,     # Worker bargaining power
        'nu': 1.6,        # Labor supply elasticity
        'mu': 0.7,        # Matching efficiency
        'delta': 0.035,   # Job separation rate
        'kappa': 7.29,    # Vacancy posting cost
        'repl_rate': 0.4, # Unemployment benefit replacement rate
        
        # Grid specifications
        'n_a': 100,       # Number of asset grid points
        'n_m': 50,        # Number of money grid points
        'n_l': 40,        # Number of deposit grid points
        'n_d': 40,        # Number of loan grid points
        'a_min': 0.0,     # Minimum asset value
        'a_max': 20.0,    # Maximum asset value
        'm_min': 0.0,     # Minimum money holdings
        'm_max': 10.0,    # Maximum money holdings
        'l_min': 0,         # Minimum loan value
        'l_max': 10.0,     # Maximum loan value
        'd_min': 0,         # Minimum deposit value
        'd_max': 10.0,     # Maximum deposit value
        'n_e': 2,         # Employment states: [0=unemployed, 1=employed]
        'n_z': 3,         # Skill types: [0=low, 1=medium, 2=high]
        'ny': 300,        # Number of grid points for DM goods
        
        # Price parameters
        'py': 1.0,        # Price of DM goods
        'Rl': 1.03,       # Return on illiquid assets
        'i': 0.02,        # Nominal interest rate

        # Government Spending
        'Ag0': 0.1,      # Exogenous government spending 

        # Convergence parameters
        'max_iter': 500,
        'tol': 1e-5
    }

    
    solver = LagosWrightAiyagariSolver(params)
    
    # Set initial market tightness (will be used to calculate job-finding probability)
    solver.market_tightness = 0.7
    
    # Solve the model for given prices
    convergence_path = solver.solve_model()
    

