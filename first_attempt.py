import numpy as np
import matplotlib.pyplot as plt
from visualisation_tools import plot_function, compare_parameterizations, compare_functions
from scipy import interpolate
from numba import njit, prange

class LagosWrightAiyagariSolver:
    """
    Solver for the Lagos-Wright-Aiyagari hybrid model using robust value function iteration.
    This implementation prioritizes robustness over speed to test for solution uniqueness.
    """
    
    def __init__(self, params):
        # Model parameters
        self.beta = params.get('beta', 0.96)  # Discount factor

        # DM meeting probability and payment type probabilities
        self.alpha = params.get('alpha', 0.075)  # Overall probability of needing to consume in DM
        self.alpha_1 = params.get('alpha_1', 0.06)  # Probability of case where both cash and assets accepted
        self.alpha_0 = 1 - self.alpha_1  # Probability of case where only cash accepted

        self.gamma = params.get('gamma', 1.5)  # Curvature of utility function
        self.Psi = params.get('Psi', 2.2)  # Level of early utility
        self.psi = params.get('psi', 0.28) # Curvature of early utility
        self.zeta = params.get('zeta', 0.75)  # Curvature of production function
        self.nu = params.get('nu', 1.6)  # Matching function curvature
        self.mu = params.get('mu', 0.7)  # Share of revenue going to workers
        self.delta = params.get('delta', 0.035)  # Job destruction rate
        self.kappa = params.get('kappa', 7.29)  # Job posting cost

        self.replace_rate = params.get('repl_rate', 0.4)  # Fraction of unemployment compensation / working wage income

        # Price vector: [py, Rl, phi_m, i]
        self.prices = np.array([
            params.get('py', 1.0),      # Price of early consumption goods
            params.get('Rl', 1.03),     # Return on illiquid assets
            params.get('phi_m', 1.0),   # Price of money
            params.get('i', 0.02)       # Nominal interest rate
        ])

        
        # Grid specifications
        self.n_a = params.get('n_a', 100)  # Grid size for assets
        self.n_m = params.get('n_m', 50)   # Grid size for money
        self.n_z = params.get('n_z', 3)    # Number of productivity states
        
        # Price parameters
        self.phi_m = params.get('phi_m', 1.0)  # Price of money
        self.interest_rate = params.get('interest_rate', 0.03)  # Interest rate
        
        # Convergence parameters
        self.max_iter = params.get('max_iter', 1000)
        self.tol = params.get('tol', 1e-6)
        
        # Initialize grids
        self.setup_grids(params)
        
        # Initialize value and policy functions
        self.initialize_functions()
    
    def setup_grids(self, params):
        """Set up state space grids with appropriate spacing"""
        # Asset grid (potentially non-uniform)
        a_min = params.get('a_min', 0.0)
        a_max = params.get('a_max', 20.0)
        self.a_grid = np.linspace(a_min, a_max, self.n_a)
        
        # Money holdings grid
        m_min = params.get('m_min', 0.0)
        m_max = params.get('m_max', 10.0)
        self.m_grid = np.linspace(m_min, m_max, self.n_m)
        
        # Productivity grid
        self.z_grid = np.array([0.58, 0.98, 2.15])  # Productivity levels
        self.gz = np.array([0.62, 0.28, 0.1])  # Productivity transition probabilities
        
    
    def initialize_functions(self):
        """Initialize value and policy functions"""
        # Value function: dimensions (productivity, assets, money)
        self.V = np.zeros((self.n_e, self.n_a, self.n_m))
        
        # Initialize with a reasonable guess (e.g., based on steady-state consumption)
        for e_idx in range(self.n_e):
            for a_idx in range(self.n_a):
                for m_idx in range(self.n_m):
                    # This is a placeholder - would need model-specific initialization
                    self.V[e_idx, a_idx, m_idx] = self.utility(
                        0.05 * (self.a_grid[a_idx] + self.phi_m * self.m_grid[m_idx]), 
                        self.e_grid[e_idx]
                    ) / (1 - self.beta)
        
        # Policy functions
        self.policy_c = np.zeros((self.n_e, self.n_a, self.n_m))  # Consumption
        self.policy_a_next = np.zeros((self.n_e, self.n_a, self.n_m))  # Next period assets
        self.policy_m_next = np.zeros((self.n_e, self.n_a, self.n_m))  # Next period money
        
        # Additional policies for decentralized market
        self.policy_y = np.zeros((self.n_e, self.n_a, self.n_m, 2))  # DM consumption (for ω={0,1})
        self.policy_b = np.zeros((self.n_e, self.n_a, self.n_m, 2))  # DM borrowing (for ω={0,1})
        self.policy_d = np.zeros((self.n_e, self.n_a, self.n_m))     # Money held for precautionary purposes
    
    def utility(self, c):
        """
        1. Utility function for centralised market
        2. c : endogenously determined by the skill type e, which impacts the budget constraint
        """
        return (c ** (1 - self.gamma) - 1) / (1 - self.gamma)
    
    def utility_dm(self, y):
        """
        Utility function for decentralized market -- paramters are different from the centralised market
        Generates 26% increase in consumption during preference shocks.
        """
        return self.Psi * (y ** (1 - self.psi) - 1) / (1 - self.psi)
    
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
        
        # Compare expected cost with expected benefit (both normalized by z)
        if kappa >= normalized_rev / Rl:
            θ = 1e-3  # Almost no vacancies if cost exceeds revenue
        else:
            θ = self.job_fill_inv(kappa * Rl / normalized_rev)
        
        # Compute job finding probability & vacancy filling probability
        λ = self.job_finding(θ)
        filling = self.job_filling(θ)
        return θ, λ, filling

    
    def solve_dm_problem(self, e_idx, a_idx, m_idx):
        """
        Solve the decentralised market problem for a given state.
        
        Key considerations:
        1. Households must consume if hit by preference shock (with probability alpha)
        2. Asset acceptance depends on omega:
        - With probability alpha_0, only cash is accepted (ω=0)
        - With probability alpha_1, both cash and illiquid assets are accepted (ω=1)
        3. Borrowing decision must compare utilities with and without borrowing
        """
        a = self.a_grid[a_idx]
        m = self.m_grid[m_idx]
        e = self.e_grid[e_idx]
        
        # Extract prices from vector
        py = self.prices[0]
        Rl = self.prices[1]
        phi_m = self.prices[2]
        i = self.prices[3]
        
        # Case ω=0: Only money is accepted
        # Maximum affordable y with only money (no borrowing)
        y0_max_no_borrow = m * phi_m / py
        
        # Without borrowing:
        # Try different y0 values up to y0_max_no_borrow
        y0_values_no_borrow = np.linspace(0, y0_max_no_borrow, 50)
        util0_no_borrow = np.zeros_like(y0_values_no_borrow)
        
        for i, y0 in enumerate(y0_values_no_borrow):
            # Calculate post-DM wealth
            a_post = a + m * self.phi_m - py * y0
            # For case ω=0 without borrowing
            b0 = 0
            # Calculate utility
            util0_no_borrow[i] = self.utility_dm(y0) + self.interpolate_CM_value(a_post, -b0, e_idx)
        
        # Find optimal y0 without borrowing
        if len(util0_no_borrow) > 0:
            best_idx = np.argmax(util0_no_borrow)
            optimal_y0_no_borrow = y0_values_no_borrow[best_idx]
            max_util0_no_borrow = util0_no_borrow[best_idx]
        else:
            optimal_y0_no_borrow = 0
            max_util0_no_borrow = self.utility_dm(0) + self.interpolate_CM_value(a + m * self.phi_m, 0, e_idx)
        
        # With borrowing:
        # We need to consider higher y values that require borrowing
        y0_max_with_borrow = min(2*y0_max_no_borrow, 10)  # Arbitrary upper bound
        y0_values_with_borrow = np.linspace(y0_max_no_borrow+0.01, y0_max_with_borrow, 50)
        util0_with_borrow = np.zeros_like(y0_values_with_borrow)
        b0_values = np.zeros_like(y0_values_with_borrow)
        
        for i, y0 in enumerate(y0_values_with_borrow):
            # Calculate required borrowing
            payment_needed = py * y0
            available_funds = m * self.phi_m
            b0 = (payment_needed - available_funds) / self.phi_m
            
            # Calculate post-DM wealth and utility
            a_post = a  # Only money was spent, assets remain
            util0_with_borrow[i] = self.utility_dm(y0) + self.interpolate_CM_value(a_post, -b0, e_idx)
            b0_values[i] = b0
        
        # Find optimal y0 with borrowing
        if len(util0_with_borrow) > 0:
            best_idx = np.argmax(util0_with_borrow)
            optimal_y0_with_borrow = y0_values_with_borrow[best_idx]
            optimal_b0_with_borrow = b0_values[best_idx]
            max_util0_with_borrow = util0_with_borrow[best_idx]
        else:
            optimal_y0_with_borrow = 0
            optimal_b0_with_borrow = 0
            max_util0_with_borrow = -np.inf
        
        # Compare utilities with and without borrowing
        if max_util0_no_borrow >= max_util0_with_borrow:
            optimal_y0 = optimal_y0_no_borrow
            optimal_b0 = 0
        else:
            optimal_y0 = optimal_y0_with_borrow
            optimal_b0 = optimal_b0_with_borrow
        
        # Case ω=1: Both money and assets are accepted
        # Maximum affordable y with money and assets (no borrowing)
        y1_max_no_borrow = (m * self.phi_m + a) / py
        
        # Without borrowing:
        # Try different y1 values up to y1_max_no_borrow
        y1_values_no_borrow = np.linspace(0, y1_max_no_borrow, 50)
        util1_no_borrow = np.zeros_like(y1_values_no_borrow)
        
        for i, y1 in enumerate(y1_values_no_borrow):
            # Calculate post-DM wealth with full payment from available funds
            payment = py * y1
            remaining_wealth = m * self.phi_m + a - payment
            # For case ω=1 without borrowing
            b1 = 0
            # Calculate utility
            util1_no_borrow[i] = self.utility_dm(y1) + self.interpolate_CM_value(remaining_wealth, -b1, e_idx)
        
        # Find optimal y1 without borrowing
        if len(util1_no_borrow) > 0:
            best_idx = np.argmax(util1_no_borrow)
            optimal_y1_no_borrow = y1_values_no_borrow[best_idx]
            max_util1_no_borrow = util1_no_borrow[best_idx]
        else:
            optimal_y1_no_borrow = 0
            max_util1_no_borrow = self.utility_dm(0) + self.interpolate_CM_value(m * self.phi_m + a, 0, e_idx)
        
        # With borrowing:
        # We need to consider higher y1 values that require borrowing
        y1_max_with_borrow = min(2*y1_max_no_borrow, 10)  # Arbitrary upper bound
        y1_values_with_borrow = np.linspace(y1_max_no_borrow+0.01, y1_max_with_borrow, 50)
        util1_with_borrow = np.zeros_like(y1_values_with_borrow)
        b1_values = np.zeros_like(y1_values_with_borrow)
        
        for i, y1 in enumerate(y1_values_with_borrow):
            # Calculate required borrowing
            payment_needed = py * y1
            available_funds = m * self.phi_m + a
            b1 = (payment_needed - available_funds) / self.phi_m
            
            # Calculate post-DM wealth and utility
            post_wealth = 0  # All assets and money spent, only borrowed amount remains
            util1_with_borrow[i] = self.utility_dm(y1) + self.interpolate_CM_value(post_wealth, -b1, e_idx)
            b1_values[i] = b1
        
        # Find optimal y1 with borrowing
        if len(util1_with_borrow) > 0:
            best_idx = np.argmax(util1_with_borrow)
            optimal_y1_with_borrow = y1_values_with_borrow[best_idx]
            optimal_b1_with_borrow = b1_values[best_idx]
            max_util1_with_borrow = util1_with_borrow[best_idx]
        else:
            optimal_y1_with_borrow = 0
            optimal_b1_with_borrow = 0
            max_util1_with_borrow = -np.inf
        
        # Compare utilities with and without borrowing
        if max_util1_no_borrow >= max_util1_with_borrow:
            optimal_y1 = optimal_y1_no_borrow
            optimal_b1 = 0
        else:
            optimal_y1 = optimal_y1_with_borrow
            optimal_b1 = optimal_b1_with_borrow
        
        # No preference shock case (depositor)
        # Try different deposit amounts
        d_max = m  # Can't deposit more money than you have
        d_values = np.linspace(0, d_max, 50)
        util_d = np.zeros_like(d_values)
        
        for i, d in enumerate(d_values):
            # Calculate post-DM wealth
            post_wealth = a + m * self.phi_m - d * self.phi_m
            # Calculate utility (no early consumption utility)
            util_d[i] = self.interpolate_CM_value(post_wealth, d, e_idx)
        
        # Find optimal d
        if len(util_d) > 0:
            best_idx = np.argmax(util_d)
            optimal_d = d_values[best_idx]
        else:
            optimal_d = 0
        
        # Store optimal policies
        self.policy_y[e_idx, a_idx, m_idx, 0] = optimal_y0
        self.policy_y[e_idx, a_idx, m_idx, 1] = optimal_y1
        self.policy_b[e_idx, a_idx, m_idx, 0] = optimal_b0
        self.policy_b[e_idx, a_idx, m_idx, 1] = optimal_b1
        self.policy_d[e_idx, a_idx, m_idx] = optimal_d
    
    def solve_bellman_iteration(self):
        """Solve the model using value function iteration"""
        converged = False
        iter_count = 0
        
        while not converged and iter_count < self.max_iter:
            V_next = np.zeros_like(self.V)
            
            # Loop over current states
            for e_idx in range(self.n_e):
                for a_idx in range(self.n_a):
                    for m_idx in range(self.n_m):
                        # Current state
                        e = self.e_grid[e_idx]
                        a = self.a_grid[a_idx]
                        m = self.m_grid[m_idx]
                        
                        # Solve for optimal policies and update value function
                        V_next[e_idx, a_idx, m_idx] = self.solve_single_state(e_idx, a_idx, m_idx)
            
            # Check convergence
            max_diff = np.max(np.abs(V_next - self.V))
            converged = max_diff < self.tol
            
            # Update value function
            self.V = V_next.copy()
            
            iter_count += 1
            if iter_count % 10 == 0:
                print(f"Iteration {iter_count}, Max Diff: {max_diff:.8f}")
        
        if converged:
            print(f"Converged after {iter_count} iterations")
        else:
            print("Failed to converge within maximum iterations")
        
        # After convergence, compute final policy functions
        self.compute_policy_functions()
        
        return converged
    
    def solve_single_state(self, e_idx, a_idx, m_idx):
        """
        Solve Bellman equation for a single state (e, a, m)
        Returns the optimized value function at this state
        """
        e = self.e_grid[e_idx]
        a = self.a_grid[a_idx]
        m = self.m_grid[m_idx]
        
        # Total resources available
        phi_m = self.phi_m
        B = m  # Money holdings
        income = a + phi_m * (1 + self.interest_rate) * B + self.income_transfer(e)
        
        # Grid search over possible choices (asset and money combination)
        best_value = -np.inf
        best_c = 0
        best_a_next_idx = 0
        best_m_next_idx = 0
        
        # Loop over all possible next period asset and money combinations
        for a_next_idx in range(self.n_a):
            for m_next_idx in range(self.n_m):
                a_next = self.a_grid[a_next_idx]
                m_next = self.m_grid[m_next_idx]
                
                # Cost of purchasing these assets
                # Simplified cost calculation - replace with actual pricing
                cost = a_next / (1 + self.interest_rate) + phi_m * m_next
                
                # Check if affordable
                if cost <= income:
                    # Current period consumption
                    c = income - cost
                    
                    if c > 0:  # Ensure positive consumption
                        # Current utility
                        current_utility = self.utility(c, e)
                        
                        # Expected future value
                        expected_future = 0
                        
                        # Loop over future productivity states
                        for e_next_idx in range(self.n_e):
                            e_next = self.e_grid[e_next_idx]
                            prob = self.e_transition[e_idx, e_next_idx]
                            
                            # With probability alpha, agent needs to consume in DM
                            dm_value = 0
                            
                            # First case: Only cash accepted (omega = 0)
                            omega = 0
                            y0, b0 = self.solve_dm_problem(a_next, m_next, e_next_idx, omega)
                            
                            # Resources after DM with only cash accepted
                            a_post_dm0 = a_next  # Assets unchanged
                            m_post_dm0 = m_next + b0  # Money holdings may change
                            
                            # Convert post-DM state to indices
                            a_post_idx0 = np.searchsorted(self.a_grid, a_post_dm0)
                            if a_post_idx0 == self.n_a:
                                a_post_idx0 = self.n_a - 1
                            
                            m_post_idx0 = np.searchsorted(self.m_grid, m_post_dm0)
                            if m_post_idx0 == self.n_m:
                                m_post_idx0 = self.n_m - 1
                            
                            # Future value including DM utility for omega = 0
                            dm_utility0 = self.utility_dm(y0)
                            dm_continuation0 = self.V[e_next_idx, a_post_idx0, m_post_idx0]
                            
                            # Second case: Both cash and assets accepted (omega = 1)
                            omega = 1
                            y1, b1 = self.solve_dm_problem(a_next, m_next, e_next_idx, omega)
                            
                            # Resources after DM with cash and assets accepted
                            # Simplified assumption: assets used after money is depleted
                            a_post_dm1 = max(0, a_next - max(0, y1 - m_next))
                            m_post_dm1 = max(0, m_next - y1) + b1
                            
                            # Convert post-DM state to indices
                            a_post_idx1 = np.searchsorted(self.a_grid, a_post_dm1)
                            if a_post_idx1 == self.n_a:
                                a_post_idx1 = self.n_a - 1
                            
                            m_post_idx1 = np.searchsorted(self.m_grid, m_post_dm1)
                            if m_post_idx1 == self.n_m:
                                m_post_idx1 = self.n_m - 1
                            
                            # Future value including DM utility for omega = 1
                            dm_utility1 = self.utility_dm(y1)
                            dm_continuation1 = self.V[e_next_idx, a_post_idx1, m_post_idx1]
                            
                            # Weighted average of DM outcomes based on alpha_0 and alpha_1
                            dm_value = self.alpha * (
                                self.alpha_0 * (dm_utility0 + dm_continuation0) + 
                                self.alpha_1 * (dm_utility1 + dm_continuation1)
                            )
                            
                            # Value without DM consumption need (probability 1-alpha)
                            no_dm_prob = 1 - self.alpha
                            no_dm_value = no_dm_prob * self.V[e_next_idx, a_next_idx, m_next_idx]
                            
                            # Total expected future value for this productivity transition
                            expected_future += prob * (dm_value + no_dm_value)
                        
                        # Total value
                        value = current_utility + self.beta * expected_future
                        
                        # Update best choice if this is better
                        if value > best_value:
                            best_value = value
                            best_c = c
                            best_a_next_idx = a_next_idx
                            best_m_next_idx = m_next_idx
        
        # Update policy functions for this state
        self.policy_c[e_idx, a_idx, m_idx] = best_c
        self.policy_a_next[e_idx, a_idx, m_idx] = self.a_grid[best_a_next_idx]
        self.policy_m_next[e_idx, a_idx, m_idx] = self.m_grid[best_m_next_idx]
        
        # Return optimized value
        return best_value
    
    
    def compute_policy_functions(self):
        """
        Compute detailed policy functions after value function convergence
        This includes DM policies that depend on future states
        """
        # This would be a more detailed policy computation
        # For now, we've already computed basic policies during value function iteration
        pass
    
    def compute_stationary_distribution(self):
        """
        Compute the stationary distribution of households over states
        Returns: distribution array of shape (n_e, n_a, n_m)
        """
        # Initialize distribution
        dist = np.ones((self.n_e, self.n_a, self.n_m))
        dist = dist / dist.sum()
        
        # Iterate until convergence
        for iter in range(1000):  # Maximum iterations for distribution
            next_dist = np.zeros_like(dist)
            
            # For each current state
            for e_idx in range(self.n_e):
                for a_idx in range(self.n_a):
                    for m_idx in range(self.n_m):
                        # Current mass at this state
                        mass = dist[e_idx, a_idx, m_idx]
                        if mass > 0:
                            # Policy at this state
                            a_next = self.policy_a_next[e_idx, a_idx, m_idx]
                            m_next = self.policy_m_next[e_idx, a_idx, m_idx]
                            
                            # Find grid indices for next period assets
                            a_next_idx = np.searchsorted(self.a_grid, a_next)
                            if a_next_idx == self.n_a:
                                a_next_idx = self.n_a - 1
                                
                            m_next_idx = np.searchsorted(self.m_grid, m_next)
                            if m_next_idx == self.n_m:
                                m_next_idx = self.n_m - 1
                            
                            # Transition to next productivity states
                            for e_next_idx in range(self.n_e):
                                prob = self.e_transition[e_idx, e_next_idx]
                                next_dist[e_next_idx, a_next_idx, m_next_idx] += mass * prob
            
            # Normalize
            next_dist = next_dist / next_dist.sum()
            
            # Check convergence
            max_diff = np.max(np.abs(next_dist - dist))
            if max_diff < 1e-8:
                print(f"Distribution converged after {iter+1} iterations")
                break
                
            dist = next_dist.copy()
        
        return dist
    
    def aggregate_quantities(self, prices, dist):
        """
        Note: Since the market tightness is independent of skill types z,
                    we didn't adjust the market clear function since employment will stay the same.
                - Optimal early consumption supply: z*y_s.
                - Optimal illiquid asset based on firm's profits Js: z*Φ^f(z).
                - Demand side: captured skill types through the household distribution.
        :param return_error: depending on the root-finding algorithm, the return of this function is different
        :param prices: early consumption price + return of partially liquid assets.
        :return:
        """

        Na, mu, rrate, delta = self.Na, self.mu, self.replace_rate, self.delta
        Nz, zgrid, gz = self.Nz, self.zgrid, self.gz
        Rm, alpha_1, alpha = self.Rm, self.alpha_1, self.alpha
        Ag, Rm = self.Ag, self.Rm

        # Unpack the price vector
        py = prices[0]  # Price of early consumption
        Rl = prices[1]  # Return on illiquid assets

        # Set taxes (or transfers)
        tau = np.full((Na, 2, Nz), ((1 / Rl) - 1.0) * Ag)
        
        # Calculate optimal early consumption supply
        Ys = self.κ_prime_inv(py=py)  # Ys = y/z: optimal supply of early consumption goods per unit of productivity
        
        # Calculate the total firm revenue for each productivity type
        # This is the revenue before wage payments, depending on productivity z
        frev = self.zgrid * (1.0 + py * Ys - self.κ_fun(Ys))
        
        # Calculate wage rates for employed workers (share μ of revenue)
        w1_bar = mu * frev
        
        # Calculate unemployment benefits (fraction rrate of employed wage)
        wo_bar = rrate * w1_bar
        
        # Store wage rates in a matrix: [employment status × skill types]
        wages_bar = np.array([wo_bar, w1_bar])

        # Update total wages including transfers
        wages = np.zeros((Na, 2, Nz))
        for j in range(Nz):
            for i in range(2):
                wages[:, i, j] = wages_bar[i, j] + tau[:, i, j]

        # Calculate firm profits (revenue minus wage payments)
        profits = frev - wages_bar[1, :]
        
        # Calculate present value of firm profits (Js)
        self.Js = profits / (1 - (1 - delta) / Rl)

        # Solve for market tightness and employment transition probabilities
        θ, λ, filling = self.solve_θ(prices=prices)
        
        # Update employment transition matrix (2×2 Markov chain)
        P = np.array([[1 - λ, λ],
                    [delta, 1 - delta]])
        
        # Calculate steady state employment rate
        emp = λ / (delta + λ)
        
        # Calculate total firm revenue (z-dependent)
        frev = self.zgrid * (1.0 + py * Ys - self.κ_fun(Ys))
        
        # Calculate wages (z-dependent)
        w1_bar = mu * frev  # Employed workers get mu share of revenue
        wo_bar = rrate * w1_bar  # Unemployed get replacement rate * wage
        wages_bar = np.array([wo_bar, w1_bar])
        
        # Calculate z-dependent profits and job values
        profits = frev - wages_bar[1, :]
        Js = profits / (1 - (1 - delta) / Rl)
        
        # Calculate z-independent market tightness
        θ, λ, filling = self.solve_θ(prices=prices)

        # add two more goods

        # Aggregate assets
        agg_assets = 0
        agg_money = 0
        
        for e_idx in range(self.n_e):
            for a_idx in range(self.n_a):
                for m_idx in range(self.n_m):
                    mass = dist[e_idx, a_idx, m_idx]
                    agg_assets += mass * self.a_grid[a_idx]
                    agg_money += mass * self.m_grid[m_idx]
        
        return {
            'K': agg_assets,
            'M': agg_money
        }
    
    def check_market_clearing(self, dist):
        """
        Check if markets clear given current prices and distribution
        Returns: excess demand in each market
        """
        # Get aggregate quantities
        agg = self.aggregate_quantities(dist)
        
        # Capital market clearing (simplified)
        capital_excess = agg['K'] - self.target_capital()
        
        # Money market clearing (simplified)
        money_excess = agg['M'] - self.money_supply()
        
        return {
            'capital_excess': capital_excess,
            'money_excess': money_excess
        }
    
    def target_capital(self):
        """Target capital based on current interest rate"""
        # Placeholder - this would depend on production function
        return 10.0
    
    def money_supply(self):
        """Money supply (exogenous in basic model)"""
        # Placeholder
        return 5.0
    
    def update_prices(self, excess):
        """
        Update prices based on excess demand
        Returns: new prices (phi_m, interest_rate)
        """
        # Damped price adjustment
        damping = 0.3
        
        # Update interest rate based on capital excess
        dr = -0.01 * excess['capital_excess']
        new_r = self.interest_rate + damping * dr
        
        # Update money price based on money excess
        dphi = -0.01 * excess['money_excess'] / self.money_supply()
        new_phi = self.phi_m + damping * dphi
        
        return new_phi, new_r
    
    def solve_equilibrium(self, max_price_iter=20):
        """
        Solve for general equilibrium prices
        """
        for iter in range(max_price_iter):
            print(f"\nPrice iteration {iter+1}")
            print(f"Current prices: phi_m = {self.phi_m:.6f}, r = {self.interest_rate:.6f}")
            
            # Solve individual problem given prices
            self.solve_bellman_iteration()
            
            # Compute stationary distribution
            dist = self.compute_stationary_distribution()
            
            # Check market clearing
            excess = self.check_market_clearing(dist)
            print(f"Market clearing errors: Capital = {excess['capital_excess']:.6f}, Money = {excess['money_excess']:.6f}")
            
            # Check if markets approximately clear
            if abs(excess['capital_excess']) < 0.01 and abs(excess['money_excess']) < 0.01:
                print("Equilibrium found!")
                break
            
            # Update prices
            new_phi, new_r = self.update_prices(excess)
            
            # Apply price updates
            self.phi_m = new_phi
            self.interest_rate = new_r
        
        # Return final equilibrium objects
        return {
            'value_function': self.V,
            'policy_c': self.policy_c,
            'policy_a': self.policy_a_next,
            'policy_m': self.policy_m_next,
            'distribution': dist,
            'phi_m': self.phi_m,
            'interest_rate': self.interest_rate
        }
    

    def visualize_convergence(self, iter_count):
        """Generate visualizations to track convergence progress"""
        # Only show for selected states to avoid clutter
        e_idx = 1  # Employed
        z_idx = 1  # Medium skill
        
        plt.figure(figsize=(15, 10))
        
        # Plot W value function for different asset levels
        plt.subplot(2, 2, 1)
        D_idx = self.n_D // 2  # Mid-point of deposit grid
        plt.plot(self.a_grid, self.W[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.W[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.title(f'CM Value Function (Iteration {iter_count})')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot V value function
        plt.subplot(2, 2, 2)
        m_idx = self.n_m // 2  # Mid-point of money grid
        plt.plot(self.a_grid, self.V[0, :, m_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.V[1, :, m_idx, z_idx], 'r-', label='Employed')
        plt.title('DM Value Function')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot consumption policy
        plt.subplot(2, 2, 3)
        plt.plot(self.a_grid, self.policy_c[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.title('Consumption Policy')
        plt.xlabel('Assets')
        plt.ylabel('Consumption')
        plt.legend()
        
        # Plot next period asset policy
        plt.subplot(2, 2, 4)
        plt.plot(self.a_grid, self.policy_a_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
        plt.plot(self.a_grid, self.policy_a_next[1, :, D_idx, z_idx], 'r-', label='Employed')
        plt.plot(self.a_grid, self.a_grid, 'k--', label='45-degree line')
        plt.title('Next Period Assets Policy')
        plt.xlabel('Current Assets')
        plt.ylabel('Next Period Assets')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'convergence_iter_{iter_count}.png')
        plt.close()

    def visualize_solution(self):
        """Generate comprehensive visualizations to verify economic sensibility of the solution"""
        # Create a folder for visualizations if it doesn't exist
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Value Functions
        plt.figure(figsize=(15, 12))
        
        # Plot CM value function (W) by employment and skill
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.W[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.W[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'CM Value Function (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Value')
            plt.legend()
            
            # Plot DM value function (V) by employment and skill
            plt.subplot(3, 2, z_idx*2 + 2)
            m_idx = self.n_m // 2  # Middle of money grid
            plt.plot(self.a_grid, self.V[0, :, m_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.V[1, :, m_idx, z_idx], 'r-', label='Employed')
            plt.title(f'DM Value Function (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/value_functions.png')
        plt.close()
        
        # 2. CM Policy Functions
        plt.figure(figsize=(15, 12))
        
        # Plot consumption policy
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.policy_c[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'Consumption Policy (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Consumption')
            plt.legend()
            
            # Plot asset policy with 45-degree line
            plt.subplot(3, 2, z_idx*2 + 2)
            plt.plot(self.a_grid, self.policy_a_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_a_next[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.plot(self.a_grid, self.a_grid, 'k--', label='45-degree line')
            plt.title(f'Next Period Assets (Skill Level {z_idx})')
            plt.xlabel('Current Assets')
            plt.ylabel('Next Period Assets')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/cm_policies.png')
        plt.close()
        
        # 3. Money Policies
        plt.figure(figsize=(15, 12))
        
        # Plot money holding policy
        for z_idx in range(self.n_z):
            plt.subplot(3, 2, z_idx*2 + 1)
            D_idx = self.n_D // 2  # Middle of deposit grid
            plt.plot(self.a_grid, self.policy_m_next[0, :, D_idx, z_idx], 'b-', label='Unemployed')
            plt.plot(self.a_grid, self.policy_m_next[1, :, D_idx, z_idx], 'r-', label='Employed')
            plt.title(f'Money Holding Policy (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Next Period Money')
            plt.legend()
            
            # Plot money-to-asset ratio
            plt.subplot(3, 2, z_idx*2 + 2)
            money_ratio_unemployed = self.policy_m_next[0, :, D_idx, z_idx] / (self.policy_a_next[0, :, D_idx, z_idx] + 1e-8)
            money_ratio_employed = self.policy_m_next[1, :, D_idx, z_idx] / (self.policy_a_next[1, :, D_idx, z_idx] + 1e-8)
            plt.plot(self.a_grid, money_ratio_unemployed, 'b-', label='Unemployed')
            plt.plot(self.a_grid, money_ratio_employed, 'r-', label='Employed')
            plt.title(f'Money-to-Asset Ratio (Skill Level {z_idx})')
            plt.xlabel('Assets')
            plt.ylabel('Money/Assets Ratio')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/money_policies.png')
        plt.close()
        
        # 4. DM Policy Functions by Skill Type
        for z_idx in range(self.n_z):
            plt.figure(figsize=(15, 12))
            
            # Early consumption when only money accepted (ω=0)
            plt.subplot(3, 2, 1)
            a_idx = self.n_a // 2  # Middle of asset grid
            plt.plot(self.m_grid, self.policy_y[0, a_idx, :, z_idx, 0], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_y[1, a_idx, :, z_idx, 0], 'r-', label='Employed')
            plt.title(f'Early Consumption (ω=0, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Consumption in DM')
            plt.legend()
            
            # Early consumption when both assets accepted (ω=1)
            plt.subplot(3, 2, 2)
            plt.plot(self.m_grid, self.policy_y[0, a_idx, :, z_idx, 1], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_y[1, a_idx, :, z_idx, 1], 'r-', label='Employed')
            plt.title(f'Early Consumption (ω=1, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Consumption in DM')
            plt.legend()
            
            # Borrowing when only money accepted (ω=0)
            plt.subplot(3, 2, 3)
            plt.plot(self.m_grid, self.policy_b[0, a_idx, :, z_idx, 0], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_b[1, a_idx, :, z_idx, 0], 'r-', label='Employed')
            plt.title(f'Borrowing (ω=0, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Borrowing Amount')
            plt.legend()
            
            # Borrowing when both assets accepted (ω=1)
            plt.subplot(3, 2, 4)
            plt.plot(self.m_grid, self.policy_b[0, a_idx, :, z_idx, 1], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_b[1, a_idx, :, z_idx, 1], 'r-', label='Employed')
            plt.title(f'Borrowing (ω=1, Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Borrowing Amount')
            plt.legend()
            
            # Deposit policy
            plt.subplot(3, 2, 5)
            plt.plot(self.m_grid, self.policy_d[0, a_idx, :, z_idx], 'b-', label='Unemployed')
            plt.plot(self.m_grid, self.policy_d[1, a_idx, :, z_idx], 'r-', label='Employed')
            plt.title(f'Deposit Policy (Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Deposit Amount')
            plt.legend()
            
            # Deposit-to-money ratio
            plt.subplot(3, 2, 6)
            deposit_ratio_unemployed = self.policy_d[0, a_idx, :, z_idx] / (self.m_grid + 1e-8)
            deposit_ratio_employed = self.policy_d[1, a_idx, :, z_idx] / (self.m_grid + 1e-8)
            plt.plot(self.m_grid, deposit_ratio_unemployed, 'b-', label='Unemployed')
            plt.plot(self.m_grid, deposit_ratio_employed, 'r-', label='Employed')
            plt.title(f'Deposit-to-Money Ratio (Skill Level {z_idx})')
            plt.xlabel('Money Holdings')
            plt.ylabel('Deposit/Money Ratio')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'visualizations/dm_policies_skill_{z_idx}.png')
            plt.close()
        
        # 5. Heat Maps for Policy Functions
        # This helps visualize how policies change across both asset and money dimensions
        for z_idx in range(self.n_z):
            for e_idx in range(self.n_e):
                employment_status = "Employed" if e_idx == 1 else "Unemployed"
                
                # Early consumption heat maps
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # For ω=0
                im0 = axes[0].imshow(
                    self.policy_y[e_idx, :, :, z_idx, 0],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axes[0].set_title(f'Early Consumption (ω=0, {employment_status}, Skill {z_idx})')
                axes[0].set_xlabel('Money Holdings')
                axes[0].set_ylabel('Asset Holdings')
                plt.colorbar(im0, ax=axes[0])
                
                # For ω=1
                im1 = axes[1].imshow(
                    self.policy_y[e_idx, :, :, z_idx, 1],
                    origin='lower',
                    cmap='coolwarm'
                )
                axes[0].set_title(f'Borrowing (ω=0, {employment_status}, Skill {z_idx})')
                axes[0].set_xlabel('Money Holdings')
                axes[0].set_ylabel('Asset Holdings')
                plt.colorbar(im0, ax=axes[0])
                
                # Borrowing for ω=1
                im1 = axes[1].imshow(
                    self.policy_b[e_idx, :, :, z_idx, 1],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='coolwarm'
                )
                axes[1].set_title(f'Borrowing (ω=1, {employment_status}, Skill {z_idx})')
                axes[1].set_xlabel('Money Holdings')
                axes[1].set_ylabel('Asset Holdings')
                plt.colorbar(im1, ax=axes[1])
                
                # Deposits
                im2 = axes[2].imshow(
                    self.policy_d[e_idx, :, :, z_idx],
                    extent=[self.m_grid[0], self.m_grid[-1], self.a_grid[0], self.a_grid[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap='viridis'
                )
                axes[2].set_title(f'Deposits ({employment_status}, Skill {z_idx})')
                axes[2].set_xlabel('Money Holdings')
                axes[2].set_ylabel('Asset Holdings')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(f'visualizations/heatmap_borrowing_deposit_{employment_status}_skill_{z_idx}.png')
                plt.close()

    def diagnostic_pre_dm(self, iter_count):
        """Economic diagnostic checks before DM stage"""
        os.makedirs('diagnostics', exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 1. Check wage distribution
        plt.subplot(2, 2, 1)
        for z_idx in range(self.n_z):
            plt.plot(self.a_grid, self.wages[:, 1, z_idx], label=f'Skill {z_idx}')
        plt.title('Wage Distribution by Skill Level')
        plt.xlabel('Assets')
        plt.ylabel('Wage')
        plt.legend()
        
        # 2. Check resource feasibility
        plt.subplot(2, 2, 2)
        z_idx = 1  # Medium skill
        plt.plot(self.a_grid, self.wages[:, 0, z_idx], label='Unemployed')
        plt.plot(self.a_grid, self.wages[:, 1, z_idx], label='Employed')
        plt.title('Wage Gap (Medium Skill)')
        plt.xlabel('Assets')
        plt.ylabel('Wage')
        plt.legend()
        
        # 3. Value function shape check
        plt.subplot(2, 2, 3)
        D_idx = self.n_D // 2
        for z_idx in range(self.n_z):
            plt.plot(self.a_grid, self.W[1, :, D_idx, z_idx], label=f'Skill {z_idx}')
        plt.title('Value Function by Skill (Employed)')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'diagnostics/pre_dm_iter_{iter_count}.png')
        plt.close()

    def diagnostic_post_dm(self, iter_count):
        """Economic diagnostic checks after DM stage"""
        plt.figure(figsize=(15, 10))
        
        # 1. DM consumption policy check
        z_idx = 1  # Medium skill
        plt.subplot(2, 2, 1)
        a_idx = self.n_a // 2  # Middle of asset grid
        plt.plot(self.m_grid, self.policy_y[0, a_idx, :, z_idx, 0], label='Unemployed ω=0')
        plt.plot(self.m_grid, self.policy_y[1, a_idx, :, z_idx, 0], label='Employed ω=0')
        plt.title('DM Consumption (Money Only)')
        plt.xlabel('Money Holdings')
        plt.ylabel('Consumption')
        plt.legend()
        
        # 2. Money holdings distribution
        plt.subplot(2, 2, 2)
        m_idx = self.n_m // 2
        for z_idx in range(self.n_z):
            plt.plot(self.a_grid, self.V[1, :, m_idx, z_idx], label=f'Skill {z_idx}')
        plt.title('DM Value by Skill Level')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        # 3. Borrowing policy check
        plt.subplot(2, 2, 3)
        plt.plot(self.m_grid, self.policy_b[0, a_idx, :, z_idx, 0], label='Unemployed ω=0')
        plt.plot(self.m_grid, self.policy_b[1, a_idx, :, z_idx, 0], label='Employed ω=0')
        plt.title('Borrowing Policy')
        plt.xlabel('Money Holdings')
        plt.ylabel('Borrowing Amount')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'diagnostics/post_dm_iter_{iter_count}.png')
        plt.close()

    def diagnostic_post_cm(self, iter_count):
        """Economic diagnostic checks after CM stage"""
        plt.figure(figsize=(15, 10))
        
        # 1. Consumption policy check
        z_idx = 1  # Medium skill
        D_idx = self.n_D // 2
        plt.subplot(2, 2, 1)
        plt.plot(self.a_grid, self.policy_c[0, :, D_idx, z_idx], label='Unemployed')
        plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], label='Employed')
        plt.title('CM Consumption Policy')
        plt.xlabel('Assets')
        plt.ylabel('Consumption')
        plt.legend()
        
        # 2. Asset accumulation check
        plt.subplot(2, 2, 2)
        plt.plot(self.a_grid, self.policy_a_next[0, :, D_idx, z_idx], label='Unemployed')
        plt.plot(self.a_grid, self.policy_a_next[1, :, D_idx, z_idx], label='Employed')
        plt.plot(self.a_grid, self.a_grid, 'k--', label='45° line')
        plt.title('Asset Accumulation')
        plt.xlabel('Current Assets')
        plt.ylabel('Next Period Assets')
        plt.legend()
        
        # 3. Budget constraint check
        plt.subplot(2, 2, 3)
        resources = self.wages[:, 1, z_idx]  # Employed resources
        portfolio = (self.policy_a_next[1, :, D_idx, z_idx] / self.prices[1] + 
                    self.policy_m_next[1, :, D_idx, z_idx] * self.prices[2])
        plt.plot(self.a_grid, resources, label='Resources')
        plt.plot(self.a_grid, portfolio, label='Portfolio Cost')
        plt.plot(self.a_grid, self.policy_c[1, :, D_idx, z_idx], label='Consumption')
        plt.title('Budget Constraint Check (Employed)')
        plt.xlabel('Assets')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'diagnostics/post_cm_iter_{iter_count}.png')
        plt.close()


# Example usage
if __name__ == "__main__":
    params = {
        'beta': 0.96,
        'alpha': 0.075,  # Probability of needing to consume in DM
        'alpha_1': 0.06,  # Prob. of case where cash and assets accepted
        'gamma': 1.5, # curvature of late utility
        'Psi': 2.2,  # level of early utility
        'psi': 0.28, # curvature of early utility
        'zeta': 0.75, # curvature of production function
        'nu': 1.6, # matching function curvature
        'mu': 0.7, # share of revenue going to workers
        'delta': 0.035, # job destruction rate
        'kappa': 7.29, # job posting cost
        'repl_rate': 0.4, # fraction of unemployment compensation / working wage income
        'py': 1.0,  # price of early consumption goods
        'Rl': 1.03,  # return on illiquid assets
        'phi_m': 1.0,  # price of money
        'i': 0.02,  # nominal interest rate
        'n_a': 50,
        'n_m': 30,
        'n_z': 3, # number of productivity states
        'n_e': 2, # number of employment states
        'a_min': 0.0,
        'a_max': 15.0,
        'm_min': 0.0,
        'm_max': 5.0,
        'phi_m': 1.0,
        'interest_rate': 0.03,
        'max_iter': 500,
        'tol': 1e-5
    }
    
   # Initialize and solve model
    solver = LagosWrightAiyagariSolver(params)
    
    # Test for uniqueness with different initial guesses
    print("Testing uniqueness with different initial value function guesses...")
    
    # Store initial value function
    original_V = solver.V.copy()
    
    # Solve with original guess
    solver.solve_bellman_iteration()
    solution1 = solver.V.copy()
    
    # Perturb value function and resolve
    print("\nResolving with perturbed value function...")
    solver.V = original_V * 1.2  # 20% higher
    solver.solve_bellman_iteration()
    solution2 = solver.V.copy()
    
    # Check if solutions converge to the same point
    max_diff = np.max(np.abs(solution1 - solution2))
    print(f"\nMaximum difference between solutions: {max_diff:.8f}")
    
    if max_diff < 1e-4:
        print("Value function appears to converge to a unique solution!")
    else:
        print("Warning: Solutions differ - may indicate multiple equilibria or convergence issues")
    
    # Solve for equilibrium prices
    print("\nSolving for general equilibrium...")
    equilibrium = solver.solve_equilibrium(max_price_iter=10)
    
    # Visualize policy functions
    e_mid = solver.n_e // 2  # Middle productivity state
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for m_idx in [0, solver.n_m//2, solver.n_m-1]:
        plt.plot(solver.a_grid, solver.policy_c[e_mid, :, m_idx], 
                 label=f"Money = {solver.m_grid[m_idx]:.2f}")
    plt.title("Consumption Policy")
    plt.xlabel("Assets")
    plt.ylabel("Consumption")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for m_idx in [0, solver.n_m//2, solver.n_m-1]:
        plt.plot(solver.a_grid, solver.policy_a_next[e_mid, :, m_idx], 
                 label=f"Money = {solver.m_grid[m_idx]:.2f}")
    plt.title("Asset Policy")
    plt.xlabel("Assets")
    plt.ylabel("Next Period Assets")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for m_idx in [0, solver.n_m//2, solver.n_m-1]:
        plt.plot(solver.a_grid, solver.policy_m_next[e_mid, :, m_idx], 
                 label=f"Money = {solver.m_grid[m_idx]:.2f}")
    plt.title("Money Policy")
    plt.xlabel("Assets")
    plt.ylabel("Next Period Money")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Simplify to 2D for visualization (sum over money dimension)
    stationary_dist_2d = np.sum(equilibrium['distribution'], axis=2)
    plt.imshow(stationary_dist_2d, aspect='auto', origin='lower',
              extent=[solver.a_grid[0], solver.a_grid[-1], 
                     solver.e_grid[0], solver.e_grid[-1]])
    plt.colorbar(label="Mass")
    plt.title("Stationary Distribution")
    plt.xlabel("Assets")
    plt.ylabel("Productivity")
    
    plt.tight_layout()
    plt.show()
