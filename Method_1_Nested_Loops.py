# Import Packages
import numpy as np

def binomial_option_pricing(S0, K, T, N, r, u):
    # Calculate down factor (d) based on up factor (u)
    d = np.round(1/u, 1)

    dt = T / N                           # delta T
    p = (np.exp(r * dt) - d) / (u - d)   # Risk-neutral probability
    q = 1 - p                            # Risk-neutral probability

    # Create N x N Matrix to hold stock prices
    stock_price = np.zeros([N+1, N+1])

    # Fill the matrix with stock prices
    for col in range(len(stock_price)):
        for row in range(col+1):
            stock_price[row, col] = S0 * (u**(col-row)) * (d**row)

    # Create N x N Matrix to hold option values
    option_payoff = np.zeros([N + 1, N + 1])

    # Determine option's payoff at the end of the nodes
    option_payoff[:, N] = np.maximum(np.zeros(N + 1), (stock_price[:, N] - K))

    # Determine option's payoff using backward induction, discounted by risk-free rate
    for col in range(N - 1, -1, -1):
        for row in range(0, col + 1):
            option_payoff[row, col] = (
                np.exp(-r * dt) * (p * option_payoff[row, col + 1] + q * option_payoff[row + 1, col + 1])
            )

    option_price = option_payoff[0, 0].round(3)
    
    return stock_price, option_payoff, option_price





# Example usage:
S0 = 100    # Current stock price
K  = 105    # Strike price
T  = 3/12   # Time to maturity in years
N  = 1      # Number of steps
r  = 0.04   # Annual risk-free rate
u  = 1.1    # Up factor

stock_price, option_payoff, option_price = binomial_option_pricing(S0, K, T, N, r, u)

print('Stock Price Paths:')
print(stock_price)

print('\nOption Payoff Paths:')
print(option_payoff)

print('\nOption Price:')
print(option_price)





















