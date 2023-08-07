import numpy as np

def binomial_option_pricing(S0, K, T, N, r, u):
    # Calculate down factor (d) based on up factor (u)
    d = np.round(1/u, 1)

    dt = T / N                           # delta T
    p = (np.exp(r * dt) - d) / (u - d)   # Risk-neutral probability
    q = 1 - p                            # Risk-neutral probability
    
    # Create N x N matrix to hold stock prices and option payoff
    stock_price = np.zeros([N+1, N+1])
    option_payoff = np.zeros([N+1, N+1])

    # Calculate stock prices and at the end of the nodes
    stock_price[:, N] = S0 * (u**np.arange(N, -1, -1)) * (d**np.arange(N + 1))

    # Calculate option payoff at the end of the the nodes
    option_payoff[:, N] = np.maximum(np.zeros(N + 1), (stock_price[:, N] - K))

    # Calculate stock price and option_payoff using backward induction
    for col in range(N - 1, -1, -1):
        # 0:col+1 = row
        stock_price[0:col+1, col] = stock_price[0:col+1, col+1] / u
        
        option_payoff[0:col+1, col] = (
            np.exp(-r * dt) * ((p * option_payoff[0:col+1, col + 1]) + (q * option_payoff[1:col+2, col + 1]))
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