import json
import os
from datetime import datetime

DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")

def load_portfolio():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
                
    default_state = {
        "cash": 100000.0,
        "positions": {},  
        "history": []     
    }
    save_portfolio(default_state)
    return default_state

def save_portfolio(state):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(state, f, indent=4)

def buy_asset(symbol: str, quantity: float, price: float) -> tuple[bool, str]:
    """Executes a BUY order (Open Long or Close Short)."""
    if quantity <= 0:
        return False, "Quantity must be greater than 0."
        
    state = load_portfolio()
    cost = quantity * price
    pos = state["positions"].get(symbol, {"quantity": 0.0, "average_price": 0.0})
    
    # Check if we are short
    if pos["quantity"] < 0:
        # Closing a Short
        if quantity > abs(pos["quantity"]):
            return False, "Cannot flip from Short to Long in one trade. Close exactly, then open new Long."
            
        # Calculate Realized PnL: (Entry - Exit) * quantity_closed
        realized_pnl = (pos["average_price"] - price) * quantity
        
        # Cash goes down by cost, but up by the realized PnL + original cost basis.
        # Actually, when we shorted, we received cash = entry_price * qty.
        # Now we buy back, spending cash = exit_price * qty.
        # So we just subtract the cost from cash.
        if state["cash"] < cost:
             return False, f"Insufficient funds to buy back short. Required: ${cost:,.2f}"
             
        state["cash"] -= cost
        pos["quantity"] += quantity # Moves closer to 0
        
        state["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": "BUY (Close Short)",
            "quantity": quantity,
            "price": price,
            "total": cost,
            "realized_pnl": realized_pnl
        })
        
    else:
        # Opening/Adding to Long
        if state["cash"] < cost:
            return False, f"Insufficient funds. Required: ${cost:,.2f}, Available: ${state['cash']:,.2f}"
            
        state["cash"] -= cost
        
        # Calculate new average price
        total_value = (pos["quantity"] * pos["average_price"]) + cost
        new_quantity = pos["quantity"] + quantity
        pos["average_price"] = total_value / new_quantity
        pos["quantity"] = new_quantity
        
        state["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": "BUY (Open Long)",
            "quantity": quantity,
            "price": price,
            "total": cost,
            "realized_pnl": 0.0
        })
        
    # Clean up zero positions
    if abs(pos["quantity"]) <= 1e-8:
        if symbol in state["positions"]:
            del state["positions"][symbol]
    else:
        state["positions"][symbol] = pos
        
    save_portfolio(state)
    return True, f"Successfully executed BUY for {quantity} {symbol} at ${price:,.2f}."

def sell_asset(symbol: str, quantity: float, price: float) -> tuple[bool, str]:
    """Executes a SELL order (Open Short or Close Long)."""
    if quantity <= 0:
        return False, "Quantity must be greater than 0."
        
    state = load_portfolio()
    revenue = quantity * price
    pos = state["positions"].get(symbol, {"quantity": 0.0, "average_price": 0.0})
    
    # Check if we are long
    if pos["quantity"] > 0:
        # Closing a Long
        if quantity > pos["quantity"]:
            return False, "Cannot flip from Long to Short in one trade. Close exactly, then open new Short."
            
        # Calculate Realized PnL: (Exit - Entry) * quantity_closed
        cost_basis = quantity * pos["average_price"]
        realized_pnl = revenue - cost_basis
        
        state["cash"] += revenue
        pos["quantity"] -= quantity # Moves closer to 0
        
        state["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": "SELL (Close Long)",
            "quantity": quantity,
            "price": price,
            "total": revenue,
            "realized_pnl": realized_pnl
        })
        
    else:
        # Opening/Adding to Short
        # Add revenue to cash
        state["cash"] += revenue
        
        # Calculate new average short price
        # Treat as absolute quantities for average price math
        current_abs_qty = abs(pos["quantity"])
        total_value = (current_abs_qty * pos["average_price"]) + revenue
        new_abs_quantity = current_abs_qty + quantity
        pos["average_price"] = total_value / new_abs_quantity
        pos["quantity"] -= quantity # Moves further negative
        
        state["history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": "SELL (Open Short)",
            "quantity": quantity,
            "price": price,
            "total": revenue,
            "realized_pnl": 0.0
        })
        
    # Clean up zero positions
    if abs(pos["quantity"]) <= 1e-8:
        if symbol in state["positions"]:
            del state["positions"][symbol]
    else:
        state["positions"][symbol] = pos
        
    save_portfolio(state)
    return True, f"Successfully executed SELL for {quantity} {symbol} at ${price:,.2f}."

def get_portfolio_summary(current_prices: dict) -> dict:
    state = load_portfolio()
    total_value = state["cash"]
    unrealized_pnl = 0.0
    active_positions = []
    
    for sym, pos in state["positions"].items():
        curr_price = current_prices.get(sym, pos["average_price"])
        qty = pos["quantity"]
        abs_qty = abs(qty)
        
        if qty > 0:
            # Long position
            current_val = qty * curr_price
            cost_basis = qty * pos["average_price"]
            pnl = current_val - cost_basis
        else:
            # Short position
            # Current value is a liability (we owe this much to buy it back)
            current_val = -1 * (abs_qty * curr_price)
            # Cost basis is the cash we originally received for shorting
            cost_basis = abs_qty * pos["average_price"]
            # PnL = (Entry - Exit) * Qty
            pnl = (pos["average_price"] - curr_price) * abs_qty
            
        total_value += current_val
        unrealized_pnl += pnl
        
        position_type = "LONG" if qty > 0 else "SHORT"
        
        active_positions.append({
            "Symbol": sym,
            "Type": position_type,
            "Quantity": abs_qty,
            "Avg Price": pos["average_price"],
            "Current Price": curr_price,
            "Current Value": current_val,
            "Unrealized PnL": pnl,
            "PnL %": (pnl / cost_basis * 100) if cost_basis > 0 else 0
        })
        
    return {
        "cash": state["cash"],
        "total_value": total_value,
        "unrealized_pnl": unrealized_pnl,
        "active_positions": active_positions,
        "history": state["history"]
    }
