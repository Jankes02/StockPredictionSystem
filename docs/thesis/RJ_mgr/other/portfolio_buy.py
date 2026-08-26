
        # Size the position against cash capacity inclusive of entry charges
        allocation = self.cash * self.position_size
        max_spend = allocation / (1.0 + buy_charge_rate)
        if max_spend <= 0:
            return

        quantity = int(max_spend // fill_price)
        if quantity <= 0:
            return

        notional = quantity * fill_price
        entry_charge = notional * buy_charge_rate
        total_cost = notional + entry_charge
        if total_cost > self.cash:
            return

        self.cash -= total_cost

        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_date": date,
            "entry_commission": entry_charge,
        }

        self.trades.append({
            "symbol": symbol,
            "action": "BUY",
            "price": fill_price,
            "quantity": quantity,
            "date": date,
            "pnl": None,
        })

    def _sell(self, symbol: str, price: Optional[float], date: pd.Timestamp) -> None:
        if price is None:
            return
        if symbol not in self.positions:
            return
