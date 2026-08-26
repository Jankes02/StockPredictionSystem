    def _decide_for_symbol(self, symbol: str, signals: List[dict]) -> Decision:
        filtered = [
            s for s in signals
            if s["signal"] != 0 and s["confidence"] >= self.min_confidence
        ]

        if not filtered:
            return self._hold(symbol)

        trend_dir = self._trend_direction(signals)
        score = 0.0
        contributors = []

        for s in filtered:
            if self.mode == "passive" and trend_dir != 0:
                if s["signal"] != trend_dir:
                    continue

            weight = self.kind_weights.get(s["kind"], 1.0)
            contribution = s["signal"] * s["confidence"] * weight
            score += contribution
            contributors.append({
                "agent": s["agent"],
                "signal": s["signal"],
                "confidence": s["confidence"],
                "kind": s["kind"],
            })

        thresholds = self.thresholds[self.mode]

        if score >= thresholds["buy"]:
            return self._decision(symbol, "BUY", score, contributors)
        elif score <= thresholds["sell"]:
            return self._decision(symbol, "SELL", score, contributors)
        else:
            return self._hold(symbol)
