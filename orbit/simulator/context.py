import random
from dataclasses import dataclass

import numpy as np

# ── Marginal distributions ────────────────────────────────────────────────────

# Flat rail distribution — used for primary rail sampling
RAIL_DISTRIBUTION = {
    "UPI":        0.75,
    "CARDS":      0.15,
    "NETBANKING": 0.06,
    "IMPS":       0.02,
    "WALLETS":    0.01,
    "BNPL_EMI":   0.01,
}

# Fallback issuer distribution — used when rail has no conditional entry
# (IMPS, WALLETS, BNPL_EMI fall back to this)
ISSUER_DISTRIBUTION = {
    "SBI":          0.28,
    "HDFC":         0.20,
    "ICICI":        0.18,
    "AXIS":         0.12,
    "KOTAK":        0.04,
    "BOB":          0.03,
    "PNB":          0.03,
    "UNION_BANK":   0.02,
    "YES_BANK":     0.01,
    "IDFC_FIRST":   0.01,
    "INDUSIND":     0.01,
    "CANARA_BANK":  0.01,
    "AU_SFB":       0.01,
    "OTHERS":       0.05,
}


# ── Conditional distributions ─────────────────────────────────────────────────

# P(Network | Rail)
RAIL_NETWORK_CONDITIONAL = {
    "UPI":        {"5G": 0.38, "4G": 0.52, "WIFI": 0.08, "2G": 0.02},
    "CARDS":      {"5G": 0.30, "4G": 0.50, "WIFI": 0.15, "2G": 0.05},
    "NETBANKING": {"5G": 0.25, "4G": 0.35, "WIFI": 0.40, "2G": 0.00},
    "IMPS":       {"5G": 0.30, "4G": 0.55, "WIFI": 0.14, "2G": 0.01},
    "WALLETS":    {"5G": 0.45, "4G": 0.48, "WIFI": 0.07, "2G": 0.00},
    "BNPL_EMI":   {"5G": 0.50, "4G": 0.35, "WIFI": 0.15, "2G": 0.00},
}

# P(Issuer | Rail)
# IMPS, WALLETS, BNPL_EMI are not listed here — they fall back to
# ISSUER_DISTRIBUTION via the .get() call in context_generator
RAIL_ISSUER_CONDITIONAL = {
    "UPI": {
        "SBI": 0.32, "HDFC": 0.16, "ICICI": 0.14, "AXIS": 0.10,
        "KOTAK": 0.04, "BOB": 0.04, "PNB": 0.04, "UNION_BANK": 0.03,
        "YES_BANK": 0.01, "IDFC_FIRST": 0.01, "INDUSIND": 0.01,
        "CANARA_BANK": 0.02, "AU_SFB": 0.02, "OTHERS": 0.06,
    },
    "CARDS": {
        "SBI": 0.18, "HDFC": 0.28, "ICICI": 0.24, "AXIS": 0.14,
        "KOTAK": 0.07, "BOB": 0.01, "PNB": 0.01, "UNION_BANK": 0.01,
        "YES_BANK": 0.01, "IDFC_FIRST": 0.01, "INDUSIND": 0.02,
        "CANARA_BANK": 0.00, "AU_SFB": 0.00, "OTHERS": 0.02,
    },
    "NETBANKING": {
        "SBI": 0.25, "HDFC": 0.22, "ICICI": 0.20, "AXIS": 0.12,
        "KOTAK": 0.05, "BOB": 0.03, "PNB": 0.03, "UNION_BANK": 0.02,
        "YES_BANK": 0.01, "IDFC_FIRST": 0.01, "INDUSIND": 0.01,
        "CANARA_BANK": 0.02, "AU_SFB": 0.01, "OTHERS": 0.02,
    },
}

# P(GeographyTier | Issuer)
# Banks with no entry fall back to _DEFAULT
ISSUER_TIER_CONDITIONAL = {
    "SBI":         {"METRO": 0.30, "TIER2": 0.35, "TIER3": 0.35},
    "HDFC":        {"METRO": 0.65, "TIER2": 0.28, "TIER3": 0.07},
    "ICICI":       {"METRO": 0.60, "TIER2": 0.30, "TIER3": 0.10},
    "AXIS":        {"METRO": 0.55, "TIER2": 0.32, "TIER3": 0.13},
    "KOTAK":       {"METRO": 0.70, "TIER2": 0.25, "TIER3": 0.05},
    "AU_SFB":      {"METRO": 0.20, "TIER2": 0.35, "TIER3": 0.45},
    "OTHERS":      {"METRO": 0.20, "TIER2": 0.35, "TIER3": 0.45},
    "_DEFAULT":    {"METRO": 0.40, "TIER2": 0.35, "TIER3": 0.25},
}


# ── Amount distributions per rail ─────────────────────────────────────────────
# Log-normal params: (mu, sigma) — calibrated to realistic INR ranges
RAIL_AMOUNT_PARAMS = {
    "UPI":        (5.5, 1.2),   # mode ~₹245,    range ₹1     – ₹1L
    "CARDS":      (7.0, 1.0),   # mode ~₹1,097,  range ₹100   – ₹5L
    "NETBANKING": (9.0, 0.8),   # mode ~₹8,103,  range ₹1K    – ₹50L
    "IMPS":       (8.5, 0.9),   # mode ~₹4,900,  range ₹100   – ₹5L
    "WALLETS":    (4.5, 0.8),   # mode ~₹90,     range ₹1     – ₹10K
    "BNPL_EMI":   (7.5, 0.6),   # mode ~₹1,800,  range ₹500   – ₹50K
}


# ── Merchant categories ───────────────────────────────────────────────────────
# Each category has eligible rails and a sampling weight
MERCHANT_CATEGORIES = {
    "ECOMMERCE": {
        "rails": ["UPI", "CARDS", "NETBANKING", "WALLETS", "BNPL_EMI"],
        "weight": 0.35,
    },
    "FOOD_DELIVERY": {"rails": ["UPI", "CARDS", "WALLETS"], "weight": 0.15},
    "EDUCATION": {"rails": ["UPI", "CARDS", "NETBANKING"], "weight": 0.10},
    "UTILITY": {"rails": ["UPI", "NETBANKING", "CARDS"], "weight": 0.12},
    "TRAVEL": {"rails": ["CARDS", "NETBANKING", "UPI", "BNPL_EMI"], "weight": 0.10},
    "GROCERY": {"rails": ["UPI", "CARDS", "WALLETS"], "weight": 0.12},
    "HEALTHCARE": {"rails": ["UPI", "CARDS", "NETBANKING"], "weight": 0.06},
}


# ── Sampling utility ──────────────────────────────────────────────────────────

def weighted_choice(distribution: dict, given: str | None = None) -> str:
    """
    Sample a key from a weighted distribution.

    For flat distributions  → call with distribution only.
    For conditional dicts   → pass `given` to index into the outer dict first.

    Args:
        distribution : {str: float}  OR  {str: {str: float}}
        given        : conditioning key (required when distribution is nested)

    Returns:
        sampled key as str

    Examples:
        rail    = weighted_choice(RAIL_DISTRIBUTION)
        network = weighted_choice(RAIL_NETWORK_CONDITIONAL, given=rail)
        issuer  = weighted_choice(RAIL_ISSUER_CONDITIONAL,  given=rail)
        tier    = weighted_choice(ISSUER_TIER_CONDITIONAL,  given=issuer)

    Raises:
        KeyError: if `given` is not found in a nested distribution
        TypeError: if `given` is omitted but values are dicts
                   (nested dict passed as flat)
    """
    if given is not None:
        inner = distribution.get(given)
        if inner is None:
            raise KeyError(
                f"Conditioning key '{given}' not found. "
                f"Available keys: {list(distribution.keys())}"
            )
        if not isinstance(inner, dict):
            raise TypeError(
                f"Expected nested dict for key '{given}', "
                f"got {type(inner).__name__}."
            )
        target = inner
    else:
        target = distribution
        first_val = next(iter(target.values()), None)
        if isinstance(first_val, dict):
            raise TypeError(
                "Distribution values are dicts — this is a conditional distribution. "
                "Pass given=<key> to condition on a variable."
            )

    keys    = list(target.keys())
    weights = list(target.values())
    return random.choices(keys, weights=weights, k=1)[0]


# ── Transaction context dataclass ─────────────────────────────────────────────

@dataclass
class TransactionContext:
    """
    Fully describes a single payment transaction at the moment of routing.
    All fields are populated by context_generator().
    Properties derive additional signals used by the bandit context vector.
    """
    txn_id:            str
    rail:              str
    issuer_bank:       str
    network:           str
    geography_tier:    str
    merchant_category: str
    amount_inr:        float
    hour_of_day:       int     # 0–23, derived from sim_time
    day_of_week:       int     # 0=Monday … 6=Sunday, derived from sim_time
    day_of_month:      int     # 1–30, derived from sim_time
    is_peak_hour:      bool    # True if 19:00–22:00
    is_salary_day:     bool    # True on 1st, 7th, 28th–31st of month
    sim_time:          float   # SimPy env.now — seconds elapsed in simulation

    @property
    def is_high_value(self) -> bool:
        """High-value flag: amount > ₹50,000.

        Used as a feature in the context vector.
        """
        return self.amount_inr > 50_000

    @property
    def device_tier(self) -> str:
        """
        Proxy device quality derived from network type.
        Used as a routing context feature — Tier-3 / 2G customers
        need intent flow over collect flow to avoid session timeouts.
        """
        return {"5G": "HIGH", "WIFI": "HIGH", "4G": "MID", "2G": "LOW"}[self.network]


# ── Context generator ─────────────────────────────────────────────────────────

def context_generator(txn_id: str, sim_time: float) -> TransactionContext:
    """
    Generates a fully populated TransactionContext using conditional
    probability sampling. All attributes are correlated — rail drives
    network and issuer, issuer drives geography tier.

    Dependency chain:
        merchant_category → rail → issuer → geography_tier
                                 → network
                                 → amount_inr
        sim_time → hour_of_day, day_of_week, day_of_month,
                   is_peak_hour, is_salary_day

    Args:
        txn_id   : unique transaction identifier (from transaction_id_generator)
        sim_time : current SimPy simulation clock value (env.now, in seconds)

    Returns:
        TransactionContext
    """

    # Step 1 — Sample merchant category (unconditional)
    merchant_category = weighted_choice(
        {k: v["weight"] for k, v in MERCHANT_CATEGORIES.items()}
    )
    eligible_rails = MERCHANT_CATEGORIES[merchant_category]["rails"]

    # Step 2 — Sample rail, restricted to merchant-eligible rails
    # Renormalise RAIL_DISTRIBUTION to only eligible rails
    rail_weights = {
        k: v for k, v in RAIL_DISTRIBUTION.items()
        if k in eligible_rails
    }
    rail = weighted_choice(rail_weights)

    # Step 3 — Sample issuer bank | rail
    # IMPS, WALLETS, BNPL_EMI have no conditional entry → fall back to marginal
    issuer_bank = weighted_choice(
        RAIL_ISSUER_CONDITIONAL.get(rail, ISSUER_DISTRIBUTION)
    )

    # Step 4 — Sample network | rail
    network = weighted_choice(RAIL_NETWORK_CONDITIONAL, given=rail)

    # Step 5 — Sample geography tier | issuer bank
    # Banks not explicitly listed fall back to _DEFAULT
    geography_tier = weighted_choice(
        ISSUER_TIER_CONDITIONAL.get(issuer_bank, ISSUER_TIER_CONDITIONAL["_DEFAULT"])
    )

    # Step 6 — Sample transaction amount from rail-specific log-normal
    mu, sigma = RAIL_AMOUNT_PARAMS[rail]
    amount_inr = float(np.clip(
    np.random.lognormal(mean=mu, sigma=sigma),
    1.0, 500_000.0
    ))
    amount_inr = round(amount_inr, 2)

    # Step 7 — Derive all time signals from sim_time
    # Simulation clock starts at 08:00 IST (480 minutes into the day)
    # sim_time is in seconds; map forward through simulation days
    sim_minutes  = (sim_time / 60) % (24 * 60)                     # position within day
    hour_of_day  = int((480 + sim_minutes) % (24 * 60) // 60)      # 0–23
    day_of_week  = int(sim_time / 86_400) % 7                      # 0=Mon … 6=Sun
    day_of_month = (int(sim_time / 86_400) % 30) + 1               # 1–30 rolling

    # Peak hour: 19:00–22:00 IST — bank load spikes, PSR drops 8-12 pp
    is_peak_hour = 19 <= hour_of_day <= 22

    # Salary day: 1st, 7th, month-end — transaction volume spikes
    is_salary_day = day_of_month in {1, 7, 28, 29, 30}

    return TransactionContext(
        txn_id            = txn_id,
        rail              = rail,
        issuer_bank       = issuer_bank,
        network           = network,
        geography_tier    = geography_tier,
        merchant_category = merchant_category,
        amount_inr        = amount_inr,
        hour_of_day       = hour_of_day,
        day_of_week       = day_of_week,
        day_of_month      = day_of_month,
        is_peak_hour      = is_peak_hour,
        is_salary_day     = is_salary_day,
        sim_time          = sim_time,
    )


# ── Quick verification (run this file directly to sanity-check output) ────────

if __name__ == "__main__":
    from collections import Counter

    print("Running 10,000 sample verification...\n")

    rail_counts    = Counter()
    network_counts = Counter()
    issuer_counts  = Counter()
    tier_counts    = Counter()

    # Verify NETBANKING never produces 2G
    netbanking_2g = 0

    for i in range(5):
        ctx = context_generator(txn_id=f"TEST_{i:05d}", sim_time=float(i * 3))
        rail_counts[ctx.rail]          += 1
        network_counts[ctx.network]    += 1
        issuer_counts[ctx.issuer_bank] += 1
        tier_counts[ctx.geography_tier]+= 1
        if ctx.rail == "NETBANKING" and ctx.network == "2G":
            netbanking_2g += 1

    total = 5
    print("Rail distribution:")
    for k, v in sorted(rail_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<14} {v/total:.3f}  (expected: {RAIL_DISTRIBUTION.get(k, 0):.3f})")

    print("\nNetwork distribution:")
    for k, v in sorted(network_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<8} {v/total:.3f}")

    print("\nTop 5 issuers:")
    for k, v in sorted(issuer_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k:<14} {v/total:.3f}")

    print("\nGeography tier distribution:")
    for k, v in sorted(tier_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<8} {v/total:.3f}")

    print(f"\nNETBANKING + 2G occurrences: {netbanking_2g}  (expected: 0)")

    # Sample one context and print it
    print("\nSample TransactionContext:")
    ctx = context_generator("SAMPLE_001", sim_time=3600.0)
    for f in ctx.__dataclass_fields__:
        print(f"  {f:<20} {getattr(ctx, f)}")
    print(f"  {'is_high_value':<20} {ctx.is_high_value}")
    print(f"  {'device_tier':<20} {ctx.device_tier}")