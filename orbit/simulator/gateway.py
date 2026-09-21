from enum import Enum
from dataclasses import dataclass
import random
import numpy as np
import simpy


# ── Enums ─────────────────────────────────────────────────────────────────────

class BankState(Enum):
    HEALTHY  = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN     = "DOWN"


class BankTier(Enum):
    LARGE_PRIVATE = "LARGE_PRIVATE"   # HDFC, ICICI, AXIS, KOTAK
    LARGE_PSU     = "LARGE_PSU"       # SBI, BOB, PNB, UNION_BANK, CANARA_BANK
    MID           = "MID"             # YES_BANK, IDFC_FIRST, INDUSIND
    SFB           = "SFB"             # AU_SFB, OTHERS


# ── BankConfiguration ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BankConfiguration:
    name:                    str
    capacity:                int
    latency_mu:              float   # log-space: e^5.2 ≈ 181ms, e^5.8 ≈ 330ms
    latency_sigma:           float   # log-space std dev
    tier:                    BankTier
    outage_rate:             float   # outages per day
    mean_outage_duration:    float   # minutes
    degradation_probability: float   # P(DEGRADED) when outage fires, else DOWN
    degraded_base_psr:       float   # PSR floor when in DEGRADED state


# ── All 14 banks — keys must exactly match ISSUER_DISTRIBUTION in context.py ──

BANK_CONFIGURATIONS: dict[str, BankConfiguration] = {

    # ── LARGE_PRIVATE ─────────────────────────────────────────────────────────

    "HDFC": BankConfiguration(
        name                    = "HDFC",
        capacity                = 1000,
        latency_mu              = 5.2,    # e^5.2 ≈ 181ms
        latency_sigma           = 0.28,
        tier                    = BankTier.LARGE_PRIVATE,
        outage_rate             = 0.25,   # ~1 outage every 4 days
        mean_outage_duration    = 5.0,    # minutes
        degradation_probability = 0.75,   # 75% DEGRADED, 25% full DOWN
        degraded_base_psr       = 0.82,
    ),

    "ICICI": BankConfiguration(
        name                    = "ICICI",
        capacity                = 900,
        latency_mu              = 5.3,    # e^5.3 ≈ 200ms
        latency_sigma           = 0.30,
        tier                    = BankTier.LARGE_PRIVATE,
        outage_rate             = 0.30,
        mean_outage_duration    = 6.0,
        degradation_probability = 0.70,
        degraded_base_psr       = 0.79,
    ),

    "AXIS": BankConfiguration(
        name                    = "AXIS",
        capacity                = 800,
        latency_mu              = 5.4,    # e^5.4 ≈ 221ms
        latency_sigma           = 0.32,
        tier                    = BankTier.LARGE_PRIVATE,
        outage_rate             = 0.35,
        mean_outage_duration    = 6.0,
        degradation_probability = 0.65,
        degraded_base_psr       = 0.76,
    ),

    "KOTAK": BankConfiguration(
        name                    = "KOTAK",
        capacity                = 600,
        latency_mu              = 5.3,
        latency_sigma           = 0.29,
        tier                    = BankTier.LARGE_PRIVATE,
        outage_rate             = 0.30,
        mean_outage_duration    = 5.5,
        degradation_probability = 0.70,
        degraded_base_psr       = 0.78,
    ),

    # ── LARGE_PSU ─────────────────────────────────────────────────────────────

    "SBI": BankConfiguration(
        name                    = "SBI",
        capacity                = 1500,   # highest volume bank
        latency_mu              = 5.5,    # e^5.5 ≈ 245ms
        latency_sigma           = 0.35,
        tier                    = BankTier.LARGE_PSU,
        outage_rate             = 1.00,   # ~1 outage per day
        mean_outage_duration    = 10.0,
        degradation_probability = 0.50,   # equal chance DEGRADED vs DOWN
        degraded_base_psr       = 0.71,
    ),

    "BOB": BankConfiguration(
        name                    = "BOB",
        capacity                = 400,
        latency_mu              = 5.7,    # e^5.7 ≈ 299ms
        latency_sigma           = 0.40,
        tier                    = BankTier.LARGE_PSU,
        outage_rate             = 1.20,
        mean_outage_duration    = 12.0,
        degradation_probability = 0.50,
        degraded_base_psr       = 0.65,
    ),

    "PNB": BankConfiguration(
        name                    = "PNB",
        capacity                = 350,
        latency_mu              = 5.8,
        latency_sigma           = 0.42,
        tier                    = BankTier.LARGE_PSU,
        outage_rate             = 1.30,
        mean_outage_duration    = 12.0,
        degradation_probability = 0.45,
        degraded_base_psr       = 0.63,
    ),

    "UNION_BANK": BankConfiguration(
        name                    = "UNION_BANK",
        capacity                = 300,
        latency_mu              = 5.9,
        latency_sigma           = 0.44,
        tier                    = BankTier.LARGE_PSU,
        outage_rate             = 1.40,
        mean_outage_duration    = 13.0,
        degradation_probability = 0.45,
        degraded_base_psr       = 0.62,
    ),

    "CANARA_BANK": BankConfiguration(       # NOTE: CANARA_BANK not CANARA
        name                    = "CANARA_BANK",
        capacity                = 300,
        latency_mu              = 5.8,
        latency_sigma           = 0.43,
        tier                    = BankTier.LARGE_PSU,
        outage_rate             = 1.35,
        mean_outage_duration    = 12.0,
        degradation_probability = 0.45,
        degraded_base_psr       = 0.62,
    ),

    # ── MID ───────────────────────────────────────────────────────────────────

    "YES_BANK": BankConfiguration(
        name                    = "YES_BANK",
        capacity                = 250,
        latency_mu              = 5.6,
        latency_sigma           = 0.38,
        tier                    = BankTier.MID,
        outage_rate             = 0.80,
        mean_outage_duration    = 8.0,
        degradation_probability = 0.55,
        degraded_base_psr       = 0.68,
    ),

    "IDFC_FIRST": BankConfiguration(
        name                    = "IDFC_FIRST",
        capacity                = 250,
        latency_mu              = 5.5,
        latency_sigma           = 0.35,
        tier                    = BankTier.MID,
        outage_rate             = 0.70,
        mean_outage_duration    = 7.5,
        degradation_probability = 0.60,
        degraded_base_psr       = 0.72,
    ),

    "INDUSIND": BankConfiguration(
        name                    = "INDUSIND",
        capacity                = 280,
        latency_mu              = 5.6,
        latency_sigma           = 0.36,
        tier                    = BankTier.MID,
        outage_rate             = 0.75,
        mean_outage_duration    = 8.0,
        degradation_probability = 0.58,
        degraded_base_psr       = 0.70,
    ),

    # ── SFB ───────────────────────────────────────────────────────────────────

    "AU_SFB": BankConfiguration(
        name                    = "AU_SFB",
        capacity                = 150,
        latency_mu              = 6.0,    # e^6.0 ≈ 403ms — slowest
        latency_sigma           = 0.50,
        tier                    = BankTier.SFB,
        outage_rate             = 2.00,   # ~2 outages per day
        mean_outage_duration    = 18.0,
        degradation_probability = 0.40,   # more likely to go fully DOWN
        degraded_base_psr       = 0.58,
    ),

    "OTHERS": BankConfiguration(
        name                    = "OTHERS",
        capacity                = 100,
        latency_mu              = 6.2,    # e^6.2 ≈ 493ms
        latency_sigma           = 0.55,
        tier                    = BankTier.SFB,
        outage_rate             = 2.50,
        mean_outage_duration    = 20.0,
        degradation_probability = 0.35,
        degraded_base_psr       = 0.55,
    ),
}


class BankNode: 

    _TIER_PARAMETERS = {
       BankTier.LARGE_PRIVATE: (72.0,  6.0,  0.70),
       BankTier.LARGE_PSU:     (24.0,  12.0, 0.50),
       BankTier.MID:           (48.0,  8.0,  0.55),
       BankTier.SFB:           (12.0,  18.0, 0.40),
    }

    _NETWORK_MOD = {
        "5G": 1.0, "WIFI" : 1.0 , "4G": 0.98 , "2G": 0.88
    }


    
    def __init__(self, env: simpy.Environment, config: BankConfiguration):
        self.env      = env
        self.config   = config
        self.state    = BankState.HEALTHY
        self.resource = simpy.Resource(env, capacity=config.capacity)
        self._proc    = env.process(self.simulate_outages())  

    def simulate_outages(self):
        mtbo_hours , dur_minutes , p_deg = self._TIER_PARAMETERS[self.config.tier]
        rate_of_outages_per_second = 1/(mtbo_hours * 3600)

        while True : #wait till the next outage occurs 
            yield self.env.timeout(random.expovariate(rate_of_outages_per_second))

            self.state = BankState.DEGRADED if random.random() < p_deg else BankState.DOWN


#Here it simulates the duration of the outage and then returns to healthy state
            yield self.env.timeout(
                random.expovariate(1.0 / (dur_minutes * 60))
            )
            self.state = BankState.HEALTHY

    def effective_psr(self, network_type: str) -> float:
        """
        Returns the effective PSR (Payment Success Rate) of the bank node based on its current state and network type.
         Pure function. No yield. Reads state + time → returns float.
        """
        if self.state == BankState.DOWN:
            return 0.0

        base = (self.config.degraded_base_psr 
                if self.state == BankState.DEGRADED 
                else 1.0)

        
        sim_min  = (self.env.now / 60) % (24 * 60)
        hour     = int((480 + sim_min) % (24 * 60) // 60)
        peak_mod = 0.92 if 19 <= hour <= 22 else 1.0

        return base * peak_mod * self._NETWORK_MOD.get(network_type, 0.95)

    def _sample_latency(self) -> float: 
        """
        Returns a latency sample in seconds based on the bank's log-normal latency distribution.
        Pure function. No yield. Reads config → returns float.
        """
        mu = self.config.latency_mu 
        #Why 0.6 for mu? In a log-normal distribution, 
        #adding 0.6 to mu multiplies the median latency 
        # by e^0.6 (approximately 1.82). This means that during a degraded state, 
        # the baseline processing time becomes roughly 82% slower. 
        # If a healthy bank usually takes 100ms, a degraded one will hover around 182ms.
        
        sigma = self.config.latency_sigma
        # Why 0.15 for sigma? This increases the variance in the log space. 
        # While 0.15 seems small, it exponentially widens the "long tail" of the distribution. 
        # This simulates the jitter of a struggling server, generating occasional extreme outliers 
        # (e.g., a transaction randomly hanging for 3000ms before succeeding).
       

        mu += 0.6 
        sigma += 0.15

        return float(np.random.lognormal(mean=mu, sigma=sigma))
    def _classify_errors(self, network: str) -> str :
         """Classifies the type of error based on the bank's state and network type.
        Pure function. No yield. Reads state + network → returns str.
         U28 → bank DOWN (unreachable)
        U30 → debit failed, bank-side (routing-relevant)
        U69 → session timeout (network/latency)
        Z9  → insufficient funds (customer-caused — classifier discards this)
        Z7  → rate limit exceeded
        """
         if self.state == BankState.DOWN:
            return "U28"    
         if self.state == BankState.DEGRADED:
            return "U30"
         if network == "2G":   return "U69"    

         return random.choices(["Z9", "Z7", "U69", "U30"], 
                               weights=[0.45, 0.25, 0.20, 0.10])[0]

    def process_transaction(self, ctx) :
        """
        Simulates processing a transaction through the bank node.
        Returns a tuple of (latency in seconds, error code).
        This is a generator function that yields to the simpy environment.
        """

        if self.state == BankState.DOWN:
            yield self.env.timeout(0.05)  # No processing time for DOWN state
            return {"success":    False,
                "error_code": "U28",
                "latency_ms": 50.0,} 

        start = self.env.now

        with self.resource.request() as req:
            yield req

            latency = self._sample_latency()
            yield self.env.timeout(latency / 1000.0)  # SimPy uses seconds

            psr     = self.effective_psr(ctx.network)
            success = random.random() < psr
            error   = None if success else self._classify_error(ctx.network)

            return {
                "success":    success,
                "error_code": error,
                "latency_ms": (self.env.now - start) * 1000,
            }
    



        