import random
import numpy as np
import simpy
import scipy.stats as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


 #Dictionary of issuer banks and their market share distribution (sums to 1.0) , Key : Value(Another dictionary of payment rails 
 # and their conditional market share distribution (sums to 1.0))
 #Rail | Network 
RAIL_NETWORK_CONDITIONAL = {
    "UPI":        {"5G": 0.38, "4G": 0.52, "WIFI": 0.08, "2G": 0.02},
    "CARDS":      {"5G": 0.30, "4G": 0.50, "WIFI": 0.15, "2G": 0.05},
    "NETBANKING": {"5G": 0.25, "4G": 0.35, "WIFI": 0.40, "2G": 0.00},
    "IMPS":       {"5G": 0.30, "4G": 0.55, "WIFI": 0.14, "2G": 0.01},
    "WALLETS":    {"5G": 0.45, "4G": 0.48, "WIFI": 0.07, "2G": 0.00},
    "BNPL_EMI":   {"5G": 0.50, "4G": 0.35, "WIFI": 0.15, "2G": 0.00},
}

#Rail | Issuer 

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
    # IMPS, WALLETS, BNPL_EMI fall back to marginal ISSUER_DISTRIBUTION
}

#Bank | Tier distribution in India (sums to 1.0)
ISSUER_TIER_CONDITIONAL = {
    "SBI":          {"METRO": 0.30, "TIER2": 0.35, "TIER3": 0.35},
    "HDFC":         {"METRO": 0.65, "TIER2": 0.28, "TIER3": 0.07},
    "ICICI":        {"METRO": 0.60, "TIER2": 0.30, "TIER3": 0.10},
    "AXIS":         {"METRO": 0.55, "TIER2": 0.32, "TIER3": 0.13},
    "KOTAK":        {"METRO": 0.70, "TIER2": 0.25, "TIER3": 0.05},
    "AU_SFB":       {"METRO": 0.20, "TIER2": 0.35, "TIER3": 0.45},
    "OTHERS":       {"METRO": 0.20, "TIER2": 0.35, "TIER3": 0.45},
    # default for all others
    "_DEFAULT":     {"METRO": 0.40, "TIER2": 0.35, "TIER3": 0.25},
}

RAIL_AMOUNT_PARAMS = {
    "UPI":        (5.5, 1.2),   # mode ~₹245, range ₹1 - ₹1L
    "CARDS":      (7.0, 1.0),   # mode ~₹1,097, range ₹100 - ₹5L
    "NETBANKING": (9.0, 0.8),   # mode ~₹8,103, range ₹1K - ₹50L
    "IMPS":       (8.5, 0.9),   # mode ~₹4,900, range ₹100 - ₹5L
    "WALLETS":    (4.5, 0.8),   # mode ~₹90, range ₹1 - ₹10K
    "BNPL_EMI":   (7.5, 0.6),   # mode ~₹1,800, range ₹500 - ₹50K
}

# Merchant categories and their rail eligibility
MERCHANT_CATEGORIES = {
    "ECOMMERCE":    {"rails": ["UPI","CARDS","NETBANKING","WALLETS","BNPL_EMI"], "weight": 0.35},
    "FOOD_DELIVERY":{"rails": ["UPI","CARDS","WALLETS"], "weight": 0.15},
    "EDUCATION":    {"rails": ["UPI","CARDS","NETBANKING"], "weight": 0.10},
    "UTILITY":      {"rails": ["UPI","NETBANKING","CARDS"], "weight": 0.12},
    "TRAVEL":       {"rails": ["CARDS","NETBANKING","UPI","BNPL_EMI"], "weight": 0.10},
    "GROCERY":      {"rails": ["UPI","CARDS","WALLETS"], "weight": 0.12},
    "HEALTHCARE":   {"rails": ["UPI","CARDS","NETBANKING"], "weight": 0.06},
}

#Till here , we have compiled data of probability distributions 
# of various parameters like issuer banks, payment rails, 
# network types, merchant categories, and transaction amounts. 
# These distributions will be used in the simulation to generate
# realistic transaction scenarios based on the defined p
# robabilities and conditions.
# Rails | Merchants
# Tier | Rails 
# Banks | Rails 
# Network | Rails 

@dataclass 
class TransactionContext:
    txn_id:           str
    rail:             str
    issuer_bank:      str
    network:          str
    geography_tier:   str
    merchant_category:str
    amount_inr : float
    hour_of_day : int 
    day_of_week : int 
    is_peak_hour : bool
    is_salary_day : bool
    sim_time : float 

# here , we check if the transaction amount is greater than 50,000 INR to classify it as a high-value transaction.
@property 
def is_high_value(self) -> bool: 
    if self.amount_inr > 50_000: 
        return True
    else: 
        return False


@property
def device_tier(self) -> str: 
      return {"5G": "HIGH", "WIFI": "HIGH", "4G": "MID", "2G": "LOW"}[self.network] # this right here checks the key , which has been assigned to the network type and returns the corresponding device tier based on the mapping provided in the dictionary.]

def weighted_choice(dictionary : dict ) -> str : 
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    return random.choices(keys, weights=values, k=1)[0] 

def context_generator()-> TransactionContext: 
    rail = weighted_choice({k: sum(v.values()) for k, v in RAIL_ISSUER_CONDITIONAL.items()})
    issuer_bank = weighted_choice(RAIL_ISSUER_CONDITIONAL[rail])
    network = weighted_choice(RAIL_NETWORK_CONDITIONAL[rail])
    geography_tier = weighted_choice(ISSUER_TIER_CONDITIONAL.get(issuer_bank, ISSUER_TIER_CONDITIONAL["_DEFAULT"]))
    merchant_category = weighted_choice({k: v["weight"] for k, v in MERCHANT_CATEGORIES.items() if rail in v["rails"]})
    amount_params = RAIL_AMOUNT_PARAMS[rail]
    amount_inr = np.clip(st.lognorm(s=amount_params[1], scale=np.exp(amount_params[0])).rvs(), 1, 5e5)
    hour_of_day = random.randint(0, 23)
    day_of_week = random.randint(0, 6)
    is_peak_hour = hour_of_day in range(8, 11) or hour_of_day in range(18, 21)
    is_salary_day = day_of_week == 4 and hour_of_day in range(9, 17)  # Assuming salary day is Friday
    sim_time = datetime.now().timestamp()
    
    return TransactionContext(
        txn_id=str(random.randint(1000000000, 9999999999)),
        rail=rail,
        issuer_bank=issuer_bank,
        network=network,
        geography_tier=geography_tier,
        merchant_category=merchant_category,
        amount_inr=amount_inr,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        is_peak_hour=is_peak_hour,
        is_salary_day=is_salary_day,
        sim_time=sim_time
    )    