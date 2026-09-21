import simpy 
import random
from collections import deque, defaultdict, Counter
from context import context_generator
from gateway import BankNode, BankState, BANK_CONFIGURATIONS

class Transaction_Record: 
    """    Immutable record of a single processed transaction. 
          Equivalent to a Java record class or DTO.    """

    def __init__(self, sim_time , strategy, issuer_bank, rail, network, tier, status, success, error_code, latency_ms):
        self.sim_time = sim_time
        self.strategy = strategy
        self.issuer_bank = issuer_bank
        self.rail = rail
        self.network = network
        self.tier = tier
        self.status = status
        self.success = success
        self.error_code = error_code
        self.latency_ms = latency_ms



#remember , this works as a java equivalent of a tostring method , where it represents the 
#string representation of the object.
    def __repr__(self):
       status = "OK" if self.success else f"ERROR:({self.error_code})"
       return (f"[T = {self.sim_time:>7.2f}s] " f"{self.strategy:<10} " # ">" this is right align , used for numbers and "<" is left align , used for strings.
                f"{self.issuer_bank:<12} " 
                f"{self.rail:<12} " 
                f"{self.network:>10} " 
                f"{self.tier:>10} " 
                f"{status} " 
                f"latency={self.latency_ms:>5.2f}ms")


class MetricCollector: 
    """
    Collects and computes all benchmark metrics.
    One instance shared across the entire simulation.
    Equivalent to a Java @Service singleton.
    """

    def __init__(self, window_size_for_rolling_metrics: int = 100): 
        self.window_size_for_rolling_metrics = window_size_for_rolling_metrics
        self._records: list[Transaction_Record] = []

    def record(self, Transaction_Record: Transaction_Record):
        self._records.append(Transaction_Record) #puts a new record in the list of records



#returns records by stratgy , if no strategy is provided , it returns all records.
    def get_records_by_strategy(self, strategy_demanded: str = None) -> list[Transaction_Record]:
        if strategy_demanded is None:
            return self._records
        return [r for r in self._records if r.strategy == strategy_demanded]


#RPFRC rolling_psr_for_determining recovery_curve
    def rolling_psr_for_recovery_curve(self, strategy_demanded: str = None) -> list[tuple]:


        """
        Returns a list of rolling PSR values for the recovery curve.
        If no strategy is provided, it returns the rolling PSR for all strategies combined.
        """

        buffer = deque(maxlen=self.window_size_for_rolling_metrics)
        result = [] 

        # adu, your buffer will skip the first few records until it reaches the window size, then it will start calculating the rolling PSR.

        for i , record in enumerate(self.get_records_by_strategy(strategy_demanded)):
            buffer.append(1 if record.success else 0)
            if len(buffer) == self.window_size_for_rolling_metrics:
               result.append((i, round(sum(buffer) / self.window_size_for_rolling_metrics, 4))) #appends the index and the rolling PSR value to the result list
        return result

    def recovery_lag(self, strategy_demanded: str, fault_start_time: float, threshold: float = 0.80) -> int: 
        """
        Returns the number of transactions processed before the first successful transaction, so essentially how many 
        transactions were processed after the fault started until Payment Success Rate (PSR) reaches the threshold.
    
        """
        #so here we are going to addrerss three main issues right , one is post fault lag . Since our 
        #rolling_psr_for_recovery_curve skips the first few records till the buffer contents == window size , there is a lag between 
        #  first failed transaction that is recorded in the rolling_psr_for_recovery_curve and how recovery_lag is calculatiing the threshold , 
        # as a result, the recovery_lag will actually say , the initial few unsuccessful transaction recorded are not problematic,  since the RPFRC has say 99 successful txn and 1 unsuccessful transaction , the PSR will be 0.99 and it will say , 
        # we are above the threshold of 0.8 , but in reality , 
        # we have a problem because we have a failed transaction that is not being accounted for in the RPFRC.
        # The global 100-item buffer is filled with 100 successful transactions (1s) from T = 200
       # to $T = 300.When the fault hits at T=300 and 10 transactions fail in a row (0s), the buffer now contains 90 1s and 10 0s.The global PSR drops from 1.00 to 0.90. 
       # It takes time for the global average to reflect the outage because the pre-fault 1s are dragging the average up.
        
        post_fault = [r for r in self.get_records_by_strategy(strategy_demanded) if r.sim_time >= fault_start_time]

        buf = deque(maxlen = 50 )
    
        for i, r in enumerate(post_fault): 
            if r.success: 
                buf.append(1)
                 
            else: 
                buf.append(0)
              
            if(len(buf) >= 25 and sum(buf)/len(buf) >= threshold):
                return i + 1 # +1 because we want to include the current transaction in the count
        return len(post_fault)    

    def overall_psr(self, strategy: str)-> float: 
        recordz = self.get_records_by_strategy(strategy)
        if len(recordz) == 0: 
            return 0.0
        return sum(1 for r in recordz if r.success)/ len(recordz)


    def error_distribution(self, strategy: str = None) -> dict:
        error_codes = {} 

        for r in self.get_records_by_strategy(strategy):
            if r.error_code: 
                 error_cd = r.error_code
                 if error_cd not in error_codes:
                   error_codes[error_cd] = 1
                 else : 
                  error_codes[error_cd] += 1
        return error_codes

    def print_summary(self, strategy: str, fault_start: float):
        recs = self.get_records_by_strategy(strategy)
        print(f"\n{'─' * 55}")
        print(f"  Strategy      : {strategy}")
        print(f"  Total txn     : {len(recs)}")
        print(f"  Overall PSR   : {self.overall_psr(strategy):.4f}")
        print(f"  Errors        : {self.error_distribution(strategy)}")
        print(f"  Recovery lag  : {self.recovery_lag(strategy, fault_start)} txn")
        print(f"{'─' * 55}")

    def print_psr_chart(self, strategy: str, step: int = 10):
        print(f"\n  Rolling PSR ({self.window_size_for_rolling_metrics}-txn window) — {strategy}")
        print(f"  {'txn':>6}  {'PSR':>6}  chart")
        print(f"  {'─' * 42}")
        for idx, psr in self.rolling_psr_for_recovery_curve(strategy)[::step]:
            bar = "█" * int(psr * 30)
            flag = " ↓" if psr < 0.85 else ""
            print(f"  {idx:>6}  {psr:.4f}  {bar}{flag}")
       