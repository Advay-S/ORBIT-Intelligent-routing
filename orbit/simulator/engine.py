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



    def rolling_psr_for_recovery_curve(self, strategy_demanded: str = None) -> list[tuple]:


        """
        Returns a list of rolling PSR values for the recovery curve.
        If no strategy is provided, it returns the rolling PSR for all strategies combined.
        """

        buffer = deque(maxlen=self.window_size_for_rolling_metrics)
        result = [] 

        for i , record in enumerate(self.get_records_by_strategy(strategy_demanded)):
            buffer.append(1 if record.success else 0)
            if len(buffer) == self.window_size_for_rolling_metrics:
               result.append((i, round(sum(buffer) / self.window_size_for_rolling_metrics))) #appends the index and the rolling PSR value to the result list
        return result


    