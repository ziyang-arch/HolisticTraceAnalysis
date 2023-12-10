
from collections import defaultdict
from typing import Dict, List, TYPE_CHECKING

import pandas as pd
import plotly.express as px

from hta.utils.utils import get_kernel_type, KernelType, merge_kernel_intervals

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.trace import Trace

class IdlenessInterpretation:
    def __init__(self):
        pass


    @classmethod
    def get_idle_distribution(self, cls, t: "Trace", visualize: bool = True) -> pd.DataFrame:
        """
        get idle distribution on event level, 

        """
        sym_table = t.symbol_table.get_sym_table()
        
        def get_idle_distribution_value(self, trace_df: pd.DataFrame) -> float:
            '''
            return tuple(comp_idleness_in: DataFrame, comm_idleness_in: DataFrame)
            '''
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
            gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
                lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
            )
            comp_kernels = merge_kernel_intervals(
                gpu_kernels[
                    gpu_kernels["kernel_type"].eq(KernelType.COMPUTATION.name)
                ].copy()
            )
            comm_kernels = merge_kernel_intervals(
                gpu_kernels[
                    gpu_kernels["kernel_type"].eq(KernelType.COMMUNICATION.name)
                ].copy()
            )
            return self.get_idleness_pctg(comp_kernels), self.get_idleness_pctg(comm_kernels)
        
        
        result: Dict[str, float] = defaultdict(list)

        for rank, trace_df in t.traces.items():
            result["rank"].append(rank)
            idle_pctg_item=self.get_idleness_pctg(trace_df)

            result["idle_in_pctg"].append(idle_pctg_item)
            print("rank: ", rank)
            print("idle pctg: ", idle_pctg_item)
            #print("idle_in:\n", result["idle_in"][-1])
        #result_df = pd.DataFrame(result)
        return result
    
    def get_idleness_pctg(kernel):
            '''
            input dataframe of 'ts' and 'end' of a trace
            '''
            kernel.sort_values("ts", inplace=True)
            idle_in=kernel['idle_after']
            dur_total=kernel['end'].max()-kernel['ts'].min()
            dur_idle= idle_in.sum()
            idle_pctg=dur_idle/dur_total
            return idle_pctg
    
    @classmethod
    def placeholder(cls, t: "Trace", visualize: bool = True) -> pd.DataFrame:
        pass
