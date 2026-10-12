"""Bounded passive evidence. Never grants health authority or clears faults."""
import math
import os
from pathlib import Path
import numpy as np


class RuntimeMeasurements:
    LIMIT=65536
    QUERY_LIMIT=128

    def __init__(self):
        self.samples={};self.dropped={};self.queries=[];self.chunks=[]
        self.row_valid_at_completions=0;self.last_completion=None
        self.last_actuator=None;self.actuator_receipts=0
        self.recent_peak_chars=0;self.pending_query_max=0;self.write_peak_bytes=0
        self.px4_sampled_cpu_seconds=None;self.px4_cpu_sample_count=0

    def sample(self,name,value):
        # Measurement failure is reported, never turned into a safety decision.
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            self.dropped[name]=self.dropped.get(name,0)+1;return
        values=self.samples.setdefault(name,[])
        if len(values)>=self.LIMIT:self.dropped[name]=self.dropped.get(name,0)+1
        else:values.append(float(value))

    def completion(self,protocol):
        end=protocol.monitor.last_response
        if end is None or end==self.last_completion:return
        self.last_completion=end;start=protocol.monitor.last_query
        if len(self.queries)>=self.QUERY_LIMIT:
            self.dropped['queries']=self.dropped.get('queries',0)+1;return
        self.queries.append(dict(start_clock_usec=start[0],start_wall_s=start[1],
            end_clock_usec=end[0],end_wall_s=end[1]))
        self.sample('query_response_seconds',end[1]-start[1])
        if (getattr(protocol,'query_needs_row',False) and protocol.query_row
                and protocol.row_generation==protocol.query_generation
                and 0<=end[1]-protocol.row_received_wall<=.65
                and 0<=end[0]-protocol.row_received_clock<=650000):
            self.row_valid_at_completions+=1

    def chunk(self,clock,wall,size,duration):
        self.sample('parse_validate_seconds',duration)
        if len(self.chunks)>=self.LIMIT:self.dropped['chunks']=self.dropped.get('chunks',0)+1
        else:self.chunks.append(dict(clock_usec=clock,wall_s=wall,bytes=size,parse_validate_s=duration))

    def actuator(self,clock,wall):
        if self.last_actuator is not None:
            self.sample('actuator_interval_wall_seconds',wall-self.last_actuator[1])
            self.sample('actuator_interval_sim_seconds',(clock-self.last_actuator[0])/1e6)
        self.last_actuator=(clock,wall);self.actuator_receipts+=1

    def process_cpu(self,cpu_seconds):
        if cpu_seconds is None:return
        self.px4_cpu_sample_count+=1
        self.px4_sampled_cpu_seconds=max(self.px4_sampled_cpu_seconds or 0.,cpu_seconds)

    def summary(self):
        summaries={}
        for name,values in self.samples.items():
            if not values:continue
            a=np.asarray(values)
            summaries[name]=dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),
                p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(a.max()))
        return dict(samples=summaries,dropped=dict(self.dropped),queries=list(self.queries),
            chunks=list(self.chunks),row_valid_at_completions=self.row_valid_at_completions,
            actuator_receipts=self.actuator_receipts,recent_peak_chars=self.recent_peak_chars,
            pending_query_max=self.pending_query_max,write_peak_bytes=self.write_peak_bytes,
            px4_sampled_cpu_seconds=self.px4_sampled_cpu_seconds,
            px4_cpu_sample_count=self.px4_cpu_sample_count,
            sample_limit=self.LIMIT,query_limit=self.QUERY_LIMIT,
            measurement_semantics='host wall time; parse_validate includes streaming parse + safety validation, excludes raw archive I/O')


def process_cpu_seconds(pid):
    """Linux process utime+stime, sampled; not wall time or whole-system load."""
    try:
        text=Path(f'/proc/{pid}/stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        return (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
    except (OSError,ValueError,IndexError):return None
