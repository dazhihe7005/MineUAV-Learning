"""Source-time health snapshots + independent query and actuator watchdogs.

Only actual uORB publication timestamps renew source leases. Console reads
do not. Times are explicitly PX4 lockstep microseconds and monotonic host
seconds; wall time continues when lockstep simulation is stalled.
"""
import math
import re
from .yaw_validation import latest_topics,heading_topics_ready


def completed_topics(response):
    """Keep complete records even when the next PTY title is fragmented.

    This is NOT query-completion evidence: the legacy whole-response parser
    remains strict. A later incomplete record must not hide an already
    received unhealthy record, nor authorize a new maneuver.
    """
    text=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",response).replace("\r\n","\n").replace("\r","\n")
    boundaries=list(re.finditer(r"(?m)^(?:pxh>[ \t]*)?(?:TOPIC:[^\n]*(?:\n|$)|(?:T|TO|TOP|TOPI|TOPIC)[ \t]*(?:\n|$))",text))
    result={}
    for i,start in enumerate(boundaries):
        block=text[start.start():boundaries[i+1].start() if i+1<len(boundaries) else len(text)]
        result.update({name:fields for name,fields in latest_topics(block).items() if fields})
    return result


class TelemetryFreshness:
    # Cadence-aware fixed source leases, independent of communication liveness.
    # Local position's125Hz source may conservatively expire before the sum
    # .008+.5+.15s at worst-case query jitter: expiry still means safe stop.
    SOURCE_LIMITS={"vehicle_local_position":650000,
                   "estimator_status_flags":1650000,"vehicle_status":1150000}
    RESPONSE_S=.15
    QUERY_S=.5
    ACTUATOR_S=.25
    REQUIRED={
        'vehicle_local_position':('xy_valid','z_valid','v_xy_valid','v_z_valid','heading_good_for_control','heading'),
        'estimator_status_flags':('cs_tilt_align','cs_yaw_align','cs_mag_aligned_in_flight','cs_in_air','cs_gnss_pos','cs_mag_hdg','cs_mag_3d','cs_mag_fault','cs_baro_fault','cs_gnss_fault','cs_gnss_yaw_fault','cs_gnss_hgt_fault'),
        'vehicle_status':('pre_flight_checks_pass','failsafe','arming_state','nav_state')}

    def __init__(self):
        self.topics={};self.receipts={};self.query=None;self.last_query=None
        self.last_response=None;self.last_actuator=None;self.last_clock=None
        self.last_wall=None;self.fault=None;self.events=[];self._seen={}
        self.observed_source={} # max3 topics, ordering evidence NOT positive source leases

    def _fail(self,reason):
        if self.fault is None:self.fault=reason

    def _event(self,event):
        if len(self.events)>=8192:
            self._fail('safety event memory bound exceeded');return
        self.events.append(event)

    def validate_source(self,name,fields,clock):
        timestamp=fields.get('timestamp',float('nan'));sample=fields.get('timestamp_sample')
        old=self.observed_source.get(name)
        if (not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool) or not math.isfinite(timestamp)
                or not float(timestamp).is_integer() or timestamp<=0 or timestamp>clock
                or (old and timestamp<old['timestamp'])):
            self._fail('invalid/future/backwards publication: '+name);return False
        if old and timestamp==old['timestamp'] and fields!=old:
            self._fail('changed payload with duplicate publication: '+name);return False
        if sample is not None and (not isinstance(sample,(int,float)) or isinstance(sample,bool) or not math.isfinite(sample)
                or not float(sample).is_integer() or sample>timestamp or sample<0
                or (old and 'timestamp_sample' in old and sample<old['timestamp_sample'])):
            self._fail('invalid/backwards sample time: '+name);return False
        self.observed_source[name]=dict(fields)
        return True

    def inspect_health(self,name,fields,required_values=None,require_mag_fusion=False,clock=None):
        """Negative evidence latches even outside a query; never renews a lease."""
        if clock is not None and not self.validate_source(name,fields,clock):return
        old=self.topics.get(name)
        for key in self.REQUIRED[name]:
            value=fields.get(key)
            if key in ('heading','arming_state','nav_state'):
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
                    self._fail('incomplete/invalid safety field: '+name+'.'+key)
            elif not isinstance(value,bool):self._fail('incomplete/invalid safety field: '+name+'.'+key)
        asserted={'vehicle_local_position':('xy_valid','z_valid','v_xy_valid','v_z_valid','heading_good_for_control'),
            'estimator_status_flags':('cs_tilt_align','cs_yaw_align','cs_mag_aligned_in_flight','cs_in_air','cs_gnss_pos'),
            'vehicle_status':('pre_flight_checks_pass',)}
        hazards={'estimator_status_flags':('cs_mag_fault','cs_baro_fault','cs_gnss_fault','cs_gnss_yaw_fault','cs_gnss_hgt_fault'),
            'vehicle_status':('failsafe',)}
        if any(fields.get(k) is True for k in hazards.get(name,())):self._fail('observed unsafe flag: '+name)
        if old and any(old.get(k) is True and fields.get(k) is not True for k in asserted.get(name,())):
            self._fail('observed healthy flag revoked: '+name)
        if required_values and any((fields.get(k) is not value if isinstance(value,bool) else fields.get(k)!=value)
                for k,value in required_values.get(name,{}).items()):self._fail('observed flight health lost: '+name)
        if require_mag_fusion and name=='estimator_status_flags' and not (fields.get('cs_mag_hdg') is True or fields.get('cs_mag_3d') is True):
            self._fail('observed magnetic heading fusion lost')

    def _check_expiry(self,clock,wall):
        # Check OLD deadlines before any receipt/actuator/query can renew them.
        # A late message must never erase evidence that the watchdog expired.
        if self.query and (clock-self.query[0]>150000 or wall-self.query[1]>self.RESPONSE_S):
            self._fail("query response timeout/truncated topic")
        if self.last_actuator and not (0<=clock-self.last_actuator[0]<=250000 and 0<=wall-self.last_actuator[1]<=self.ACTUATOR_S):
            self._fail("actuator communication watchdog")
        if self.last_query and (clock-self.last_query[0]>650000 or wall-self.last_query[1]>.65):
            self._fail("safety query service stopped")
        for name,fields in self.topics.items():
            if not 0<=clock-fields["timestamp"]<=self.SOURCE_LIMITS[name]:
                self._fail("source publication expired: "+name)
            if wall-self.receipts[name][1]>.65:self._fail("state receipt watchdog: "+name)

    def next_wall_deadline(self):
        """Absolute host-monotonic offset, even while simulation is stalled."""
        deadlines=[]
        if self.query:deadlines.append(self.query[1]+self.RESPONSE_S)
        if self.last_actuator:deadlines.append(self.last_actuator[1]+self.ACTUATOR_S)
        if self.last_query:deadlines.append(self.last_query[1]+.65)
        deadlines.extend(receipt[1]+.65 for receipt in self.receipts.values())
        return min(deadlines,default=float("inf"))

    def begin_query(self,clock,wall):
        self._check_expiry(clock,wall)
        if self.query is not None:self._fail("overlapping/incomplete safety query")
        self.query=(clock,wall);self.last_query=(clock,wall);self._seen={}
        self._event({"kind":"query","clock_usec":clock,"wall_s":wall})

    def observe(self,response,clock,wall):
        # The runner passes only bytes received AFTER this query was issued.
        parsed=latest_topics(response)
        complete=completed_topics(response)
        self.observe_topics(complete,clock,wall,finish=all(parsed.get(name) for name in self.SOURCE_LIMITS))

    def observe_topics(self,complete,clock,wall,finish=False,required_values=None,require_mag_fusion=False):
        """Same validation for bounded completed records; no receipt-time rewrite."""
        self._check_expiry(clock,wall)
        if self.query is None:return
        for name in self.SOURCE_LIMITS:
            fields=complete.get(name,{})
            if not fields or fields==self._seen.get(name):continue
            self._seen[name]=dict(fields)
            timestamp=fields.get("timestamp",float("nan"))
            sample=fields.get("timestamp_sample")
            old=self.topics.get(name)
            if not self.validate_source(name,fields,clock):continue
            self.inspect_health(name,fields,required_values,require_mag_fusion)
            self.topics[name]=dict(fields)
            self.receipts[name]=(clock,wall)
            # Validate completeness independently of temporary flight eligibility.
            self._event({"kind":"receipt","topic":name,"clock_usec":clock,"wall_s":wall,
                                "publication_usec":timestamp,"sample_usec":sample,
                                "duplicate":bool(old and timestamp==old["timestamp"])})
        if finish:
            self.last_response=(clock,wall);self.query=None

    def actuator(self,timestamp,wall):
        if (not isinstance(timestamp,int) or timestamp<=0 or not math.isfinite(wall)
            or (self.last_actuator and (timestamp<=self.last_actuator[0] or wall<self.last_actuator[1]))):
            self._fail("nonmonotonic actuator communication")
        else:
            self._check_expiry(timestamp,wall)
            self.last_actuator=(timestamp,wall)

    def decision(self,clock,wall,require_flight=True):
        if (not isinstance(clock,int) or clock<=0 or not math.isfinite(wall)
            or (self.last_clock is not None and (clock<self.last_clock or clock-self.last_clock>150000))
            or (self.last_wall is not None and wall<self.last_wall)):
            self._fail("monitor clock rollback/jump")
        self.last_clock=clock;self.last_wall=wall
        if self.last_actuator is None or not (0<=clock-self.last_actuator[0]<=250000 and 0<=wall-self.last_actuator[1]<=self.ACTUATOR_S):
            self._fail("actuator communication watchdog")
        if self.query and (clock-self.query[0]>150000 or wall-self.query[1]>self.RESPONSE_S):self._fail("query response timeout/truncated topic")
        if self.last_query is None or clock-self.last_query[0]>650000 or wall-self.last_query[1]>.65:self._fail("safety query service stopped")
        ages={}
        for name,limit in self.SOURCE_LIMITS.items():
            fields=self.topics.get(name)
            if fields is None:
                if self.query is None:self._fail("missing safety topic: "+name)
                continue
            age=clock-fields["timestamp"]
            if not 0<=age<=limit:self._fail("source publication expired: "+name)
            receipt=self.receipts[name]
            if wall-receipt[1]>.65:self._fail("state receipt watchdog: "+name)
            ages[name]={"publication_usec":fields["timestamp"],"sample_usec":fields.get("timestamp_sample"),
                        "received_clock_usec":receipt[0],"received_wall_s":receipt[1],"source_age_s":age/1e6}
        # Evaluate real values separately from source leases. No timestamp rewriting.
        healthy=heading_topics_ready(self.topics,clock,age_limits=self.SOURCE_LIMITS)
        if require_flight and self.topics and not healthy:self._fail("EKF/heading/localization/vehicle unhealthy")
        reason=self.fault or ("query pending; no new maneuver" if self.query else "healthy source + live transport")
        return {"safe_stop":self.fault is not None,"allow_new_maneuver":bool(healthy and not self.query and not self.fault),
                "reason":reason,"query_pending":self.query is not None,"clock_usec":clock,"wall_s":wall,
                "last_query":self.last_query,"last_response":self.last_response,"last_actuator":self.last_actuator,"topics":ages}
