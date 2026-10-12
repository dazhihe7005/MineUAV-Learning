"""Single-flight dual-clock query service and bounded nonblocking console I/O."""
import math
import codecs
from pathlib import Path


class BoundedQueryScheduler:
    PERIOD_S=.5
    def __init__(self,protocol):
        self.protocol=protocol;self.next_wall=None;self.next_clock=None
        self.last_clock=None;self.last_wall=None;self.starts=0;self.max_lateness_s=0.

    def service(self,clock,wall,send,dispatch_ready=True):
        m=self.protocol.monitor
        if (not isinstance(clock,int) or clock<=0 or not math.isfinite(wall)
                or (self.last_clock is not None and (clock<self.last_clock or clock-self.last_clock>150000))
                or (self.last_wall is not None and wall<self.last_wall)):
            m._fail('query scheduling clock rollback/jump');return
        self.last_clock=clock;self.last_wall=wall
        m._check_expiry(clock,wall)
        if m.fault or clock<7000000:return
        if self.next_wall is None:self.next_wall=wall;self.next_clock=clock
        if m.last_query is None and (wall-self.next_wall>.65 or clock-self.next_clock>650000):
            m._fail('first safety query service stopped by backlog');return
        if not dispatch_ready:return
        if m.query is not None:return # no backlog and no incomplete overlap
        if wall<self.next_wall and clock<self.next_clock:return
        self.max_lateness_s=max(self.max_lateness_s,wall-self.next_wall)
        self.protocol.query_requested(clock,wall,'')
        if m.fault:return
        try:send(self.protocol.queries().encode())
        except (OSError,ValueError,TimeoutError) as error:
            m._fail('query dispatch failed: '+str(error));return
        self.starts+=1
        # Advance absolute phase, never enqueue a burst of overdue jobs.
        if self.next_wall<=wall:
            self.next_wall+=(math.floor((wall-self.next_wall)/self.PERIOD_S)+1)*self.PERIOD_S
        else:self.next_wall=wall+self.PERIOD_S
        if self.next_clock<=clock:
            self.next_clock+=((clock-self.next_clock)//500000+1)*500000
        else:self.next_clock=clock+500000


class BoundedConsoleWriter:
    LIMIT=4096
    def __init__(self):self.data=b'';self.deadline=None
    @property
    def pending_bytes(self):return len(self.data)
    def enqueue(self,data,wall,response_limit=.15):
        if self.data:raise ValueError('console write backlog; one batch only')
        if not data or len(data)>self.LIMIT:raise ValueError('console command batch bound')
        self.data=bytes(data);self.deadline=wall+response_limit
    def flush(self,write,wall):
        if not self.data:return True
        if wall>self.deadline:raise TimeoutError('console write absolute deadline')
        try:count=write(self.data)
        except BlockingIOError:return False
        if not isinstance(count,int) or not 0<count<=len(self.data):raise OSError('invalid console write count')
        self.data=self.data[count:]
        return not self.data


class ConsoleArchive:
    """Preserve every raw character on disk; recent hot-path text is bounded.

    Final historical materialization is outside the control loop and capped
    at16MiB for this finite45s runner. Exceeding the cap is a safety failure,
    not silent eviction of safety state. Raw archive still retains the chunk.
    """
    RECENT_LIMIT=65536
    TOTAL_LIMIT=16*1024*1024
    def __init__(self,path):
        self.file=Path(path).open('w+b');self.recent='';self.total_chars=0;self.total_bytes=0
    def append(self,chunk):
        raw=chunk.encode('utf-8') if isinstance(chunk,str) else chunk
        self.file.write(raw);self.total_bytes+=len(raw)
        text=raw.decode('utf-8',errors='replace');self.total_chars+=len(text)
        self.recent=(self.recent+text)[-self.RECENT_LIMIT:]
        if self.total_bytes>self.TOTAL_LIMIT:raise ValueError('console archive lifetime bound exceeded')
    def read_final(self):
        self.file.flush();self.file.seek(0)
        return self.file.read(self.TOTAL_LIMIT).decode('utf-8',errors='replace')
    def close(self):self.file.close()


class ConsoleService:
    """Same bounded service at physical ticks and between≤8ms receive slices."""
    READS_PER_SERVICE=4
    def __init__(self,protocol,archive,read,write):
        self.protocol=protocol;self.archive=archive;self.read=read;self.write=write
        self.scheduler=BoundedQueryScheduler(protocol);self.writer=BoundedConsoleWriter()
        self.wall_now=None;self.fallback_wall=0.;self.monitoring=True
        self.decoder=codecs.getincrementaldecoder('utf-8')(errors='strict');self.fragment_since=None
        protocol.wall_now=lambda:self.now()
    def now(self):return self.wall_now() if self.wall_now is not None else self.fallback_wall
    def check_fragment_deadline(self,clock):
        if self.monitoring and self.fragment_since and (clock-self.fragment_since[0]>150000 or self.now()-self.fragment_since[1]>.15):
            self.protocol.monitor._fail('incomplete UTF-8 console input timeout');return False
        return True
    def drain(self,clock,wall):
        self.fallback_wall=wall
        self.check_fragment_deadline(clock)
        for _ in range(self.READS_PER_SERVICE):
            chunk=self.read()
            if not chunk:
                if self.monitoring:
                    self.protocol.console_backlog_pending=False
                    self.protocol.complete_query_if_ready(clock,self.now())
                return not self.protocol.transport_pending # complete byte boundary as well as empty PTY
            if len(chunk)>4096:raise ValueError('console read slice bound exceeded')
            received_wall=self.now()
            self.archive.append(chunk)
            if self.monitoring:
                if not self.check_fragment_deadline(clock):return False
                # A complete record is not proof that subsequent queued data
                # is absent. Finish only AFTER observing a true empty PTY.
                self.protocol.console_backlog_pending=True
                try:text=self.decoder.decode(chunk) if isinstance(chunk,bytes) else chunk
                except UnicodeError:
                    self.protocol.monitor._fail('invalid UTF-8 console bytes; raw evidence archived');return False
                if not self.check_fragment_deadline(clock):return False
                self.protocol.transport_pending=bool(self.decoder.getstate()[0])
                if self.protocol.transport_pending:
                    if self.fragment_since is None:self.fragment_since=(clock,received_wall)
                    if not self.check_fragment_deadline(clock):return False
                else:self.fragment_since=None
                self.protocol.on_console_chunk(text,clock,self.now())
        return False
    def service(self,clock,wall):
        drained=self.drain(clock,wall)
        try:
            self.writer.flush(self.write,self.now())
            self.scheduler.service(clock,self.now(),lambda data:self.writer.enqueue(data,self.now()),dispatch_ready=drained)
            self.writer.flush(self.write,self.now())
        except (OSError,ValueError,TimeoutError) as error:
            self.protocol.monitor._fail('bounded console service: '+str(error))
    def diagnostic(self,data,wall):
        # Defer, not enqueue: no diagnostic backlog behind safety queries.
        m=self.protocol.monitor
        if m.query is not None or self.writer.pending_bytes:return False
        if self.scheduler.next_wall is not None and self.scheduler.next_wall-wall<.15:return False
        self.writer.enqueue(data,wall);self.writer.flush(self.write,wall)
        return True
