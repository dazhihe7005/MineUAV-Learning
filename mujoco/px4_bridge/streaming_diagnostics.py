"""Bounded line/record parser. Input is NEW bytes decoded by the PTY reader.

No timestamp is invented. Allocator rows have host receipt time only (assigned
by the caller), not a uORB publication time. History is never searched here.
"""
import re
from .yaw_validation import latest_topics
from .telemetry_freshness import TelemetryFreshness


class StreamingDiagnostics:
    LINE_LIMIT=8192
    RECORD_LIMIT=32768
    SLICE_LIMIT=4096
    ANSI=re.compile(r'\x1b\[[0-9;]*m') # SGR styling only, not cursor semantics
    FIELD=re.compile(r'^\s{4}[a-zA-Z0-9_]+:')

    def __init__(self):
        self.line='';self.record=[];self.record_size=0;self.topic=None
        self.instance=0;self.processed_chars=0;self.peak_buffer_chars=0
        self.previous_cr=False

    @property
    def pending(self):
        partial=self.ANSI.sub('',self.line).strip()
        # Any nonterminated content could be the start of a safety record,
        # including an INFO allocator prefix split at ANY byte. Only the
        # complete idle shell prompt is an unambiguous non-record boundary.
        if partial=='pxh>':partial=''
        return bool(self.record or partial)

    def _finish(self):
        events=[]
        if self.record:
            text=''.join(self.record)+'\n'
            # Exact whitespace grammar of latest_topics, including Unicode
            # whitespace. A narrower guard would permit hidden overwrites.
            keys=re.findall(r'(?m)^\s{4}([a-zA-Z0-9_]+):',text)
            if len(keys)!=len(set(keys)):raise ValueError('duplicate safety fields')
            fields=latest_topics(text).get(self.topic,{})
            if not fields:raise ValueError('incomplete/corrupt primary safety record')
            events.append({'kind':'topic','topic':self.topic,'fields':fields})
        self.record=[];self.record_size=0
        return events

    def _line(self,raw):
        line=self.ANSI.sub('',raw)
        if '\x1b' in line:
            # Clear-line controls are legitimate only on a blank editor line
            # or explicit shell-command echo, never before/inside data.
            cleared=re.sub(r'\x1b\[[0-9;]*K','',line)
            if (cleared.strip() and not cleared.startswith('pxh>')) or 'TOPIC:' in cleared or 'Instance ' in cleared or 'MineUAV yaw axis:' in cleared:
                raise ValueError('terminal control inside console data')
            line=cleared
        if '\x1b' in line:raise ValueError('unsupported/damaged console escape')
        events=[]
        header=line.lstrip()
        if header.startswith('pxh>'):header=header[4:].lstrip()
        if header.strip() in ('T','TO','TOP','TOPI','TOPIC'):
            raise ValueError('truncated diagnostic topic boundary')
        if re.match(r'TOPIC\b',header) and not header.startswith('TOPIC:'):
            raise ValueError('damaged diagnostic topic boundary')
        if header.startswith('TOPIC:'):line=header
        elif 'TOPIC:' in line and line.lstrip().startswith('pxh>'):
            line=line[line.index('TOPIC:'):]
        if line.startswith('TOPIC:'):
            if self.record:
                raise ValueError('truncated safety record before next topic')
            self.record=[];self.record_size=0
            title=re.fullmatch(r'TOPIC:\s+([a-z0-9_]+)(?: +[0-9]+ +instances)?[ \t]*\n',line)
            if not title:raise ValueError('invalid topic title')
            self.topic=title[1];self.instance=0
            if self.topic not in TelemetryFreshness.SOURCE_LIMITS and any(name.startswith(self.topic) for name in TelemetryFreshness.SOURCE_LIMITS):
                raise ValueError('truncated safety topic title')
            if self.topic in TelemetryFreshness.SOURCE_LIMITS:
                self.record=[line];self.record_size=len(line)
            return events
        instance_line=line.lstrip()
        if instance_line.startswith('pxh>'):instance_line=instance_line[4:].lstrip()
        # Recognize EVERY instance-like header before fields/shell handling.
        # Noncanonical separators fail closed rather than hiding the boundary.
        if instance_line.startswith('Instance'):
            line=instance_line
            match=re.fullmatch(r'Instance ([0-9]+):[^\n]*\n',line)
            if not match:raise ValueError('invalid topic instance')
            # Native type labels / repeated instance declarations do not
            # constitute a primary payload. Body validation below rejects
            # malformed fields before they can be silently discarded here.
            if self.record and self.instance==0 and any(self.FIELD.match(x) for x in self.record):
                # A new instance cannot be used as a primary terminator.
                raise ValueError('unterminated primary instance')
            self.instance=int(match[1])
            if self.instance!=0:self.record=[];self.record_size=0
            elif self.record:
                self.record.append(line);self.record_size+=len(line)
                if self.record_size>self.RECORD_LIMIT:raise ValueError('console record bound exceeded')
            return events
        if self.record:
            if not line.strip() or line.startswith('pxh>'):
                if len(self.record)>1:events+=self._finish()
            elif self.instance==0:
                # Started primary snapshots are not an opaque log bucket.
                # Malformed fields cannot disappear simply because FIELD
                # cannot recognize them. Only the native label, canonical
                # field lines and explicitly tagged interleaved logs are valid.
                if (line.strip()!=self.topic and not self.FIELD.match(line)
                        and not re.match(r'(?:INFO|WARN|ERROR|DEBUG)\s+\[',line)
                        and not line.startswith('MineUAV yaw axis:')):
                    raise ValueError('malformed primary safety body')
                self.record.append(line);self.record_size+=len(line)
                if self.record_size>self.RECORD_LIMIT:raise ValueError('console record bound exceeded')
        match=re.fullmatch(r'(?:INFO\s+\[control_allocator\]\s+)?MineUAV yaw axis: ([^\n]+)\n',line)
        if match:
            events.append({'kind':'yaw_row','value':match[1]})
        elif 'MineUAV yaw axis:' in line:
            raise ValueError('corrupt allocator diagnostic')
        self.peak_buffer_chars=max(self.peak_buffer_chars,self.record_size)
        return events

    def feed(self,chunk):
        events=[]
        # Bound temporary work as well as retained state. CR is a terminal
        # boundary in PX4's command editor; CRLF's extra blank is harmless.
        for offset in range(0,len(chunk),self.SLICE_LIMIT):
            raw=chunk[offset:offset+self.SLICE_LIMIT]
            self.processed_chars+=len(raw)
            part=raw[1:] if self.previous_cr and raw.startswith('\n') else raw
            self.previous_cr=raw.endswith('\r')
            part=part.replace('\r\n','\n').replace('\r','\n')
            for fragment in part.splitlines(keepends=True):
                self.line+=fragment
                if len(self.line)>self.LINE_LIMIT:raise ValueError('console line bound exceeded')
                self.peak_buffer_chars=max(self.peak_buffer_chars,len(self.line)+self.record_size)
                if self.line.endswith('\n'):
                    events+=self._line(self.line);self.line=''
            # Listener can end with a shell prompt without a newline.
            if self.record and self.ANSI.sub('',self.line).startswith('pxh>'):
                events+=self._finish()
        return events
