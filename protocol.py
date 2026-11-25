"""Data center backbone protocol: Implementation 1.

Each sender keeps one FIFO queue per destination and sends from them in
round-robin order, one packet per destination per turn. Sender i starts its
rotation at receiver i+1, so the nine senders do not all hit the same receiver
first. The receiver hands every DATA packet to the application.

Packet format (network byte order):

    DATA = [type: 1 byte = 0][send number: 1 byte][message ID: 4 bytes][payload]
    ACK  = [type: 1 byte = 1][send number: 1 byte][message ID: 4 bytes]

The send number counts transmissions of one message (1, 2, ...), and an ACK
echoes the number of the copy it answers. The sender can then time every
ACK against the exact copy it answers, resends included. Without it, Karn's
rule must discard every resent message's ACK, and once the path's RTT grows
past the RTO, every message is resent and no sample is ever valid again.

Notes for Implementation 2:

  - With pacing, on_app_data must not arm SEND_TIMER at 0 right after a send.
    Keep a next-allowed-send time and arm at max(0, next_send_at - now).
  - Encode each packet once and keep the bytes for resends: msg.payload is
    rebuilt from about 44 SHA-256 hashes on every read.
  - Do not take RTT samples from resent packets; the ACK could answer either
    copy (Karn's rule).

Run it with `make run SCENARIO=datacenter`.
"""

from __future__ import annotations

import re
import struct
from collections import deque
from typing import NamedTuple

from transportlab.api import Receiver, Sender

# The scenario your team chose: "leo", "datacenter" or "sensor". Gradescope
# grades your protocol on this scenario, and rejects it while this is empty.
SCENARIO = "datacenter"


# ---------------------------------------------------------------------------
# Packet format (shared by sender and receiver)
# ---------------------------------------------------------------------------

TYPE_DATA = 0
TYPE_ACK = 1

HEADER = struct.Struct("!BBI")           # type, send number, message ID -> 6 bytes
MAX_DATAGRAM = 1472                       # send_datagram's limit
MAX_PAYLOAD = MAX_DATAGRAM - HEADER.size  # 1466
MAX_SEND_NO = 255                         # the send number saturates here

SEND_TIMER = "send"   # fires the round-robin pump
LOG_SENDS = 20        # log only the first few sends; logs are capped at 5000 lines


def encode_data(msg_id: int, payload: bytes, send_no: int = 1) -> bytes:
    """DATA for one copy of a message; send_no 1 is its first transmission."""
    return HEADER.pack(TYPE_DATA, min(send_no, MAX_SEND_NO), msg_id) + payload


def encode_ack(msg_id: int, send_no: int) -> bytes:
    """ACK one copy of a message; send_no is echoed from that copy's DATA."""
    return HEADER.pack(TYPE_ACK, send_no, msg_id)


class Packet(NamedTuple):
    kind: int
    send_no: int     # which transmission of the message (1 = first)
    msg_id: int
    payload: bytes


def decode(data: bytes) -> Packet | None:
    """Return the packet, or None for one we don't understand."""
    if len(data) < HEADER.size:
        return None
    kind, send_no, msg_id = HEADER.unpack_from(data)
    if kind == TYPE_DATA:
        return Packet(kind, send_no, msg_id, bytes(data[HEADER.size:]))
    if kind == TYPE_ACK and len(data) == HEADER.size:
        return Packet(kind, send_no, msg_id, b"")
    return None


def host_index(name: str) -> int | None:
    """The trailing number of a host name: "sender3" -> 3, "receiver0" -> 0."""
    m = re.search(r"\d+$", name)
    return int(m.group()) if m else None


def rotated_order(receivers, me: int | None) -> list[str]:
    """Receivers in round-robin order, starting after receiver `me` and wrapping.

    Why start at me+1: if every sender sends one packet per step, sender i
    sends to receiver (i+1+t) mod 9 at step t. For each t that maps the nine
    senders one-to-one onto the nine receivers, so every receiver gets exactly
    one packet per step. Each sender's own receiver comes last (step 8), so
    all senders skip it at the same step and stay in line.

    Receivers without a number go last, in name order.
    """
    numbered = sorted((r for r in receivers if host_index(r) is not None), key=host_index)
    others = sorted(r for r in receivers if host_index(r) is None)
    if me is not None:
        start = next((i for i, r in enumerate(numbered) if host_index(r) > me), 0)
        numbered = numbered[start:] + numbered[:start]
    return numbered + others


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

class MySender(Sender):
    def on_start(self):
        self.me = host_index(self.name)
        receivers = [p for p in self.peers if p.startswith("receiver")] or list(self.peers)
        self.order = rotated_order(receivers, self.me)
        self.queues = {dst: deque() for dst in self.order}
        self.rr = 0   # position in self.order of the next destination to try

        self.data_sent = 0
        self.oversized = 0
        self.log("order", " ".join(self.order))

    def on_app_data(self, msg):
        if msg.size > MAX_PAYLOAD:
            # No fragmentation yet: skip it rather than crash the trial.
            self.oversized += 1
            self.log("oversized: skipping msg", msg.msg_id, "of", msg.size, "bytes")
            return
        self.queues[msg.dst].append(msg)
        # The harness offers a whole round in one burst, grouped by destination.
        # Sending here would go one destination at a time, so wait for a 0 s
        # timer, which fires after the burst is queued.
        if not self.timer_pending(SEND_TIMER):
            self.set_timer(SEND_TIMER, 0)

    def on_timer(self, name):
        if name == SEND_TIMER:
            self._pump()

    def _next_message(self):
        """Round robin: (dst, msg) from the next non-empty queue, or None."""
        n = len(self.order)
        for step in range(n):
            i = (self.rr + step) % n
            q = self.queues[self.order[i]]
            if q:
                self.rr = (i + 1) % n
                return self.order[i], q.popleft()
        return None

    def _pump(self):
        # Implementation 1 sends everything now. Implementation 2 sends one
        # packet, then sets SEND_TIMER for the pacing gap.
        while (picked := self._next_message()) is not None:
            self._transmit(*picked)

    def _transmit(self, dst, msg):
        self.send_datagram(encode_data(msg.msg_id, msg.payload), to=dst)
        self.data_sent += 1
        if self.data_sent <= LOG_SENDS:
            self.log("send", dst, msg.msg_id)

    def on_datagram(self, data, addr):
        pass   # Implementation 2: ACKs arrive here

    def on_stop(self):
        self.log("data_sent", self.data_sent, "oversized", self.oversized)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class MyReceiver(Receiver):
    def on_start(self):
        self.malformed = 0
        self.duplicates = 0

    def on_datagram(self, data, addr):
        pkt = decode(data)
        if pkt is None:
            self.malformed += 1
            return
        if pkt.kind != TYPE_DATA:
            return
        if self.delivered(pkt.msg_id):
            self.duplicates += 1   # a resend whose first copy got through
        else:
            self.deliver(pkt.msg_id, pkt.payload)
        # ACK only what we hold, so a bad copy is resent. A duplicate is ACKed
        # again because it means our first ACK may have been lost. The echoed
        # send number lets the sender time this exact copy.
        if self.delivered(pkt.msg_id):
            self.send_datagram(encode_ack(pkt.msg_id, pkt.send_no), to=addr)

    def on_stop(self):
        self.log("malformed", self.malformed, "duplicates", self.duplicates)


SENDER, RECEIVER = MySender, MyReceiver
