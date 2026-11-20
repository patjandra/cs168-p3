"""Data center backbone protocol: Implementation 1.

Each sender keeps one FIFO queue per destination and sends from them in
round-robin order, one packet per destination per turn. Sender i starts its
rotation at receiver i+1, so the nine senders do not all hit the same receiver
first. The receiver hands every DATA packet to the application.

Packet format (network byte order):

    DATA = [type: 1 byte = 0][message ID: 4 bytes][payload]
    ACK  = [type: 1 byte = 1][message ID: 4 bytes]

The ACK format is defined, but nothing sends one yet.

Where Implementation 2 goes:

    pacing                   MySender._pump: send one packet, then set
                             SEND_TIMER for the gap
    outstanding table + RTO  MySender._transmit: record the packet, start its
                             timer
    resends first            MySender._next_message: check a resend queue
                             before the destination queues
    ACK handling             MySender.on_datagram
    duplicate check + ACK    MyReceiver.on_datagram

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

HEADER = struct.Struct("!BI")            # type, message ID -> 5 bytes
MAX_DATAGRAM = 1472                       # send_datagram's limit
MAX_PAYLOAD = MAX_DATAGRAM - HEADER.size  # 1467

SEND_TIMER = "send"   # fires the round-robin pump
LOG_SENDS = 20        # log only the first few sends; logs are capped at 5000 lines


def encode_data(msg_id: int, payload: bytes) -> bytes:
    return HEADER.pack(TYPE_DATA, msg_id) + payload


def encode_ack(msg_id: int) -> bytes:
    return HEADER.pack(TYPE_ACK, msg_id)


class Packet(NamedTuple):
    kind: int
    msg_id: int
    payload: bytes


def decode(data: bytes) -> Packet | None:
    """Return the packet, or None for one we don't understand."""
    if len(data) < HEADER.size:
        return None
    kind, msg_id = HEADER.unpack_from(data)
    if kind == TYPE_DATA:
        return Packet(kind, msg_id, bytes(data[HEADER.size:]))
    if kind == TYPE_ACK and len(data) == HEADER.size:
        return Packet(kind, msg_id, b"")
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

    def on_datagram(self, data, addr):
        pkt = decode(data)
        if pkt is None:
            self.malformed += 1
            return
        if pkt.kind == TYPE_DATA:
            # Implementation 2 adds the duplicate check and the ACK back to addr.
            self.deliver(pkt.msg_id, pkt.payload)

    def on_stop(self):
        self.log("malformed", self.malformed)


SENDER, RECEIVER = MySender, MyReceiver
