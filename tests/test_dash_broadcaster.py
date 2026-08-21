import asyncio

from ids_dashboard.broadcaster import AlertBroadcaster


class _FakeWebSocket:
    def __init__(self, fail: bool = False) -> None:
        self.accepted = False
        self.received = []
        self.fail = fail

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data) -> None:
        if self.fail:
            raise RuntimeError("send failed")
        self.received.append(data)


def test_connect_accepts_and_registers_client():
    async def run():
        broadcaster = AlertBroadcaster()
        ws = _FakeWebSocket()
        await broadcaster.connect(ws)
        assert ws.accepted
        assert broadcaster.client_count == 1

    asyncio.run(run())


def test_broadcast_sends_to_all_connected_clients():
    async def run():
        broadcaster = AlertBroadcaster()
        ws1, ws2 = _FakeWebSocket(), _FakeWebSocket()
        await broadcaster.connect(ws1)
        await broadcaster.connect(ws2)

        await broadcaster.broadcast({"alert_id": "a1"})

        assert ws1.received == [{"alert_id": "a1"}]
        assert ws2.received == [{"alert_id": "a1"}]

    asyncio.run(run())


def test_disconnect_removes_client():
    async def run():
        broadcaster = AlertBroadcaster()
        ws = _FakeWebSocket()
        await broadcaster.connect(ws)
        await broadcaster.disconnect(ws)
        assert broadcaster.client_count == 0

        await broadcaster.broadcast({"alert_id": "a1"})
        assert ws.received == []

    asyncio.run(run())


def test_broadcast_drops_stale_clients_without_raising():
    async def run():
        broadcaster = AlertBroadcaster()
        good, bad = _FakeWebSocket(), _FakeWebSocket(fail=True)
        await broadcaster.connect(good)
        await broadcaster.connect(bad)

        await broadcaster.broadcast({"alert_id": "a1"})  # must not raise despite `bad` failing

        assert good.received == [{"alert_id": "a1"}]
        assert broadcaster.client_count == 1  # `bad` was pruned

    asyncio.run(run())
