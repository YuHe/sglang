# fake 传输后端包入口，导出用于单机模拟测试的 KV 管理器/收发器（无真实网络传输）
from sglang.srt.disaggregation.fake.conn import (
    FakeKVManager,
    FakeKVReceiver,
    FakeKVSender,
)
