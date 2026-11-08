from __future__ import annotations



def read_time_order_manifest(
    path: str,
) -> Dict[str, object]:
    """读取并校验时间顺序清单（每行一个时间戳）。

    支持 JSONL（``{"timestamp": "..."}``）或纯文本
    （每行一个 ISO-8601 时间戳）。用于 --split-policy
    chronological：只有清单按文件顺序单调不减时，前缀切分
    才可以称为「按时间切分」；否则拒绝启动。
    """
    records: List[str] = []
    parsed: List[datetime.datetime] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # 纯文本清单：整行就是一个 ISO-8601 时间戳。
                payload = text
            if isinstance(payload, dict):
                value = payload.get("timestamp")
            elif isinstance(payload, str):
                value = payload
            else:
                # 全数字的 ISO-8601 basic 日期（如 20200101）会被
                # json.loads 解析成整数；它不是 JSON 清单，仍应
                # 按纯文本时间戳回退，而不是报「缺少 timestamp」。
                value = text
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"时间清单第 {line_number} 行缺少 timestamp"
                )
            try:
                timestamp = datetime.datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError as error:
                raise ValueError(
                    f"时间清单第 {line_number} 行不是 ISO-8601: "
                    f"{value!r}（{error}）"
                )
            records.append(value)
            parsed.append(timestamp)
    if not records:
        raise ValueError("时间清单为空")

    # 带时区与不带时区的时间戳不能直接比较（Python 会抛
    # TypeError）。这里显式拒绝混合清单，要求用户统一口径，
    # 而不是假定 naive 时间戳的时区或让进程裸崩溃。
    awareness = {
        "aware"
        if timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        else "naive"
        for timestamp in parsed
    }
    if len(awareness) > 1:
        raise ValueError(
            "时间清单不能混用带时区与不带时区的 ISO-8601 "
            "时间戳；请统一写成同一时区（或全部省略时区）"
        )

    violations = [
        index
        for index in range(1, len(parsed))
        if parsed[index] < parsed[index - 1]
    ]
    min_index = min(
        range(len(parsed)),
        key=parsed.__getitem__,
    )
    max_index = max(
        range(len(parsed)),
        key=parsed.__getitem__,
    )
    return {
        "path": path,
        "records": len(records),
        "min_timestamp": records[min_index],
        "max_timestamp": records[max_index],
        "monotonic_non_decreasing": not violations,
        "order_violations": violations[:20],
        "order_violation_count": len(violations),
        "note": (
            "清单按语料文件顺序逐条给出时间戳；只有单调不减时"
            "前缀切分才是时间切分。文件顺序本身不构成时间证据"
        ),
    }



# 固定评估批设备侧结果的总字节上限。
_EVAL_BATCH_CACHE_MAX_BYTES = 64 * 1024 * 1024
