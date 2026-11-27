# 先算 mean/std 再乘 √n：极小尺度下 std/√n 可能下溢为 0，
    # 直接做 mean / se 会抛 ZeroDivisionError；该写法对公共尺度
    # 不变，也不改变正常量级的逐位结果。
    try:
        t = (mean / std) * math.sqrt(max(0.0, n))
    except (OverflowError, ZeroDivisionError):
        t = None
    if n == 0:
        se = 0.0
    else:
        se = std / math.sqrt(max(0.0, n))