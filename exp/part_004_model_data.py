from __future__ import annotations



class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        cfg.validate()

        if cfg.mode not in Config.MODEL_MODES:
            raise ValueError(
                f"LanguageModel 不支持 mode={cfg.mode}"
                "（ngram 等统计基线不走本类）"
            )

        self.cfg = cfg

        self.token_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
        )

        self.position_embedding = nn.Embedding(
            cfg.seq_len,
            cfg.d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg.n_layers)
        ])

        self.final_norm = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
        )

        # 共享输入 embedding 和输出 projection。
        self.lm_head.weight = self.token_embedding.weight

        self.middle_start = cfg.middle_start
        self.middle_end = cfg.middle_end

        # 中间区域 Block 的元组。
        self.middle_blocks = tuple(
            self.blocks[i]
            for i in range(self.middle_start, self.middle_end)
        )

        # 与 position_embedding.weight 指向同一参数（共享存储）。
        # 经 object.__setattr__ 写入实例字典，不重复注册到
        # _parameters。
        object.__setattr__(
            self,
            "position_embedding_weight",
            self.position_embedding.weight,
        )

        self.middle_blocks_rest = self.middle_blocks[1:]

        # 是否对每个 Block 使用 activation checkpointing：
        # 开启后反向时重算前向激活。
        # 由训练入口根据命令行参数开启，默认关闭。
        self.grad_checkpoint = False

        if cfg.init_scheme == "name_hash":
            self._init_weights_by_name()
        else:
            self.apply(self._init_weights)

        # 初始化指纹在训练前计算，用于同种子跨结构核对：
        # shared 覆盖所有模式共有的参数，moe/ffn 分别记录
        # 两条结构分支独有的参数。指纹不进入 state_dict。
        self.init_fingerprints = self._initialization_fingerprints()

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _init_weights_by_name(self) -> None:
        """
        初始化不依赖模块构造顺序，只依赖运行种子与参数名。

        同一参数名在不同 mode 中会得到完全相同的初始权重，
        A/B/local_base 的共享层因此可以在同种子下逐位配对；
        mode 独有的参数按各自名字取独立随机流。使用 CPU
        generator 生成后再搬运，保证同一份代码在 CPU/CUDA
        上得到相同的初始值。
        """
        base_seed = int(torch.initial_seed())
        seen: set = set()
        for module_name, module in self.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            weight = module.weight
            if id(weight) in seen:
                continue
            seen.add(id(weight))
            name = (
                f"{module_name}.weight"
                if module_name
                else "weight"
            )
            digest = hashlib.sha256(
                f"{base_seed}:{name}".encode("utf-8")
            ).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int.from_bytes(digest[:8], "little")
                & ((1 << 63) - 1)
            )
            values = torch.empty(
                weight.shape,
                dtype=torch.float32,
            )
            values.normal_(
                0.0,
                0.02,
                generator=generator,
            )
            with torch.no_grad():
                weight.copy_(
                    values.to(
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                )

    def _initialization_fingerprints(self) -> Dict[str, object]:
        """按参数类别计算训练前初始权重的 sha256。"""
        buckets: Dict[str, List[Tuple[str, str]]] = {
            "shared": [],
            "moe": [],
            "ffn": [],
        }
        for name, parameter in self.named_parameters():
            if ".moe." in name:
                bucket = "moe"
            elif ".ffn." in name:
                bucket = "ffn"
            else:
                bucket = "shared"
            buckets[bucket].append(
                (name, _tensor_content_sha256(parameter))
            )

        fingerprints: Dict[str, object] = {
            "scheme": self.cfg.init_scheme,
        }
        for bucket, entries in buckets.items():
            digest = hashlib.sha256()
            for name, content_digest in sorted(entries):
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(content_digest.encode("ascii"))
                digest.update(b"\n")
            fingerprints[f"{bucket}_sha256"] = (
                digest.hexdigest() if entries else None
            )
            fingerprints[f"{bucket}_tensors"] = len(entries)
        return fingerprints

    def _call(self, fn, *args):
        """
        统一所有 Block 调用的入口。

        开启 grad_checkpoint 时对 Block 前向使用
        _BlockCheckpoint 的非重入 activation checkpointing，
        数值结果与直接调用一致。
        """
        # 梯度关闭时（evaluate 的 inference_mode / no_grad、以及
        # 纯前向探测）checkpoint 只会原样调用 fn，此时直接调用，
        # 数值结果完全一致。
        if self.grad_checkpoint and torch.is_grad_enabled():
            return _BlockCheckpoint.apply(fn, *args)
        return fn(*args)

    def invalidate_weight_caches(self) -> None:
        """
        使所有 MoE Block 的专家权重 stack 缓存失效。

        fused AdamW 原地更新参数时不会推进张量的版本号，
        _cached_weight_stack 的版本签名无法感知更新，因此
        optimizer.step() 后必须显式声明失效，否则后续前向会
        复用更新前的专家权重。
        """
        for block in self.blocks:
            if block.moe is not None:
                block.moe.invalidate_weight_cache()

    def run_independent_blocks(
        self,
        block: TransformerBlock,
        branches: torch.Tensor,
        top_k: int,
        scale: Optional[float] = None,
        attn_mask: Optional[SegmentAttentionMask] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每个 branch 独立执行完整 Transformer Block。

        输入:
            [batch, n_branches, time, dim]

        将 branch 维并入 batch 维以后：
            [batch * n_branches, time, dim]

        因而不同 branch 不会互相做 self-attention。
        """
        batch, n_branches, time, dim = branches.shape

        flat = branches.reshape(
            batch * n_branches,
            time,
            dim,
        )

        flat, aux_loss = block(
            flat,
            top_k,
            scale,
            attn_mask,
        )

        branches = flat.view(
            batch,
            n_branches,
            time,
            dim,
        )

        return branches, aux_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        batch, time = input_ids.shape

        if time > cfg.seq_len:
            raise ValueError(
                f"序列长度 {time} 超过 cfg.seq_len={cfg.seq_len}"
            )

        # token embedding 与按序列长度切片的位置 embedding 相加。
        x = F.embedding(input_ids, self.token_embedding.weight)
        x.add_(
            self.position_embedding_weight[:time]
        )

        # 文档级 segment attention mask（论文 3.1 节）。显式传入
        # segment_ids 时优先使用（测试/分析）；否则在启用
        # cfg.document_masks 时从 token 流里的文档分隔符推导。
        # B/B_early 把分支并入 batch 维，需要一份按 k1 复制的
        # 掩码；其余模式的 Attention 在原始 batch 上执行。
        if segment_ids is None and cfg.document_masks:
            segment_ids = document_segment_ids(
                input_ids,
                cfg.document_separator or (),
            )
        attn_bias: Optional[SegmentAttentionMask] = None
        branch_attn_bias: Optional[SegmentAttentionMask] = None
        if segment_ids is not None:
            attn_bias = SegmentAttentionMask(segment_ids)
            if cfg.mode in ("B", "B_early"):
                # k1==1 时分支仍会把 branch 维并入 batch 维执行
                # 后续中间 Block；掩码必须同样按 k1 复制，否则
                # 除第一个中间 Block 外的所有 Block 都会退化为
                # 跨文档的普通因果注意力。
                branch_attn_bias = SegmentAttentionMask(
                    segment_ids.repeat_interleave(
                        int(cfg.k1), 0
                    )
                )

        aux_losses: List[torch.Tensor] = []

        blocks = self.blocks
        call = self._call

        # dense 模式没有中间循环区域，全部 Block 由下面的 dense
        # 分支统一执行；跳过前后区域循环，避免边界 Block 重复执行。
        pre_end = 0 if cfg.mode == "dense" else self.middle_start

        # 残差缩放：循环区域外的子层用 residual_scale；循环
        # 区域内的子层在 paper 口径下额外除以循环次数 r。
        global_scale = cfg.residual_scale
        loop_scale = cfg.loop_residual_scale

        # 中间区域之前的层。
        for index in range(pre_end):
            x, aux = call(
                blocks[index],
                x,
                cfg.top_k,
                global_scale,
                attn_bias,
            )
            aux_losses.append(aux)

        middle = self.middle_blocks

        # ----------------------------------------------------
        # Dense baseline: 纯稠密 Transformer，每个 Block 只
        # 执行一次，无 MoE、无中间层循环。
        # ----------------------------------------------------
        if cfg.mode == "dense":
            for index in range(cfg.n_layers):
                x, aux = call(
                    blocks[index],
                    x,
                    cfg.top_k,
                    global_scale,
                    attn_bias,
                )
                aux_losses.append(aux)

        # ----------------------------------------------------
        # 论文的未循环 MoE 基线：middle 只执行一次。
        # ----------------------------------------------------
        elif cfg.mode == "base":
            for block in middle:
                x, aux = call(
                    block,
                    x,
                    cfg.top_k,
                    global_scale,
                    attn_bias,
                )
                aux_losses.append(aux)

        # ----------------------------------------------------
        # Original SMELT：middle 循环两次，循环区域内每个子层
        # 的残差更新按 1/r 缩放（paper 口径）。
        # ----------------------------------------------------
        elif cfg.mode == "smelt":
            for _ in range(2):
                for block in middle:
                    x, aux = call(
                        block,
                        x,
                        cfg.top_k,
                        loop_scale,
                        attn_bias,
                    )
                    aux_losses.append(aux)

        # ----------------------------------------------------
        # Scheme B / B 的等计算量机制对照 B_early
        # ----------------------------------------------------
        elif cfg.mode in ("B", "B_early"):
            if len(middle) == 0:
                raise RuntimeError(
                    "中间层不能为空"
                )

            # 第一轮：
            # 在第一个中间 Block 的 MoE 处创建 k1 个分支。
            branches, gates, _, aux = call(
                middle[0].first_branch_step,
                x,
                cfg.k1,
                loop_scale,
                attn_bias,
            )
            aux_losses.append(aux)

            # 第一轮剩余中间 Block。
            for block in self.middle_blocks_rest:
                branches, aux = call(
                    self.run_independent_blocks,
                    block,
                    branches,
                    cfg.k1,
                    loop_scale,
                    branch_attn_bias,
                )
                aux_losses.append(aux)

            # B_early：分支数与后续 Attention/MoE 执行次数保持
            # 与 B 一致，只在第一轮结束后提前合并，再把合并状态
            # 复制回 k1 个分支继续第二轮。这样合并位置是唯一
            # 改变的因素；第二轮各分支状态相同是提前合并的
            # 直接后果，而不是额外减少了计算量。
            if cfg.mode == "B_early":
                if branches.size(1) == 1:
                    merged = branches.squeeze(1)
                else:
                    branch_weights = gates.to(
                        branches.dtype
                    ).unsqueeze(-1).unbind(-2)
                    merged = branches[:, 0].mul(
                        branch_weights[0]
                    )
                    for index in range(1, len(branch_weights)):
                        merged.addcmul_(
                            branches[:, index],
                            branch_weights[index],
                        )
                branches = merged.unsqueeze(1).expand(
                    -1,
                    cfg.k1,
                    -1,
                    -1,
                ).contiguous()

            # 第二轮完整执行整个中间区域。
            for block in middle:
                branches, aux = call(
                    self.run_independent_blocks,
                    block,
                    branches,
                    cfg.k2,
                    loop_scale,
                    branch_attn_bias,
                )
                aux_losses.append(aux)

            # 只有第二轮结束后，才跨第一轮分支合并。
            #
            # gates:
            # [batch, time, k1]
            #
            # branches:
            # [batch, k1, time, dim]
            #
            if branches.size(1) == 1:
                # k1 == 1 时分支已包含完整残差，且第一轮专家
                # 输出已按选中概率 p1 缩放（Switch 风格）；
                # 不能再把整条分支乘一次 p1，否则残差也会被
                # 缩放。单个分支求和等价于恒等。
                x = branches.squeeze(1)
            else:
                # 归一化后的 top-k gate 在每个 token 上和为 1。
                # 跨分支收缩按分支顺序加权累加。
                branch_weights = gates.to(
                    branches.dtype
                ).unsqueeze(-1).unbind(-2)

                x = branches[:, 0].mul(
                    branch_weights[0]
                )

                for index in range(1, len(branch_weights)):
                    x.addcmul_(
                        branches[:, index],
                        branch_weights[index],
                    )

        # ----------------------------------------------------
        # Local base / Scheme A / local_base_k2
        # ----------------------------------------------------
        else:
            delayed = cfg.mode == "A"
            # local_base_k2 是「只改变合并时机」的机制对照：
            # 第二轮固定 k2，不跟随 A 的 k1*k2 专家计算量。
            second_round_width = (
                cfg.k2 if cfg.mode == "local_base_k2" else None
            )

            for block in middle:
                x, aux1, aux2 = call(
                    block.local_two_stage,
                    x,
                    cfg.k1,
                    cfg.k2,
                    delayed,
                    loop_scale,
                    second_round_width,
                    attn_bias,
                )
                aux_losses.extend((aux1, aux2))

        # 中间区域之后的层（dense 模式已在上面统一执行）。
        post_start = (
            cfg.n_layers if cfg.mode == "dense" else self.middle_end
        )
        for index in range(post_start, cfg.n_layers):
            x, aux = call(
                blocks[index],
                x,
                cfg.top_k,
                global_scale,
                attn_bias,
            )
            aux_losses.append(aux)

        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head.weight)

        aux_loss = torch.stack(aux_losses).mean()

        output = {
            "logits": logits,
            "aux_loss": aux_loss,
        }

        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    cfg.vocab_size,
                ),
                targets.reshape(-1),
            )

            total_loss = (
                lm_loss
                + cfg.balance_coef * aux_loss
            )

            output.update({
                "lm_loss": lm_loss,
                "loss": total_loss,
            })

        return output



# ============================================================
# Dataset
# ============================================================

# 近重复（near-duplicate）群组审计/清洗口径。
#
# 精确群组指纹与内容定义分块只能发现「完全相同」的内容；同一篇
# 文档的两次修订、转载时改标题/改一两句话，shingle 集合仍高度
# 重合但指纹不同。这里用字节 k-shingle 的 Jaccard 相似度定义
# 近重复：|S_a ∩ S_b| / |S_a ∪ S_b|。为把成本压到与语料大小线性
# 相关，先对每个群组取 shingle 哈希的 bottom-k 草图（MinHash
# 家族的最小哈希样本），用草图哈希做倒排索引生成候选，再对候选
# 用完整 shingle 集合复核精确 Jaccard；只有精确值达到阈值才删除。
# 同一口径也用于评估集内部/val↔test 的只读近重复审计（不删除）。
_NEAR_DUP_DEFAULT_THRESHOLD = 0.8

_NEAR_DUP_DEFAULT_SHINGLE_BYTES = 8

_NEAR_DUP_DEFAULT_SKETCH_SIZE = 128

_NEAR_DUP_MAX_SHINGLE_BYTES = 64

_NEAR_DUP_MAX_SKETCH_SIZE = 4096

# 单个群组过大时完整 shingle 集合的内存不可控；显式跳过并在
# 审计里记录，而不是静默截断内容（截断会把长文档误判为短文档）。
_NEAR_DUP_MAX_GROUP_BYTES = 4 * 1024 * 1024

# 精确复核缓存：候选对很少，但同一评估群组可能被很多训练群组
# 比较；缓存按近似字节数限界，避免大语料把内存吃满。
_NEAR_DUP_EXACT_CACHE_BYTES = 64 * 1024 * 1024

# 64 位滚动多项式哈希（模 2^64）。碰撞概率在千万级 shingle 下
# 约 1e-6 量级；审计同时记录这是近似哈希而非逐字节比较。
_NEAR_DUP_HASH_BASE = 0x100000001B3

_NEAR_DUP_HASH_MASK = (1 << 64) - 1



# 群组分隔符允许的反斜杠转义：\n / \t / \r / \0 / \\ / \xHH / \uXXXX。
_GROUP_SEPARATOR_SIMPLE_ESCAPES = {
    "n": 10,
    "t": 9,
    "r": 13,
    "0": 0,
    "\\": 92,
    "'": 39,
    '"': 34,
}



def parse_group_separator(text: str) -> bytes:
    """
    把命令行传入的群组分隔符解析为字节序列。

    支持常见反斜杠转义（\\n、\\t、\\r、\\0、\\\\、\\xHH、\\uXXXX），
    其余字符按 UTF-8 编码；解析结果不能为空。分隔符用于把语料
    切成「群组」（如同一文档/同一用户的一段内容），群组是划分
    的最小单位，绝不允许跨 train/val/test 边界。
    """
    result = bytearray()
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char != "\\":
            try:
                result.extend(char.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(
                    "分隔符包含无法编码为 UTF-8 的字符: "
                    f"U+{ord(char):04X}"
                ) from error
            index += 1
            continue

        if index + 1 >= length:
            raise ValueError("分隔符不能以单个反斜杠结尾")

        escape = text[index + 1]

        if escape in _GROUP_SEPARATOR_SIMPLE_ESCAPES:
            result.append(_GROUP_SEPARATOR_SIMPLE_ESCAPES[escape])
            index += 2
            continue

        if escape in ("x", "u"):
            digits = 2 if escape == "x" else 4
            hex_part = text[index + 2:index + 2 + digits]
            if (
                len(hex_part) != digits
                or any(
                    digit not in "0123456789abcdefABCDEF"
                    for digit in hex_part
                )
            ):
                raise ValueError(
                    f"\\{escape} 转义需要 {digits} 位十六进制数字"
                )
            codepoint = int(hex_part, 16)
            if escape == "u":
                try:
                    result.extend(
                        chr(codepoint).encode("utf-8")
                    )
                except UnicodeEncodeError as error:
                    raise ValueError(
                        "\\u 转义指向无法编码为 UTF-8 的"
                        f"码点: U+{codepoint:04X}"
                    ) from error
            else:
                result.append(codepoint)
            index += 2 + digits
            continue

        raise ValueError(
            f"不支持的分隔符转义: \\{escape}"
        )

    if not result:
        raise ValueError("群组分隔符不能为空")

    return bytes(result)



# `--clean-train-against-test` 与
# `--allow-eval-content-for-cleaning` 是同义入口：只要任一为真，
# 训练集清洗就会读取 test 内容。所有下游（ByteDataset、resume、
# comparison_protocol、外部预注册核对）必须看到同一个归一化值，
# 否则同一种清洗口径会因使用不同别名被误报为协议不一致。
_CLEAN_AGAINST_TEST_ALIASES = (
    "clean_train_against_test",
    "allow_eval_content_for_cleaning",
)



def _clean_against_test_effective(args: object) -> bool:
    return bool(
        getattr(args, "clean_train_against_test", False)
        or getattr(args, "allow_eval_content_for_cleaning", False)
    )



def _normalize_clean_against_test(args: object) -> bool:
    effective = _clean_against_test_effective(args)
    setattr(args, "clean_train_against_test", effective)
    setattr(args, "allow_eval_content_for_cleaning", effective)
    return effective



def _saved_clean_against_test_effective(
    saved_args: object,
) -> bool:
    """
    从检查点/配置的 args 快照还原「清洗是否读取 test」。

    旧代码可能只记录了两个同义入口中的一个（且另一个为
    False）；必须按任一为真归一化，不能因为
    ``clean_train_against_test`` 键存在就忽略
    ``allow_eval_content_for_cleaning=True``。两个键都缺的
    更老检查点按历史行为回退：启用了去重/共享块清洗即读取
    test。
    """
    if not isinstance(saved_args, dict):
        return False
    if (
        "clean_train_against_test" in saved_args
        or "allow_eval_content_for_cleaning" in saved_args
    ):
        return bool(
            saved_args.get("clean_train_against_test", False)
            or saved_args.get(
                "allow_eval_content_for_cleaning", False
            )
        )
    return bool(
        saved_args.get("deduplicate_train_groups", False)
        or saved_args.get("drop_train_shared_chunks", False)
        or saved_args.get(
            "drop_train_near_duplicate_groups", False
        )
    )



def _iter_group_boundaries(
    buf: bytearray,
    separator: bytes,
    start: int = 0,
):
    """
    依次产出群组边界位置（每个分隔符之后的字节位置）与文件末尾。

    边界是群的起始字节；文件末尾 N 永远是一个合法边界。
    只产出 > start 的边界。
    """
    if not separator:
        # bytes.find(b"") 永远返回 position，len=0 会让循环原地
        # 踏步；调用方必须先拒绝空分隔符，而不是挂死进程。
        raise ValueError("群组分隔符不能为空字节串")
    position = start

    while True:
        found = buf.find(separator, position)
        if found < 0:
            break
        end = found + len(separator)
        if start < end < len(buf):
            yield end
        position = end

    if len(buf) > start:
        yield len(buf)



def _iter_group_spans(
    buf: bytearray,
    separator: bytes,
    start: int,
    end: int,
):
    """
    依次产出 [start, end) 内每个群组的内容区间（不含分隔符本身）。

    与 _iter_group_boundaries 同一口径：群组以分隔符之后的位置
    为起点（首个群组为 start），以分隔符起点或 end 为终点；空
    群组不产出区间。文本完整性审计用它把用户显式配置的群组
    分隔符（可能是 NUL 或非 UTF-8 字节）排除在损坏统计之外。
    """
    if not separator:
        raise ValueError("群组分隔符不能为空字节串")
    position = start

    while position < end:
        found = buf.find(separator, position, end)
        if found < 0:
            yield position, end
            return
        if found > position:
            yield position, found
        position = found + len(separator)



def _count_groups(
    buf: bytearray,
    separator: bytes,
    cuts: List[int],
) -> List[int]:
    """
    统计每个 split 内部的完整（非空）群组数量（cuts 必须落在
    群组边界上）。

    与 _iter_group_spans 同一口径：空群组（前导/连续分隔符
    之间）不计入「完整群组」；群组按起始字节归属 split，cuts
    本身是群组起点，因此恰好等于切分点的群组只计入右侧
    split。末尾分隔符之后的空群组同样不计入。
    """
    first, second = cuts[0], cuts[-1]
    counts = [0, 0, 0]

    for span_start, _ in _iter_group_spans(
        buf,
        separator,
        0,
        len(buf),
    ):
        if span_start < first:
            counts[0] += 1
        elif span_start < second:
            counts[1] += 1
        else:
            counts[2] += 1

    return counts



def _group_content_digest(
    raw: memoryview,
    start: int,
    end: int,
) -> bytes:
    """
    群组内容的精确指纹（blake2b-128）。

    群组级去重/泄露审计使用完整内容指纹，不受窗口采样步长
    影响：短群组、跨 split 错位重复和窗口审计抽不到的重复
    文档都能被精确识别。指纹只用于审计/清洗决策，不进入
    模型特征。
    """
    return hashlib.blake2b(
        raw[start:end],
        digest_size=16,
    ).digest()



def _merge_byte_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """合并重叠/相邻的 [start, end) 区间，输出按起点升序。"""
    if not intervals:
        return []

    ordered = sorted(
        (start, end)
        for start, end in intervals
        if end > start
    )
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged



def _effective_train_fingerprint(
    raw: memoryview,
    train_end: int,
    excluded_spans: List[Tuple[int, int]],
) -> str:
    """
    对「实际进入训练的 train 内容」做独立指纹。

    只比对整体数据 sha256、split 边界与清洗开关还不够：同一份
    数据、同一组开关下，清洗实现变化（或 bug）会剔除不同区间，
    实际参与训练/计数的字节随之改变，resume 或跨运行对比却仍
    显示协议一致。这里把 train 的原始区间、清洗区间清单与保留
    内容的长度前缀一起哈希，任何一项变化都会得到不同指纹。
    """
    merged = _merge_byte_intervals(excluded_spans)
    manifest = json.dumps(
        {
            "train_end": int(train_end),
            "excluded_spans": [
                [int(start), int(end)]
                for start, end in merged
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    hasher = hashlib.sha256()
    hasher.update(len(manifest).to_bytes(8, "little"))
    hasher.update(manifest)

    for start, end in _subtract_byte_intervals(
        0,
        train_end,
        merged,
    ):
        hasher.update(
            (end - start).to_bytes(8, "little")
        )
        hasher.update(raw[start:end])

    return hasher.hexdigest()



def _valid_window_start_intervals(
    total_starts: int,
    seq_len: int,
    excluded_spans: List[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """
    计算「窗口完全不触碰被清洗字节」的合法起点区间。

    窗口起点 s 覆盖训练字节 [s, s + seq_len]（含下一 token
    监督位），因此与排除区间 [start, end) 相交当且仅当
    s ∈ [start - seq_len, end - 1]。返回按起点升序的闭区间
    列表；没有排除区间时返回 None（调用方走无掩码快路径）。
    """
    if not excluded_spans:
        return None

    blocked: List[Tuple[int, int]] = []
    for start, end in _merge_byte_intervals(excluded_spans):
        low = max(0, start - seq_len)
        high = min(total_starts - 1, end - 1)
        if low <= high:
            blocked.append((low, high))

    if not blocked:
        return None

    valid: List[Tuple[int, int]] = []
    cursor = 0
    for low, high in blocked:
        if cursor < low:
            valid.append((cursor, low - 1))
        cursor = max(cursor, high + 1)
    if cursor < total_starts:
        valid.append((cursor, total_starts - 1))

    return valid



def _plan_train_group_dedup(
    raw: bytearray,
    separator: bytes,
    train_end: int,
    eval_ranges: Dict[str, Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, object]]:
    """
    规划训练集群组清洗：只从 train 中剔除群组，绝不改动
    val/test。

    剔除规则：
    - 内容与 val 或 test 中任一完整群组完全相同（跨 split
      重复，属于泄露）；
    - 与训练集中更早出现的群组完全相同（保留第一次出现）。
    返回被剔除群的字节区间与可追踪的统计。调用方只会在
    训练窗口采样时屏蔽这些区间，验证/测试数据保持原样。
    """
    eval_digests: Dict[str, set] = {
        "val": set(),
        "test": set(),
    }
    for split, (begin, end) in eval_ranges.items():
        if end <= begin:
            continue
        for span_start, span_end in _iter_group_spans(
            raw,
            separator,
            begin,
            end,
        ):
            eval_digests[split].add(
                _group_content_digest(raw, span_start, span_end)
            )

    seen = set()
    excluded: List[Tuple[int, int]] = []
    stats: Dict[str, object] = {
        "enabled": True,
        "policy": (
            "drop train groups duplicated in val/test or earlier "
            "in train; val/test untouched"
        ),
        "groups_total": 0,
        "groups_kept": 0,
        "groups_dropped": 0,
        "dropped_bytes": 0,
        "dropped_duplicate_of": {
            "val": 0,
            "test": 0,
            "train": 0,
        },
        "examples": [],
    }

    for span_start, span_end in _iter_group_spans(
        raw,
        separator,
        0,
        train_end,
    ):
        stats["groups_total"] += 1
        digest = _group_content_digest(
            raw,
            span_start,
            span_end,
        )

        reason: Optional[str] = None
        if digest in eval_digests["val"]:
            reason = "val"
        elif digest in eval_digests["test"]:
            reason = "test"
        elif digest in seen:
            reason = "train"

        if reason is None:
            seen.add(digest)
            stats["groups_kept"] += 1
            continue

        excluded.append((span_start, span_end))
        stats["groups_dropped"] += 1
        stats["dropped_bytes"] += span_end - span_start
        stats["dropped_duplicate_of"][reason] += 1
        if len(stats["examples"]) < 5:
            stats["examples"].append(
                [span_start, span_end, reason]
            )

    stats["post_clean_train_bytes"] = (
        train_end - int(stats["dropped_bytes"])
    )
    return excluded, stats



def _snap_group_split(
    buf: bytearray,
    separator: bytes,
    nominal_train: int,
    nominal_val_end: int,
    min_len: int,
    has_test: bool,
) -> Tuple[int, int]:
    """
    把名义切分位置吸附到群组边界：train/val/test 的分界点都是
    某个群组的起始字节，任何群组整体落在一个 split 内。

    在全部群组边界上求全局最优的可行组合（目标为
    |cut_train - nominal_train| + |cut_val - nominal_val_end|），
    找不到合法组合时抛出 ValueError（宁可报错，也不产生一个
    拆散群组的划分）。

    对全部候选组合按偏离量取全局最小。
    """
    if buf.find(separator) < 0:
        raise ValueError(
            "语料中没有出现群组分隔符，无法按群组划分；"
            "请检查 --group-separator 是否正确"
        )

    total = len(buf)

    # val 右端上界：有 test 时 test 也要至少 min_len 字节；
    # 无 test 时 val 延伸到文件末尾（cut_val = total 是离
    # nominal_val_end = total 最近的边界）。
    upper_val = total if not has_test else total - min_len

    best: Optional[Tuple[int, int]] = None
    best_key: Optional[Tuple[int, int, int, int, int]] = None

    # 窗口内是「> 当前 cut_val - min_len」的边界；被移出窗口的
    # 边界里，最后一个 >= min_len 的可作 train 右端。
    window: deque = deque()
    trailing: Optional[int] = None
    left_candidate: Optional[int] = None
    right_candidate: Optional[int] = None

    def _consider(cut_val: int) -> None:
        nonlocal best, best_key

        if cut_val > upper_val:
            return
        # 没有 test split 时 val 必须延伸到文件末尾：否则
        # ByteDataset 会把 cut_val 之后的字节当成 test，凭空
        # 产生一个 test_fraction=0 却非空的测试集。这里不能
        # 只把 total 当作上界，必须把 cut_val 固定为 total。
        if not has_test and cut_val != total:
            return

        lower = cut_val - min_len
        candidates: List[int] = []

        for candidate in (left_candidate, right_candidate):
            if candidate is not None and candidate <= lower:
                candidates.append(candidate)

        if trailing is not None:
            candidates.append(trailing)

        if not candidates:
            return

        cut_train = min(
            candidates,
            key=lambda cut: (abs(cut - nominal_train), cut),
        )
        deviation_train = abs(cut_train - nominal_train)
        deviation_val = abs(cut_val - nominal_val_end)
        key = (
            deviation_train + deviation_val,
            deviation_train,
            cut_train,
            deviation_val,
            cut_val,
        )

        if best_key is None or key < best_key:
            best_key = key
            best = (cut_train, cut_val)

    for boundary in _iter_group_boundaries(buf, separator, 0):
        # 先把已进入「cut_val - min_len」之内的边界移出窗口；
        # 最后一个移出的边界就是当前 cut_val 下最右的可行
        # 左侧 train 右端。
        while window and window[0] <= boundary - min_len:
            popped = window.popleft()
            if popped >= min_len:
                trailing = popped

        _consider(boundary)

        # 该边界之后可作 train 右端（需要 >= min_len）。
        if boundary >= min_len:
            if boundary <= nominal_train:
                left_candidate = boundary
            elif right_candidate is None:
                right_candidate = boundary

        window.append(boundary)

    if best is None:
        raise ValueError(
            "语料按群组分隔符划分后，无法同时保证 train/val/test "
            f"各自至少 {min_len} 字节（seq_len + 1）；"
            "请调整 --val-fraction / --test-fraction、减小 "
            "--seq-len、检查分隔符，或合并过小的群组"
        )

    return best



def _split_isolation_record(
    split_bounds: Dict[str, Tuple[int, int]],
    total_bytes: int,
) -> Dict[str, object]:
    """
    生成可复现的时序隔离记录：三段的绝对字节边界、顺序、
    是否首尾相接且覆盖全文、任意两段的重叠字节数。

    只记录边界与结构，不读取或改变数据；写入 config 的
    data.split_isolation 并作为受控变量核对。这样即使源文件
    sha256 相同，切分顺序/群组吸附/边界实现变化也能被对比
    与 resume 发现。
    """
    order = ("train", "val", "test")
    bounds = {
        split: [
            int(split_bounds[split][0]),
            int(split_bounds[split][1]),
        ]
        for split in order
    }
    contiguous = bounds["train"][0] == 0
    for left, right in zip(order, order[1:]):
        if bounds[left][1] != bounds[right][0]:
            contiguous = False
    covers_all = (
        contiguous
        and bounds["test"][1] == int(total_bytes)
        and all(
            start <= end
            for start, end in bounds.values()
        )
    )
    # 逐对求交，而不是 covered - span：后者在同时存在空隙与
    # 重叠时会相互抵消（例如 train=[0,30)、val=[20,50)、
    # test=[60,70) 会被算成 0，而 train∩val 实际为 10 字节）。
    ranges = list(bounds.values())
    overlap_bytes = 0
    for left_index in range(len(ranges)):
        for right_index in range(left_index + 1, len(ranges)):
            left_start, left_end = ranges[left_index]
            right_start, right_end = ranges[right_index]
            overlap_bytes += max(
                0,
                min(left_end, right_end)
                - max(left_start, right_start),
            )

    return {
        "split_order": list(order),
        "bounds": bounds,
        "contiguous": contiguous,
        "covers_all_bytes": covers_all,
        "overlap_bytes": overlap_bytes,
        "policy": "prefix_contiguous_no_shuffle",
    }



def audit_entity_manifest(
    buf: bytes,
    group_separator: Optional[bytes],
    split_bounds: Dict[str, Tuple[int, int]],
    manifest_path: str,
) -> Dict[str, object]:
    """用显式实体 ID 检查同一实体是否跨越 split。

    ``--group-separator`` 只保证「分隔符之间的连续片段」不被
    切开，不能识别同一用户/作者/文档版本在不相邻片段中的
    重复出现。实体清单按群组序号（0 基）给出 entity_id
    （JSON 数组或 {"groups": [...]}）；审计把同一 entity 的
    全部群组映射到 split，任何实体同时出现在两个 split 即
    报告为泄露级问题。

    没有群组分隔符时无法建立群组序号，直接返回 skipped。
    """
    if group_separator is None:
        return {
            "available": False,
            "reason": "需要 --group-separator 才能建立群组序号",
        }
    with open(manifest_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        values = payload.get("groups") or payload.get("entity_ids")
    else:
        values = payload
    if not isinstance(values, list) or not values:
        raise ValueError(
            "实体清单必须是非空 JSON 数组（按群组序号给 entity_id）"
        )
    spans = list(
        _iter_group_spans(
            buf,
            group_separator,
            0,
            len(buf),
        )
    )
    if len(values) != len(spans):
        raise ValueError(
            "实体清单条目数（"
            f"{len(values)}）与群组数（{len(spans)}）不一致"
        )
    split_of = {}
    ordered = sorted(
        split_bounds.items(), key=lambda item: item[1][0]
    )
    for index, (start, _end) in enumerate(spans):
        split_name = None
        for name, (begin, finish) in ordered:
            if begin <= start < finish:
                split_name = name
                break
        split_of[index] = split_name
    entities: Dict[str, set] = {}
    for index, entity in enumerate(values):
        entity_id = str(entity)
        entities.setdefault(entity_id, set()).add(
            split_of.get(index)
        )
    crossing = {
        entity: sorted(
            str(split) for split in splits if split is not None
        )
        for entity, splits in entities.items()
        if len([split for split in splits if split is not None]) > 1
    }
    return {
        "available": True,
        "manifest": manifest_path,
        "groups": len(spans),
        "entities": len(entities),
        "cross_split_entities": len(crossing),
        "examples": list(crossing.items())[:20],
        "note": (
            "实体 ID 来自外部清单；分隔符本身不提供实体隔离。"
            "跨 split 实体是泄露级问题，应重新切分或剔除，"
            "而不是声称「群组不跨界」已足够"
        ),
    }
