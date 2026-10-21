# 实验日志：你的指令、我做的事、代码改动

每条记录：时间｜你的指令（原话或摘要）｜我做了什么｜改了哪些代码｜结果。

---

## 2026-10-19 ~ 10-20　阶段 0（之前的会话，摘要）

- **你的指令**：在 CCPP 上做记忆压缩实验，A 无记忆 / B 完整记忆 / C 压缩记忆；"代码一定要在计算节点跑，严禁在登陆节点跑代码"；"计算节点分区优先用主分区，满了用备用分区"（原话含分区名，已隐去）；"下载一个 4b 到当前目录吧，然后如果要创建虚拟环境，就在当前目录创建"。
- **我做的事**：建环境、下载 Qwen3-4B、划分数据、老师检查、Cartridges 正确性检查、主实验（3 个种子）、统计检验、伪标签回归对照。
- **代码**：scripts/ccpp_common.py、prepare_data.py、teacher_gate.py、make_teacher_data.py、train_cartridge.py、eval_cartridge.py 等；Cartridges 补丁 P1（缓存读回）、P2（到步数停止）、P4（fp32 主参数）。
- **结果**：Z 恢复了相对无记忆约 97% 的 RMSE 改进（Z 10.01，B 9.79）。

## 2026-10-20　Word 报告与对照实验

- **你的指令**："你可以单独把这次结果写个 word 给我吗"；同意做打乱/平移 +10 对照；"只平移低温区"。
- **我做的事**：生成 Word 报告；跑对照实验（打乱老师、无关文本初始化、蒸馏 256/64 条、随机输入、整体 +10、低温区 +10）。
- **代码**：新增 scripts/make_stage0_docx.js、jobs/96_stage0_docx.sh；make_teacher_data.py 加 --shuffle_pe / --n_synth / --pe_shift / --low_at_only；ccpp_common.py 加 shifted_pe；train_cartridge.py 加 --limit；eval_cartridge.py 加 --pe_shift / --low_at_only；新增 jobs/10–14、scripts/ctrl_report.py、rshift_report.py。
- **结果**：Z 跟着老师变（包括只在低温区变），结论表述为"行为蒸馏"。

## 2026-10-20　报告修改

- **你的指令**："还原之前的，这个你解释的不好，除了按那份意见改的地方：5-NN 的两个数…低温区实验…这个，其他还原"。
- **我做的事**：删掉"一页看懂"首页，只保留 5-NN 标注和低温区分开表述两处修改。
- **代码**：scripts/make_stage0_docx.js。

## 2026-10-20 18:00　第一类验证

- **你的指令**："你分的不好，我希望你先完成第一类的几个验证，再说扩大试验的事，验证代码工作我自己来"。
- **我做的事**：提交三项验证：关键对照补种子 43/44、换记忆 r1/r2、笔记大小 k=16/256。
- **代码**：新增 jobs/15_ctrl_seeds.sh、16_other_memory.sh、17_k_sweep.sh、18_robust_report.sh、scripts/robust_report.py。
- **结果（10-20 23:31 完成）**：全部通过。3 个种子稳定；r1/r2 恢复率 108%/103%；k=16/64/256 都约 10.0。

## 2026-10-21 00:05　Word 报告补第一类结果

- **你的指令**："跑完了吗"（结果出来后按之前约定补进报告）。
- **我做的事**：报告新增第八节"第一类验证"，原限定/下一步顺延为九、十。
- **代码**：scripts/make_stage0_docx.js。

## 2026-10-21 00:15　扩大试验：512 行记忆

- **你的指令**："你开始扩大试验吧，你觉得要多少数据跑一次的结果才可信？"
- **我做的事**：答复"可信靠 3 份记忆 × 3 种子，测试 2000 题已够"；提交 512 行记忆实验（r0 含 k=16/64/256，r1，r2）。
- **代码**：新增 jobs/19_scale512.sh、jobs/20_scale_report.sh、scripts/scale_report.py。
- **结果**：r0、r1 完成（r0：B 6.64，Z 6.49；r1：B 6.81，Z 6.49）；r2 作业被暂停（JobHeldUser，非我操作），待你决定是否恢复。

## 2026-10-21 00:50　10 倍记忆可行性检查（YaRN）

- **你的指令**：老师说"现在就是 1000 对…别人要很大的数据去跑，相当于最开始的数据就要很大"；"可以，先确认 1，2"；"你去做吧"。
- **我做的事**：扩大记忆池（从备用数据抽 1024/2048/2560 行记忆，不动测试/开发/蒸馏题），开 YaRN 做老师检查。
- **代码**：新增 scripts/make_large_subsets.py、jobs/21_gate_large.sh；ccpp_common.py 加 YARN_KWARGS、自动读取 memory_subsets_large.json；teacher_gate.py 加 --yarn、--chunk（分段读入）和分段读入自检（原文件备份为 teacher_gate.py.bak_stage0）。
- **结果**：自检通过，但开 YaRN 后模型不直接给数字（解析率 0），未通过。
- **注意**：队列中另有一个同类作业 mz_big_gate（scripts/big_gate.py），不是本会话提交的，结果一致。

## 2026-10-21 11:20　换模型 Qwen3-4B-Instruct-2507

- **你的指令**："我觉得是不是可以直接换成 B 直接试呢？"
- **我做的事**：下载 Qwen3-4B-Instruct-2507（原生 26 万词元），用它做 256–2560 行老师检查。
- **代码**：ccpp_common.py 加 MZ_MODEL 环境变量选模型（默认仍是 Qwen3-4B）、HAS_THINKING（新模型无思考标记）；新增 jobs/22_gate_2507.sh。
- **结果（12:34 完成）**：能直接给数字，但记忆越长越差：B 误差 256 行 11.66 → 2048 行 23.55，2048 行以上接近打乱记忆，未通过。
- **另**：我在登录节点误执行了一次空的 `python3 -c ""`，违反规则，已向你说明。

## 2026-10-21 14:40　Google 文档与 skill

- **你的指令**："把整个代码和流程帮我传到我的 GitHub"；"链接我的 Google doc 生成一个 skill"；"你创建一个新的"；"新文档"。
- **我做的事**：新建 Google 文档"记忆压缩实验：流程与规则"；新建 skill .claude/skills/mz-workflow/SKILL.md（链接该文档）。GitHub 推送等待你存 token。
- **代码**：无实验代码改动。

## 2026-10-21　建立本日志

- **你的指令**："最好就是更新每一次你的代码的改动，我对你的命令，这样我可以 track"。
- **我做的事**：写本日志 docs/experiment_log.md；以后每次改代码或提交作业都追加一条。

## 2026-10-21 15:10　准备推送 GitHub

- **你的指令**："你和 GPT 都能看到"——同意用 GitHub 做日志；给出仓库 https://github.com/1karess/memory-compression.git。
- **我做的事**：写好 README（项目说明、流程、主要结果）、.gitignore（不传模型、训练输出、虚拟环境、日志、第三方代码）、推送作业。Cartridges 只传补丁文件 patches/cartridges.diff。
- **代码**：新增 README.md、.gitignore、jobs/23_git_push.sh（token 从 ~/.mz_gh_token 读取，不写进仓库或日志）。
- **状态**：等你在终端存好 token 后提交推送作业。

## 2026-10-21 15:48　推送到 GitHub

- **你的指令**：在终端存好 token。
- **我做的事**：提交推送作业 31346288，首次推送成功（commit b6d6a64，分支 main）。
- **以后**：每次改动后追加日志，再用 `MSG="说明" sbatch jobs/23_git_push.sh` 推送。

## 2026-10-21 16:10　GitHub 去除个人信息

- **你的指令**："我和别人共用 GitHub 的时候不想要显示名字"；"可以，改吧，仓库不用私有"。
- **我做的事**：
  - 提交作者改为 1karess + GitHub 隐藏邮箱；重写仓库历史（旧提交被覆盖）。
  - 代码里的服务器路径改为自动识别项目目录：作业脚本用提交目录，ccpp_common.py / make_stage0_docx.js 用文件所在位置。
  - 去掉作业脚本中的账户行（默认账户即可）；分区名移到不上传的 jobs/cluster.local.sh。
  - 不再上传：.claude/（本地 skill）、docs/stage0_plan_full.txt（含本机路径）、docs/*.xlsx。日志中的分区名已隐去。
- **代码**：jobs/*.sh（路径、账户行、uv 路径）、jobs/submit.sh、jobs/23_git_push.sh、scripts/ccpp_common.py、scripts/make_stage0_docx.js、scripts/fill_record.py、results/token_audit.json、.gitignore、README.md。
- **注意**：作业现在必须在项目目录下提交（日志写到 ./logs）。
