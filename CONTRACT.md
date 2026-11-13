# 封存导出的身份与汇总核定

## 已有材料与事实
data/manifest.json 按 A、B、C、D 指定四批导出的顺序、行数及材料 SHA-256。registry.csv 是事故前完整登记，id 按字符串保存；不同前导零就是不同实体，region 在本次期间不变。routes.json 是原系统保留的分片白名单，不是每个分片必然出现的实体清单。四批是本次核账的完整输入，开始前账本为空；未出现在任何一批的登记实体不入账。原始带字符串 ID 的源行已经丢失，没有其他身份线索。

每份 CSV 的列为 rid,token,shard,rev,deleted,amount。rid 是批内导出行号字符串；全局行位置写为 A/01。token 是 exporter.py 产生的十进制数字字符串；合法原始身份须在登记表和该批该 shard 白名单内，且 str(int(id)) 等于 token。一批中每个原始 ID 至多出现一次，未导出的登记 ID 允许缺席。白名单可重叠，不能逐行独立选择。rev 为正整数，deleted 是 0 或 1，amount 是整数分，允许负数和零。

四批顺序不代表修订顺序。确定整批身份解释后，对每个原始 ID 在四批所有行中取最高 rev；同一 ID 的相同 rev 必须有相同 deleted 和 amount，否则该解释非法。墓碑也参与胜出；胜者为墓碑则该实体不计金额和笔数，不能改取旧活值。活的零金额仍计一笔。金额按登记 region 汇总，空分区金额及笔数均为零。现有 importer.py 按数字 token 合并，是历史行为，不能用它作为恢复正确性的判定器。

import.log 是原导入运行日志，warehouse.json 是该次导入后的完整账本与分区汇总。它们描述有损系统实际做了什么，不额外限定正确原始身份。source.load(directory) 校验哈希并读取全部材料；importer.replay(pack) 用真实 SQLite 复现封存输出。audit.evaluate(pack, assignment) 仅核对给定完整身份见证并计算胜出账本与汇总，不选择身份，也不求边界。保留这些源码、封存材料及 checks.py 既有断言。

## 对外分析入口
python3 analysis.py run DATA OUT 从材料目录重算 OUT/result.json 及 OUT/REPORT.md，OUT 不得位于 DATA 内。重复运行结果的业务内容一致，见证编号或同值见证可不同。只需处理本批与同格式的小型验证材料，不开发通用解析引擎。
python3 analysis.py verify DATA OUT 重算分析并核对已有结果、正向见证及报告数值，同时执行小材料穷举对照；不能只检查文件存在或只调用 audit.check_result。
python3 analysis.py demo DATA OUT 展示重算后的真实分区边界、历史金额及至少一个身份仍有歧义的行。
完成后 .pointwise-runtime.json 的 verify、demo 分别指向上述 verify、demo，DATA 用 data，OUT 用 out；setup 保留依赖安装。正常回归仍须通过。

## result.json
成功时顶层为：
- status：字符串 ok。
- materials：逐文件 SHA-256 字典，等于 source.load 返回的 materials，含 manifest.json 自身。
- witnesses：见证编号到完整 assignment 的字典；assignment 将所有全局行位置映射到原始 ID 字符串。见证可共享，不能只写单行或单分区的局部映射。
- row_support：全局行位置到 {原始ID:见证编号} 的字典，精确列出该行在全部合法解释中的可行身份。仅一项即唯一可恢复，多项即仍有歧义；不能把白名单候选直接当成可行身份。
- regions：分区名到 {observed,min,max,min_witness,max_witness} 的字典；observed 是封存 warehouse.json 的金额。min、max 是全部合法联合解释下的整数金额界，各端点引用一个完整见证。不同分区可用不同见证，不宣称其端点能同时达到。
audit.check_result(pack,result) 检查已给见证是否合法、端点是否可达及封存金额是否一致；它不证明支持集完整或金额界最优。分析自身须补齐排除和最优性的可重跑核验，不要求形式化理论证明或自研优化器。

材料哈希或字段不合法、或不存在满足所有身份和修订规则的解释时停止核定：result.json 只含 status=inconsistent、reason 字符串和 sources 材料位置列表；REPORT.md 解释冲突，不保留旧成功结果或输出部分边界。位置使用文件名与 CSV 行号（含表头）或 JSON 路径。

## 小例与报告
examples/tiny.json 是已解析的两批小材料，字段与 source.load 的业务部分相同，可交给独立穷举判定；它不含封存快照与哈希。A 的两行须分别属于 7 与 07，B 的墓碑可撤回其中任一 ID，共四份合法身份解释。每行支持均为 7、07；north、south 的金额界各为 0 至 20，但不能同时取 20。这只是小例预期，不是事故包答案。verify 还须用小型矛盾材料确认停止核定。

REPORT.md 说明哪些归属可恢复、哪些不可唯一恢复，列出各分区历史金额及恢复上下界、差额范围和端点见证编号，引用原始材料位置。解释至少一处身份选择如何改变修订胜者，以及墓碑和旧修订为何不能先行丢弃；结论须能由 result.json 及入口重算。排除和边界核验可使用锁定的 SciPy 整数优化并检查完整整数见证，不以浮点近似金额或抽样代替精确支持集。本轮不修生产导入器，不迁移数据库，不构建服务或展示页面。

## 基线运行
python3 checks.py verify 检查正常导入、封存重放与求解依赖；python3 checks.py demo 展示历史真实结果。requirements.txt 固定 NumPy 与 SciPy 版本。基线没有恢复搜索、恢复结果或事故分析报告。
