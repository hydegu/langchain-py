## 10天极速路线

统一约定（从 Day1 就用）：

- 文档目录：`./data_md/`（放 `.md` 文件）。
- 向量库目录：`./storage/chroma_db/`（可重复运行）。
- 你每次写代码都尽量放在 `src/` 下，脚本放 `scripts/` 下（可复用、可迭代）。

Day 1（环境最小闭环）

- 今日目标：把 Python 项目跑起来，并能稳定安装/冻结依赖。
- 必学点（≤5）：`venv`（虚拟环境；例子：每个项目一个 `.venv`，避免依赖冲突）、pip 安装、requirements.txt、环境变量、项目入口脚本。
- 代码示例（≤30行，可运行）：`scripts/hello.py`

```
pythonfrom pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    print("python:", sys.version.split()[0])
    print("project_root:", root)
    data_dir = root / "data_md"
    data_dir.mkdir(exist_ok=True)
    print("data_dir_ok:", data_dir.exists())

if __name__ == "__main__":
    main()
```

- 练习任务（2个，含验收标准）
- 任务1：创建并激活虚拟环境；验收：`python -c "import sys; print(sys.prefix)"` 输出路径包含你的项目目录。
- 任务2：生成 `requirements.txt`；验收：`pip freeze > requirements.txt` 后文件非空且可用于 `pip install -r requirements.txt`。
- Java 对照提示：不要把 `venv` 当成 Maven scope；它更像“每个项目一套独立解释器+site-packages”，切换靠激活脚本而不是 profile。

命令（Windows / Linux）

- 创建 venv：
  - Windows PowerShell：`py -m venv .venv`
  - Linux/WSL：`python3 -m venv .venv`
- 激活：
  - Windows PowerShell：`.\.venv\Scripts\Activate.ps1`
  - Linux/WSL：`source .venv/bin/activate`
- 安装：
  - Windows：`py -m pip install -U pip`
  - Linux：`python -m pip install -U pip`

Day 2（Python 模块化 + 基础类型提示）

- 今日目标：能把“工具函数/配置/主程序”拆成模块并正确 import。
- 必学点：`module`（模块；例子：`utils.py` 就是一个模块）、`package`（包；例子：有 `__init__.py` 的目录可作为包）、`type hint`（类型提示；例子：`def f(x: int) -> str:`）、`dataclass`（数据类；例子：把配置变成强结构对象）、路径与 `pathlib`。
- 代码示例：`src/app/config.py`

```
pythonfrom dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("MODEL", "gpt-4o-mini")
    top_k: int = int(os.getenv("TOP_K", "3"))

def load_settings() -> Settings:
    return Settings()

if __name__ == "__main__":
    s = load_settings()
    print(s)
```

- 练习任务
- 任务1：新建 `src/app/__init__.py` 并用 `python -m src.app.config` 运行；验收：能打印 Settings。
- 任务2：给你后续会用到的函数都补上输入/输出类型提示；验收：你能解释每个函数返回什么类型（不用上 mypy 也行）。
- Java 对照提示：Python import 更像“运行时加载”，循环依赖比 Java 更容易踩坑；先做“单向依赖”（config→utils→业务）最稳。

Day 3（文件/JSON + 异常与日志）

- 今日目标：把本地 Markdown 读进来，并用日志定位错误。
- 必学点：JSON（JSON：结构化数据格式；例子：把配置/结果写成 `result.json`）、异常 `try/except`（exception：异常；例子：捕获 `FileNotFoundError` 给出提示）、logging（logging：日志；例子：`logger.info("loaded %d docs", n)`）、编码 UTF‑8、简单单元测试思路。
- 代码示例：`scripts/read_md.py`

```
pythonfrom pathlib import Path
import logging, json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("read_md")

def main():
    data_dir = Path("data_md")
    files = sorted(data_dir.glob("*.md"))
    docs = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
            docs.append({"path": str(p), "chars": len(text)})
        except Exception as e:
            log.exception("read_failed: %s", p)
            raise
    Path("out").mkdir(exist_ok=True)
    Path("out/index_preview.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ok: %d files", len(docs))

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：在 `data_md/` 放 3 个 `.md`；验收：运行后 `out/index_preview.json` 里有 3 条记录。
- 任务2：故意放一个不可读文件（比如权限/编码问题）；验收：日志能打印出失败文件路径与堆栈。
- Java 对照提示：别把异常全吞掉（空 except）；Python 很容易“看似成功实际跳过”，要么记录日志要么直接 raise。

Day 4（最小 LLM 调用 + Prompt 模板）

- 今日目标：能用 LangChain 调一次模型，并用模板化提示词复用。
- 必学点：`prompt template`（提示词模板；例子：把“系统提示+用户输入”参数化）、环境变量注入 key、最小 Chat 调用、温度/模型名这些参数。
- 代码示例（缺 KEY 也能跑，会给提示）：`scripts/llm_ping.py`

```
pythonimport os
from langchain_openai import ChatOpenAI

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY（或你的 OpenAI 兼容 Key）")
        return
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0)
    resp = llm.invoke("只回答：OK")
    print(resp.content)

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：把 `MODEL`、`OPENAI_API_KEY` 写到 `.env` 并在启动时加载（Day5 会给固定做法）；验收：不手动 export 也能调用成功。
- 任务2：把用户输入变成命令行参数；验收：`python scripts/llm_ping.py "你好"` 能返回内容。
- Java 对照提示：不要把 Prompt 当成硬编码字符串拼接；把可变参数抽成模板，就像 Java 里抽成方法参数/DTO，后续迭代成本低。

补充说明：结构化输出

- 结构化输出（structured output：结构化输出；例子：让模型“只返回 JSON”，你就不需要靠正则去猜字段）能显著降低后处理复杂度，因为你可以直接按 schema 解析/校验，而不是在自由文本里“找字段”。

Day 5（结构化输出：JSON/Schema）

- 今日目标：让模型按你定义的字段返回结果（dict/对象），并在代码里稳定消费。
- 必学点：schema（schema：数据结构约束；例子：定义 `answer` 必须是字符串、`citations` 必须是列表）、`with_structured_output`（把结构约束绑定到模型；例子：直接得到可解析的结构化结果）、失败重试/兜底思路。
- 代码示例（示意：字段固定，便于后续接 RAG 引用）：`scripts/structured.py`

```
pythonimport os
from typing import TypedDict, List
from langchain_openai import ChatOpenAI

class Out(TypedDict):
    answer: str
    citations: List[str]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY")
        return
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0).with_structured_output(Out)
    out = llm.invoke("用一句话解释 RAG，并给 2 条假引用（写成字符串数组）。")
    print(out)

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：把 `citations` 改成“必须包含 2 个元素”；验收：当模型没给到时，你的代码能检测并报错/重试一次。
- 任务2：把输出写入 `out/answer.json`；验收：文件能被 `python -c "import json;print(json.load(open('out/answer.json','r',encoding='utf-8'))['answer'])"` 读到。
- Java 对照提示：TypedDict/Pydantic 的思路类似 Java DTO + 校验；别在业务里到处传 dict“野生字段”，后面很难维护。

Day 6（文档加载→切分）

- 今日目标：把 `data_md/` 的 Markdown 转成“可检索的小块”。
- 必学点：chunk（chunk：文本块；例子：把长文切成 800 字符一块）、overlap（overlap：重叠区；例子：相邻块重叠 100 字符减少断句损失）、Text Splitter（文本切分器；例子：RecursiveCharacterTextSplitter）。
- 代码示例：`scripts/split_md.py`

```
pythonfrom pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_md(dir_path="data_md"):
    docs=[]
    for p in sorted(Path(dir_path).glob("*.md")):
        docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                             metadata={"source": str(p)}))
    return docs

def main():
    docs = load_md()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print("docs:", len(docs), "chunks:", len(chunks))
    print("sample_meta:", chunks[0].metadata)

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：调 `chunk_size`/`chunk_overlap`；验收：你能观察到 chunks 数量变化，并解释原因。
- 任务2：给每个 chunk 增加 `chunk_id` 元数据；验收：打印任意 chunk 都包含 `source` + `chunk_id`。
- Java 对照提示：别用“按行切/按段切”的固定规则一条走到黑；切分策略会直接影响召回质量，这点更像搜索系统的分词/索引策略而不是普通字符串处理。

Day 7（Embedding→向量库入库：Chroma 持久化）

- 今日目标：把 chunks 变成向量并写入本地可持久化的 Chroma。
- 必学点：embedding（embedding：向量表示；例子：把一段文字编码成向量用于相似度检索）、vector store（向量库；例子：Chroma 保存向量并支持检索）、持久化目录 `persist_directory`（例子：下次启动不用重建）。
- 代码示例：`scripts/build_index.py`

```
pythonimport os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_chunks():
    docs=[]
    for p in sorted(Path("data_md").glob("*.md")):
        docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                             metadata={"source": str(p)}))
    return RecursiveCharacterTextSplitter(800,120).split_documents(docs)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY")
        return
    chunks = load_chunks()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="./storage/chroma_db",
        collection_name="md_rag",
    )
    print("indexed_chunks:", len(chunks))

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：重复运行两次，观察是否会重复入库；验收：你在代码里加一个“清库/重建”开关（例如删除目录）并解释取舍。
- 任务2：把 `collection_name` 改成可配置（环境变量）；验收：能在不同集合之间切换而不互相污染。
- Java 对照提示：向量库不是关系库事务模型；“重建索引”往往比“增量修修补补”更简单可靠，先跑通再谈优化。

Day 8（检索：top‑k 片段 + 可解释输出）

- 今日目标：给定问题，先打印检索到的 top‑k 文段（降低幻觉）。
- 必学点：retriever（retriever：检索器；例子：`as_retriever(k=3)`）、top‑k（top‑k：返回最相关的 k 条；例子：取 3 段作为上下文）、metadata 回显（source/chunk_id）。
- 代码示例：`scripts/retrieve_only.py`

```
pythonimport os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY")
        return
    vs = Chroma(
        collection_name="md_rag",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./storage/chroma_db",
    )
    q = input("Question> ").strip()
    docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(q)
    for i, d in enumerate(docs, 1):
        print(f"\n--- top{i} source={d.metadata.get('source')} ---")
        print(d.page_content[:300])

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：把 `k` 暴露成命令行参数；验收：`k=5` 时会多打印两段。
- 任务2：输出同时包含“source + 片段前 120 字符 + 片段长度”；验收：每条都完整打印三项。
- Java 对照提示：别急着“先问 LLM 再说”；RAG 的关键是“先检索再生成”，检索结果打印出来相当于你的调试日志/断点。

Day 9（生成：把检索片段拼进 Prompt）

- 今日目标：完成“检索→拼 Prompt→回答”，并要求答案带引用。
- 必学点：context stuffing（把检索片段注入上下文；例子：把 top‑k 段落拼成 `CONTEXT:`）、引用格式约定（例子：`[source:xxx]`）、温度设为 0 便于稳定回归测试。
- 代码示例：`scripts/rag_qa.py`

```
pythonimport os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY")
        return
    vs = Chroma(collection_name="md_rag", embedding_function=OpenAIEmbeddings(),
                persist_directory="./storage/chroma_db")
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "你是严谨助手。只基于 CONTEXT 回答；如果缺信息就说不知道。\n"
        "回答末尾给出引用 source 列表。\n\nCONTEXT:\n{context}\n\nQUESTION:\n{q}"
    )
    llm = ChatOpenAI(model=os.getenv("MODEL","gpt-4o-mini"), temperature=0)

    q = input("Question> ").strip()
    docs = retriever.invoke(q)
    context = "\n\n".join([f"[source:{d.metadata.get('source')}]\n{d.page_content}" for d in docs])
    resp = llm.invoke(prompt.format_messages(context=context, q=q))
    print(resp.content)

if __name__ == "__main__":
    main()
```

- 练习任务
- 任务1：当 `docs` 为空时直接提示“未检索到相关内容”；验收：不会把空 CONTEXT 送进模型。
- 任务2：把检索到的 top‑k 片段也一并输出到 `out/retrieved.txt`；验收：文件包含每段的 source 标记。
- Java 对照提示：把“拼 Prompt 的逻辑”封装成函数/类（像 service 层），不要散落在 CLI 里；否则你很快就会写出“无法复用的脚本泥球”。

Day 10（做成可复用 CLI：一条命令启动，一条命令提问）

- 今日目标：形成最小作品：`build_index` 一次，`ask` 多次。
- 必学点：CLI（CLI：命令行工具；例子：`python -m app.cli ask "..."`）、配置集中化、可重复运行、输出规范（含 top‑k 或引用）。
- 代码示例：`src/app/cli.py`

```
pythonimport argparse, os
from app.rag import build_index, ask

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("question")
    args = p.parse_args()

    if args.cmd == "build":
        build_index()
    else:
        print(ask(args.question))

if __name__ == "__main__":
    os.environ.setdefault("TOP_K", "3")
    main()
```

- 练习任务
- 任务1：实现 `app.rag` 里的 `build_index()/ask()`（模板在下个小节给）；验收：`python -m app.cli build` 成功且 `python -m app.cli ask "..."` 有输出。
- 任务2：加一个 `--show-context` 开关；验收：打开时先打印 top‑k，再打印最终答案。
- Java 对照提示：CLI 参数解析相当于 Spring Boot 的 `@ConfigurationProperties` + Controller 入参校验；先把“入参/配置/输出契约”定死，后面迭代很省心。

可选扩展阅读（不要求）：LangChain Quickstart。

## 最小工程模板

目录结构（可直接照抄）

```
textyour-rag/
  data_md/                    # 放 Markdown 数据
  storage/
    chroma_db/                # Chroma 持久化目录（可清空重建）[web:21]
  src/
    app/
      __init__.py
      config.py               # 读取环境变量/默认值
      rag.py                  # build_index / retrieve / ask
      cli.py                  # 统一入口
  tests/
    test_smoke.py             # 最小冒烟测试（先断言函数可导入）
  scripts/
    llm_ping.py
    build_index.py
    rag_qa.py
  config/
    prompt.txt                # 可选：把提示词外置
  .env                        # 不提交 git
  .gitignore
  requirements.txt
  README.md
```

`.env` 与密钥安全建议

- `.env`（.env：环境变量文件；例子：把 `OPENAI_API_KEY=...` 放进 `.env`）放本地即可，并把它加入 `.gitignore`，任何 key 都不要提交到 Git。
- 在 Windows 用 PowerShell、Linux 用 bash 时，推荐用同一个方式加载：安装 `python-dotenv`，在 `app/config.py` 启动时 `load_dotenv()`（这样你不用记两套 export/set 命令）。

依赖管理默认：`requirements.txt`

- 原因：对“最低门槛、最快出作品”最友好，`pip install -r requirements.txt` 即可复现环境；等你要做打包/发布再升级到 `pyproject.toml` 也不迟。
- 建议依赖（示例）：`langchain-openai`、`langchain-chroma`、`langchain-text-splitters`、`python-dotenv`、`pytest`（具体版本你先不锁死也行，跑通后再 freeze）。

## 作品验收标准

输入与启动

- 输入：`data_md/` 目录下的一组 Markdown 文件（你只要把资料丢进去即可）。
- 一条命令启动（建库）：`python -m app.cli build`（会进行“加载→切分→embedding→写入 Chroma 持久化目录”这一链路）。
- 一条命令提问：`python -m app.cli ask "你的问题"`（会进行“向量检索 top‑k→拼接 prompt→生成回答”这一链路）。

输出要求（降低幻觉）

- 回答必须附带引用：至少输出检索到的 top‑k 文段（含 `source`），或在最终回答末尾列出引用的 `source` 列表（二选一，但建议两者都做）。
- 可重复运行：清空 `storage/chroma_db/` 后重建索引，结果仍能正常问答；不清空时也能直接提问（持久化可用）。

## 接下来我需要你做的事

先确认你想用哪类“OpenAI 兼容”模型服务（比如某国内厂商的 OpenAI 兼容网关，或直接 OpenAI），并告诉我你更希望最终作品形态是 **CLI** 还是最小 **API 服务**（API service：接口服务；例子：`POST /ask` 返回 JSON）。
