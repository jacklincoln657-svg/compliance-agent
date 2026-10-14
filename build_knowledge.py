import os
import shutil
import sys

# Windows 控制台默认 GBK，避免 print 中文/emoji 时报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 国内访问 HuggingFace 不稳定时走镜像（可被环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from parent_store import (
    CHROMA_DIR,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    make_source_prefix,
    save_parent_store,
    source_title_from_doc,
)

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_docs")


def build_vector_store():
    """
    ParentDocument 最小实现：
    - 父块 (~1000字) 存 parent_store.json
    - 子块 (~150字) 向量化进 Chroma，metadata 带 parent_id
    - 每个子块正文前强制注入 [来源]/[分类]，避免「标题只在文首」导致后文漏召回
    """
    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"找不到目录 {KNOWLEDGE_DIR}，请先创建。")
        return

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    subfolders = [
        f for f in os.listdir(KNOWLEDGE_DIR)
        if os.path.isdir(os.path.join(KNOWLEDGE_DIR, f))
    ]
    if not subfolders:
        print("未发现分类子文件夹，建库终止。")
        return

    parent_store: dict = {}
    child_docs: list[Document] = []
    parent_count = 0
    child_count = 0

    for category in subfolders:
        category_path = os.path.join(KNOWLEDGE_DIR, category)
        print(f"扫描领域目录: [{category}] ...")

        loader = DirectoryLoader(
            category_path,
            glob="**/*.*",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        raw_docs = loader.load()
        cat_parents = 0
        cat_children = 0

        for raw in raw_docs:
            source_path = raw.metadata.get("source", "")
            source_title = source_title_from_doc(raw.page_content, source_path)
            prefix = make_source_prefix(source_title, category)

            parents = parent_splitter.split_documents([raw])
            for p_idx, parent in enumerate(parents):
                parent_id = f"{category}::{source_title}::{p_idx}"
                parent_body = parent.page_content.strip()
                parent_store[parent_id] = {
                    "content": f"{prefix}\n{parent_body}",
                    "category": category,
                    "source": source_title,
                    "parent_index": p_idx,
                }
                cat_parents += 1
                parent_count += 1

                children = child_splitter.split_text(parent_body)
                print(
                    f"   · 文档[{source_title}] 父块#{p_idx} "
                    f"({len(parent_body)}字) → {len(children)} 个子块"
                )
                for c_idx, child_text in enumerate(children):
                    # 子块带上来源前缀：后文即使没写规范全名，也能被「规范名」类查询召回
                    child_docs.append(
                        Document(
                            page_content=f"{prefix}\n{child_text}",
                            metadata={
                                "category": category,
                                "source": source_title,
                                "parent_id": parent_id,
                                "child_index": c_idx,
                            },
                        )
                    )
                    cat_children += 1
                    child_count += 1

        print(f"   └── [{category}] 父块 {cat_parents} 个 / 子块 {cat_children} 个")

    if not child_docs:
        print("未扫描到文本。")
        return

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    save_parent_store(parent_store)
    Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(
        f"\n知识库构建完成！"
        f"父块 {parent_count} 个 → parent_store.json；"
        f"子块 {child_count} 个 → {CHROMA_DIR}"
    )


if __name__ == "__main__":
    build_vector_store()
