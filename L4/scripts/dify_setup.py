#!/usr/bin/env python3
"""Dify 自动化设置脚本：登录 -> 配置API -> 上传文件 -> 创建知识库 -> 创建文档 -> 创建智能体"""
import sys
import time
import logging
import mimetypes
import base64
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DIFY_URL = "http://localhost"
CHATANYWHERE_API_KEY = "sk-dov1cDqufMuws5jjSFrEDKmbUJYuiqCeEXao0nhlv1MyA2Je"
CHATANYWHERE_BASE_URL = "https://api.chatanywhere.tech"
ADMIN_EMAIL = "1757882878@qq.com"
ADMIN_PASSWORD = "IronAging2024!"

PDF_FILES = [
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\2026-刘黎啸-Vitamin C inhibits ACSL4 to alleviate ferro-aging in primates(1).pdf",
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\基于机器学习与实验验证的缺血性卒中核心机制解析及中药防治策略预测_廖昊森 (1).pdf",
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\基于迁移学习与图对比学习的药物-蛋白质相互作用预测方法_王煦.pdf",
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\基于深度学习的疾病及药物相关代谢物预测研究_刘文智.pdf",
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\基于图神经网络的化合物—蛋白质相互作用研究_万晓喆.pdf",
    r"D:\铁衰老 绝不重蹈覆辙\参考论文\基于异质图动态特征学习的药物重定位预测_朱昊坤.pdf",
]
MD_FILES = [
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\06_GENNDTI_MDL-HTI_abstracts.md",
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\01_DHGT-DTI.md",
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\02_GHCDTI.md",
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\03_H2GnnDTI.md",
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\04_MHGNN-DTI.md",
    r"D:\铁衰老 绝不重蹈覆辙\L4\papers\05_HTINet2.md",
]
ALL_FILES = PDF_FILES + MD_FILES


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def login():
    log.info("登录 Dify Console")
    s = requests.Session()
    r = s.post(f"{DIFY_URL}/console/api/login",
               json={"email": ADMIN_EMAIL, "password": b64(ADMIN_PASSWORD)}, timeout=10)
    if r.status_code != 200:
        log.error(f"登录失败: {r.status_code} {r.text[:200]}")
        return None, ""
    csrf = s.cookies.get("csrf_token", "")
    log.info("登录成功")
    return s, csrf


def api_post(s, csrf, path, **kw):
    h = {"X-CSRF-Token": csrf}
    if "json" in kw:
        h["Content-Type"] = "application/json"
    h.update(kw.pop("headers", {}))
    return s.post(f"{DIFY_URL}{path}", headers=h, **kw)


def api_get(s, csrf, path, **kw):
    h = {"X-CSRF-Token": csrf}
    h.update(kw.pop("headers", {}))
    return s.get(f"{DIFY_URL}{path}", headers=h, **kw)


def configure_chatanywhere(s, csrf):
    log.info("=== 配置 ChatAnywhere 免费 API ===")
    # 检查是否已配置
    r = api_get(s, csrf, "/console/api/workspaces/current/model-providers")
    if r.status_code == 200:
        for p in r.json().get("data", []):
            if p.get("provider") == "openai_api_compatible":
                # 检查是否已有凭证
                rc = api_get(s, csrf, "/console/api/workspaces/current/model-providers/openai_api_compatible/credentials")
                if rc.status_code == 200 and rc.json().get("credentials"):
                    log.info("openai_api_compatible 已配置，跳过")
                    return True
    # 创建凭证
    payload = {
        "credentials": {
            "api_key": CHATANYWHERE_API_KEY,
            "base_url": CHATANYWHERE_BASE_URL,
            "models": ["deepseek-v4", "gpt-4o", "text-embedding-3-small"],
        },
        "name": "ChatAnywhere Free API",
    }
    r = api_post(s, csrf, "/console/api/workspaces/current/model-providers/openai_api_compatible/credentials", json=payload)
    if r.status_code in (200, 201):
        log.info("ChatAnywhere 凭证配置成功")
    else:
        log.warning(f"凭证配置: {r.status_code} {r.text[:300]}")
    # 设置默认模型
    for mt, mn in [("llm", "deepseek-v4"), ("text-embedding", "text-embedding-3-small")]:
        r2 = api_post(s, csrf, "/console/api/workspaces/current/default-model",
                      json={"model": mn, "model_type": mt, "provider": "openai_api_compatible"})
        if r2.status_code in (200, 201):
            log.info(f"默认 {mt} 模型 = {mn}")
        else:
            log.warning(f"设置默认 {mt}: {r2.status_code}")
    return True


def upload_file(s, csrf, filepath):
    fpath = Path(filepath)
    if not fpath.exists():
        log.warning(f"文件不存在: {fpath.name}")
        return None
    # 跳过超过 15MB 的文件（Dify 上传限制）
    max_size = 15 * 1024 * 1024
    if fpath.stat().st_size > max_size:
        log.warning(f"文件过大 ({fpath.stat().st_size/1024/1024:.1f}MB, 限制{max_size/1024/1024:.0f}MB)，跳过: {fpath.name}")
        return None
    try:
        with open(fpath, "rb") as f:
            mime = mimetypes.guess_type(fpath.name)[0] or "application/octet-stream"
            r = api_post(s, csrf, "/console/api/files/upload",
                         data={"source": "datasets"},
                         files={"file": (fpath.name, f, mime)},
                         timeout=120)
        if r.status_code == 201:
            fid = r.json().get("id", "")
            log.info(f"  [OK] {fpath.name} -> {fid}")
            return fid
        log.warning(f"  [FAIL] {fpath.name}: {r.status_code}")
        return None
    except Exception as e:
        log.error(f"  [ERROR] {fpath.name}: {e}")
        return None


def create_knowledge_base(s, csrf):
    log.info("=== 创建/查找知识库 ===")
    # 先检查是否已存在同名知识库
    r = api_get(s, csrf, "/console/api/datasets?page=1&limit=50")
    if r.status_code == 200:
        datasets = r.json().get("data", [])
        for ds in datasets:
            if ds.get("name") == "铁衰老与脑缺血再灌注损伤文献库":
                did = ds.get("id", "")
                log.info(f"知识库已存在，ID: {did}")
                return did
    # 创建新知识库
    r = api_post(s, csrf, "/console/api/datasets",
                 json={"name": "铁衰老与脑缺血再灌注损伤文献库",
                       "description": "铁衰老、脑缺血再灌注损伤、GNN 方法学论文",
                       "provider": "vendor",
                       "indexing_technique": "economy"})
    if r.status_code in (200, 201):
        did = r.json().get("id") or r.json().get("data", {}).get("id", "")
        if did:
            log.info(f"知识库创建成功 ID: {did}")
            return did
    log.warning(f"创建知识库失败: {r.status_code} {r.text[:300]}")
    return None


def create_documents(s, csrf, dataset_id, file_ids):
    log.info(f"=== 创建 {len(file_ids)} 个文档 ===")
    success = 0
    for fid in file_ids:
        time.sleep(1)
        config = {
            "original_document_id": None,
            "duplicate": True,
            "indexing_technique": "economy",
            "data_source": {
                "info_list": {
                    "data_source_type": "upload_file",
                    "file_info_list": {"file_ids": [fid]},
                },
            },
            "process_rule": {"mode": "automatic"},
            "doc_form": "text_model",
            "doc_language": "Chinese",
        }
        r = api_post(s, csrf, f"/console/api/datasets/{dataset_id}/documents", json=config)
        if r.status_code in (200, 201):
            success += 1
            docs = r.json().get("documents", [])
            doc_id = docs[0].get("id", "") if docs else "?"
            log.info(f"  [OK] doc_id={doc_id}")
        else:
            log.warning(f"  [FAIL] file_id={fid}: {r.status_code} {r.text[:200]}")
    return success


def create_agent_app(s, csrf, dataset_id):
    log.info("=== 创建铁衰老研究智能体 ===")
    payload = {
        "name": "铁衰老研究助手",
        "description": "基于铁衰老、脑缺血再灌注损伤、图神经网络文献的智能研究助手",
        "mode": "agent-chat",
        "model_config": {
            "provider": "openai_api_compatible",
            "model": "deepseek-v4",
            "parameters": {
                "temperature": 0.3, "top_p": 0.9, "max_tokens": 4096,
                "presence_penalty": 0.1, "frequency_penalty": 0.1,
            },
        },
        "dataset_ids": [dataset_id],
        "prompt": (
            "你是一个铁衰老领域的专业研究助手。\n"
            "请根据知识库中的文献为用户提供准确、专业的回答。\n"
            "优先引用文献证据，不确定的内容诚实说明。"
        ),
        "tools": [],
    }
    r = api_post(s, csrf, "/console/api/apps", json=payload)
    if r.status_code in (200, 201):
        app_id = r.json().get("id") or r.json().get("data", {}).get("id", "")
        log.info(f"智能体应用创建成功 ID: {app_id}")
        return app_id
    log.warning(f"创建应用失败: {r.status_code} {r.text[:300]}")
    return None


def main():
    s, csrf = login()
    if not s:
        sys.exit(1)
    configure_chatanywhere(s, csrf)
    log.info(f"\n=== 上传 {len(ALL_FILES)} 个文件 ===")
    file_ids = []
    for fp in ALL_FILES:
        fid = upload_file(s, csrf, fp)
        if fid:
            file_ids.append(fid)
    log.info(f"文件上传: {len(file_ids)}/{len(ALL_FILES)}")
    if not file_ids:
        log.error("无文件上传成功")
        sys.exit(1)
    dataset_id = create_knowledge_base(s, csrf)
    if not dataset_id:
        sys.exit(1)
    doc_count = create_documents(s, csrf, dataset_id, file_ids)
    app_id = create_agent_app(s, csrf, dataset_id)
    log.info("=" * 60)
    log.info("Dify 自动化设置完成！")
    log.info(f"文档: {doc_count}/{len(file_ids)}")
    if app_id:
        log.info(f"智能体 ID: {app_id}")
    log.info(f"Dify: {DIFY_URL}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
