"""系统化数据检索模块

提供 PubMed 学术文献检索和 GitHub 开源仓库查询的统一接口，支持：
- PubMed: 文献搜索、详情获取、引文格式化、MeSH 词查询
- GitHub: 仓库搜索、代码搜索、元数据获取
- 本地缓存：检索结果缓存，避免重复查询
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCacheConfig:
    """检索缓存配置。"""

    cache_dir: str = "L4/retrieval_cache"
    ttl_hours: float = 24.0
    max_cache_entries: int = 500
    enabled: bool = True


class RetrievalCache:
    """检索结果本地缓存，避免重复查询外部 API。"""

    def __init__(self, cache_dir: Path | str, config: RetrievalCacheConfig | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RetrievalCacheConfig()
        self._cached_pmids: set[str] = set()
        self._total_queries: int = 0
        self._cache_hits: int = 0
        self._api_calls: int = 0

    def _cache_key(self, source: str, query_hash: str) -> str:
        return f"{source}_{query_hash}"

    def _cache_path(self, source: str, query_hash: str) -> Path:
        key = self._cache_key(source, query_hash)
        return self.cache_dir / f"{key}.json"

    def get(self, source: str, query_str: str) -> dict | None:
        if not self.config.enabled:
            return None
        self._total_queries += 1
        query_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        cache_path = self._cache_path(source, query_hash)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01T00:00:00"))
            if datetime.now() - cached_time > timedelta(hours=self.config.ttl_hours):
                logger.debug(f"检索缓存已过期: {cache_path.name}")
                return None
            logger.info(f"  检索缓存命中: {source}/{query_hash}")
            self._cache_hits += 1
            return data.get("result")
        except Exception as e:
            logger.warning(f"检索缓存加载失败: {e}")
            return None

    def set(self, source: str, query_str: str, result: dict, is_api_call: bool = False) -> None:
        if not self.config.enabled:
            return
        if is_api_call:
            self._api_calls += 1

        # PMID 去重：检查结果中的 PMID 是否已缓存
        if source.startswith("pubmed"):
            pmids_in_result = self._extract_pmids(result)
            new_pmids = pmids_in_result - self._cached_pmids
            if not new_pmids and pmids_in_result:
                logger.debug(f"检索结果 PMID 已全部缓存，跳过写入: {source}")
                return
            self._cached_pmids.update(pmids_in_result)

        query_hash = hashlib.sha256(query_str.encode()).hexdigest()[:16]
        cache_path = self._cache_path(source, query_hash)
        try:
            cache_path.write_text(
                json.dumps({
                    "result": result,
                    "cached_at": datetime.now().isoformat(),
                    "source": source,
                    "query_hash": query_hash,
                }, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"检索缓存保存失败: {e}")
            return

        if self.config.max_cache_entries > 0:
            self._enforce_entry_limit()

    @staticmethod
    def _extract_pmids(result: dict, depth: int = 0) -> set[str]:
        """从检索结果中提取 PMID 集合。
        
        Args:
            result: 检索结果字典。
            depth: 当前递归深度，最大 10 层以防止循环引用。
        """
        if depth >= 10:
            logger.warning("_extract_pmids 递归深度达到上限 (10)，停止递归")
            return set()
        pmids: set[str] = set()
        if "pmids" in result:
            pmids.update(str(p) for p in result["pmids"] if p)
        if "articles" in result:
            for article in result["articles"]:
                if isinstance(article, dict) and "pmid" in article:
                    pmids.add(str(article["pmid"]))
        if "result" in result and isinstance(result["result"], dict):
            return RetrievalCache._extract_pmids(result["result"], depth + 1)
        return pmids

    def _enforce_entry_limit(self) -> int:
        if self.config.max_cache_entries <= 0:
            return 0
        files = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
        )
        removed = 0
        while len(files) > self.config.max_cache_entries:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"缓存条目限制清理失败 {oldest}: {e}")
        if removed > 0:
            logger.info(f"检索缓存条目限制: 删除 {removed} 个文件 (max={self.config.max_cache_entries})")
        return removed

    def clear_expired(self) -> int:
        removed = 0
        cutoff = datetime.now() - timedelta(hours=self.config.ttl_hours)
        for f in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01T00:00:00"))
                if cached_time < cutoff:
                    f.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"清理过期检索缓存失败 {f}: {e}")
        return removed

    def get_stats(self) -> dict:
        """返回检索缓存统计信息。"""
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "api_calls": self._api_calls,
            "cached_pmids_count": len(self._cached_pmids),
            "hit_rate": self._cache_hits / self._total_queries if self._total_queries > 0 else 0.0,
        }


@dataclass
class PubMedQuery:
    """PubMed 查询构建器，生成标准 NCBI 查询语法。"""

    query: str
    max_results: int = 20
    sort: str = "relevance"
    date_range: dict | None = None
    publication_types: list[str] = field(default_factory=list)
    has_abstract: bool = True
    free_full_text: bool = False
    language: str = "english"
    species: str = "humans"

    def to_args(self) -> dict:
        args: dict[str, Any] = {
            "query": self.query,
            "maxResults": self.max_results,
            "sort": self.sort,
            "hasAbstract": self.has_abstract,
            "language": self.language,
        }
        if self.free_full_text:
            args["freeFullText"] = True
        if self.species:
            args["species"] = self.species
        if self.publication_types:
            args["publicationTypes"] = self.publication_types
        if self.date_range:
            args["dateRange"] = self.date_range
        return args


@dataclass
class GitHubQuery:
    """GitHub 查询构建器。"""

    query: str
    page: int = 1
    per_page: int = 30
    sort: str = "stars"
    order: str = "desc"
    language: str = "python"

    def to_args(self) -> dict:
        args: dict[str, Any] = {
            "query": self.query,
            "page": self.page,
            "perPage": self.per_page,
            "sort": self.sort,
            "order": self.order,
        }
        if self.language:
            args["query"] = f"{self.query} language:{self.language}"
        return args


class QueryBuilder:
    """领域专用查询构建器，封装铁衰老研究相关的常见搜索模式。"""

    @staticmethod
    def iron_aging_mechanisms() -> PubMedQuery:
        return PubMedQuery(
            query='("iron metabolism"[MeSH] OR "iron overload"[MeSH]) AND ("aging"[MeSH] OR "cellular senescence"[MeSH])',
            max_results=30,
            sort="pub_date",
        )

    @staticmethod
    def natural_compounds_iron_aging() -> PubMedQuery:
        return PubMedQuery(
            query='("Biological Products"[MeSH] OR "Phytochemicals"[MeSH] OR "Plant Extracts"[MeSH]) AND ("iron"[MeSH] OR "iron metabolism"[MeSH]) AND ("aging"[MeSH] OR "senescence"[MeSH])',
            max_results=30,
            sort="pub_date",
        )

    @staticmethod
    def gnn_dti_prediction() -> PubMedQuery:
        return PubMedQuery(
            query='("graph neural network" OR "GNN" OR "graph convolutional") AND ("drug-target interaction" OR "DTI" OR "compound-protein interaction")',
            max_results=30,
            sort="pub_date",
        )

    @staticmethod
    def heterogeneous_gnn_biology() -> PubMedQuery:
        return PubMedQuery(
            query='("heterogeneous graph" OR "knowledge graph") AND ("drug discovery" OR "target prediction" OR "biological network")',
            max_results=30,
            sort="pub_date",
        )

    @staticmethod
    def ferroptosis_iron_aging() -> PubMedQuery:
        return PubMedQuery(
            query='("ferroptosis"[MeSH] OR "ferroptosis") AND ("aging"[MeSH] OR "cellular senescence"[MeSH] OR "iron"[MeSH])',
            max_results=30,
            sort="pub_date",
        )

    @staticmethod
    def gnn_dti_github() -> GitHubQuery:
        return GitHubQuery(
            query="drug target interaction GNN prediction",
            sort="stars",
            per_page=20,
        )

    @staticmethod
    def heterogeneous_graph_github() -> GitHubQuery:
        return GitHubQuery(
            query="heterogeneous graph neural network pytorch geometric",
            sort="stars",
            per_page=20,
        )

    @staticmethod
    def contrastive_learning_dti_github() -> GitHubQuery:
        return GitHubQuery(
            query="contrastive learning drug target prediction",
            sort="stars",
            per_page=20,
        )


class PubMedRetriever:
    """PubMed 学术文献检索器。

    通过 MCP pubmed 工具或 NCBI Entrez API 直接查询 PubMed 数据库。
    支持搜索、获取详情、引文格式化、MeSH 词查询。
    """

    QUERY_BUILDER = QueryBuilder()

    def __init__(self, cache: RetrievalCache | None = None, use_mcp: bool = True):
        self.cache = cache
        if use_mcp:
            try:
                from ..mcp import call_mcp_tool  # noqa: F401
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(
                    f"PubMedRetriever: MCP 模块不可用 ({e})，自动设置 use_mcp=False，"
                    f"将使用 Entrez API 作为后备"
                )
                use_mcp = False
        self.use_mcp = use_mcp

    def search(self, query: PubMedQuery | None = None, **kwargs) -> dict:
        """搜索 PubMed 文献。

        Args:
            query: PubMedQuery 对象，或通过 kwargs 覆盖字段。
            **kwargs: 覆盖 query 的字段，如 query="cancer", max_results=10。

        Returns:
            包含 pmids, results 的字典。
        """
        if query is None:
            query = PubMedQuery(**kwargs) if kwargs else PubMedQuery(query="iron aging", max_results=10)
        elif kwargs:
            for k, v in kwargs.items():
                setattr(query, k, v)

        cache_key = json.dumps(query.to_args(), sort_keys=True)
        if self.cache:
            cached = self.cache.get("pubmed_search", cache_key)
            if cached is not None:
                return cached

        logger.info(f"PubMed 搜索: {query.query[:80]}...")
        result = self._search_mcp(query) if self.use_mcp else self._search_entrez(query)

        # 限制检索结果数量不超过 max_results
        if "pmids" in result and isinstance(result["pmids"], list):
            result["pmids"] = result["pmids"][:query.max_results]
        if "articles" in result and isinstance(result["articles"], list):
            result["articles"] = result["articles"][:query.max_results]

        if self.cache:
            self.cache.set("pubmed_search", cache_key, result, is_api_call=True)
        return result

    def _search_mcp(self, query: PubMedQuery) -> dict:
        """通过 MCP 工具搜索 PubMed。"""
        try:
            from ..mcp import call_mcp_tool
            return call_mcp_tool("mcp_pubmed", "pubmed_search_articles", query.to_args())
        except Exception as e:
            logger.warning(f"MCP PubMed 搜索失败 ({e})，降级为 Entrez API")
            return self._search_entrez(query)

    def _search_entrez(self, query: PubMedQuery) -> dict:
        """通过 NCBI Entrez API 直接搜索 PubMed。"""
        try:
            from Bio import Entrez
            Entrez.email = "research@iron-aging.org"
            Entrez.tool = "iron-aging-gnn"

            handle = Entrez.esearch(
                db="pubmed", term=query.query, retmax=query.max_results,
                sort=query.sort,
            )
            record = Entrez.read(handle)
            handle.close()
            return {"pmids": list(record["IdList"]), "count": int(record["Count"])}
        except Exception as e:
            logger.error(f"Entrez API 搜索失败: {e}")
            return {"pmids": [], "count": 0, "error": str(e)}

    def fetch_articles(self, pmids: list[str]) -> dict:
        """获取指定 PMID 的文章详情。"""
        cache_key = ",".join(sorted(pmids))
        if self.cache:
            cached = self.cache.get("pubmed_fetch", cache_key)
            if cached is not None:
                return cached

        logger.info(f"PubMed 获取文章: {len(pmids)} 篇")
        result = self._fetch_mcp(pmids) if self.use_mcp else self._fetch_entrez(pmids)

        if self.cache:
            self.cache.set("pubmed_fetch", cache_key, result)
        return result

    def _fetch_mcp(self, pmids: list[str]) -> dict:
        try:
            from ..mcp import call_mcp_tool
            return call_mcp_tool("mcp_pubmed", "pubmed_fetch_articles", {"pmids": pmids})
        except Exception as e:
            logger.warning(f"MCP PubMed 获取失败 ({e})，降级为 Entrez API")
            return self._fetch_entrez(pmids)

    def _fetch_entrez(self, pmids: list[str]) -> dict:
        try:
            from Bio import Entrez
            Entrez.email = "research@iron-aging.org"
            Entrez.tool = "iron-aging-gnn"

            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            articles = []
            for article in records.get("PubmedArticle", []):
                medline = article.get("MedlineCitation", {})
                article_info = medline.get("Article", {})
                articles.append({
                    "pmid": str(medline.get("PMID", "")),
                    "title": str(article_info.get("ArticleTitle", "")),
                    "abstract": str(article_info.get("Abstract", {}).get("AbstractText", [""])[0]) if article_info.get("Abstract") else "",
                    "journal": str(article_info.get("Journal", {}).get("Title", "")),
                    "year": str(article_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "")),
                })
            return {"articles": articles}
        except Exception as e:
            logger.error(f"Entrez API 获取失败: {e}")
            return {"articles": [], "error": str(e)}


class GitHubRetriever:
    """GitHub 开源仓库检索器。

    通过 MCP GitHub 工具或 GitHub REST API 查询开源项目。
    支持仓库搜索、代码搜索、元数据获取。
    """

    QUERY_BUILDER = QueryBuilder()

    def __init__(self, cache: RetrievalCache | None = None, use_mcp: bool = True):
        self.cache = cache
        if use_mcp:
            try:
                from ..mcp import call_mcp_tool  # noqa: F401
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(
                    f"GitHubRetriever: MCP 模块不可用 ({e})，自动设置 use_mcp=False，"
                    f"将使用 REST API 作为后备"
                )
                use_mcp = False
        self.use_mcp = use_mcp

    def search_repositories(self, query: GitHubQuery | None = None, **kwargs) -> dict:
        """搜索 GitHub 仓库。"""
        if query is None:
            query = GitHubQuery(query="graph neural network drug target", per_page=20)
        elif kwargs:
            for k, v in kwargs.items():
                setattr(query, k, v)

        cache_key = json.dumps(query.to_args(), sort_keys=True)
        if self.cache:
            cached = self.cache.get("github_search", cache_key)
            if cached is not None:
                return cached

        logger.info(f"GitHub 搜索仓库: {query.to_args()['query'][:80]}...")
        result = self._search_mcp(query) if self.use_mcp else self._search_rest(query)

        # 限制检索结果数量不超过 per_page
        if "items" in result and isinstance(result["items"], list):
            result["items"] = result["items"][:query.per_page]

        if self.cache:
            self.cache.set("github_search", cache_key, result, is_api_call=True)
        return result

    def _search_mcp(self, query: GitHubQuery) -> dict:
        try:
            from ..mcp import call_mcp_tool
            return call_mcp_tool("mcp_GitHub", "search_repositories", query.to_args())
        except Exception as e:
            logger.warning(f"MCP GitHub 搜索失败 ({e})，降级为 REST API")
            return self._search_rest(query)

    def _search_rest(self, query: GitHubQuery) -> dict:
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query.to_args()['query'])}&sort={query.sort}&order={query.order}&per_page={query.per_page}&page={query.page}"
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "iron-aging-gnn"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return {"items": data.get("items", []), "total_count": data.get("total_count", 0)}
        except Exception as e:
            logger.error(f"GitHub REST API 搜索失败: {e}")
            return {"items": [], "total_count": 0, "error": str(e)}

    def search_code(self, query: str, per_page: int = 20) -> dict:
        """搜索 GitHub 代码。"""
        cache_key = f"code:{query}:{per_page}"
        if self.cache:
            cached = self.cache.get("github_code", cache_key)
            if cached is not None:
                return cached

        logger.info(f"GitHub 搜索代码: {query[:80]}...")
        result = self._search_code_mcp(query, per_page) if self.use_mcp else self._search_code_rest(query, per_page)

        # 限制检索结果数量不超过 per_page
        if "items" in result and isinstance(result["items"], list):
            result["items"] = result["items"][:per_page]

        if self.cache:
            self.cache.set("github_code", cache_key, result, is_api_call=True)
        return result

    def _search_code_mcp(self, query: str, per_page: int) -> dict:
        try:
            from ..mcp import call_mcp_tool
            return call_mcp_tool("mcp_GitHub", "search_code", {"q": query, "per_page": per_page})
        except Exception as e:
            logger.warning(f"MCP GitHub 代码搜索失败 ({e})，降级为 REST API")
            return self._search_code_rest(query, per_page)

    def _search_code_rest(self, query: str, per_page: int) -> dict:
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.github.com/search/code?q={urllib.parse.quote(query)}&per_page={per_page}"
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "iron-aging-gnn"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return {"items": data.get("items", []), "total_count": data.get("total_count", 0)}
        except Exception as e:
            logger.error(f"GitHub REST API 代码搜索失败: {e}")
            return {"items": [], "total_count": 0, "error": str(e)}


class DataRetriever:
    """统一的数据检索入口，组合 PubMed 和 GitHub 检索器。"""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        cache_config: RetrievalCacheConfig | None = None,
        use_mcp: bool = True,
    ):
        cache = None
        if cache_dir is not None:
            cache = RetrievalCache(cache_dir, cache_config)
        self.cache = cache
        self.pubmed = PubMedRetriever(cache=cache, use_mcp=use_mcp)
        self.github = GitHubRetriever(cache=cache, use_mcp=use_mcp)

    def search_iron_aging_literature(self, max_results: int = 20) -> dict:
        query = QueryBuilder.iron_aging_mechanisms()
        query.max_results = max_results
        return self.pubmed.search(query)

    def search_natural_compounds(self, max_results: int = 20) -> dict:
        query = QueryBuilder.natural_compounds_iron_aging()
        query.max_results = max_results
        return self.pubmed.search(query)

    def search_gnn_methods(self, max_results: int = 20) -> dict:
        query = QueryBuilder.gnn_dti_prediction()
        query.max_results = max_results
        return self.pubmed.search(query)

    def search_github_gnn_dti(self, per_page: int = 20) -> dict:
        query = QueryBuilder.gnn_dti_github()
        query.per_page = per_page
        return self.github.search_repositories(query)

    def clear_expired_cache(self) -> int:
        """清理过期的检索缓存。"""
        if self.cache:
            return self.cache.clear_expired()
        return 0

    def clear_all_cache(self) -> int:
        """清空所有检索缓存文件。"""
        if self.cache:
            removed = 0
            for f in self.cache.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"删除检索缓存文件失败 {f}: {e}")
            logger.info(f"已清空全部检索缓存: {removed} 个文件")
            return removed
        return 0

    def batch_prefetch(self, queries: list[dict] | None = None) -> dict:
        """批量检索并缓存结果，使用预定义查询列表。

        Args:
            queries: 查询列表，每项为 {"name": str, "source": "pubmed"|"github",
                     "params": dict}。默认使用预定义查询。

        Returns:
            {"results": {name: result}, "errors": {name: error}, "stats": dict}
        """
        if queries is None:
            queries = [
                {"name": "iron_aging_mechanisms", "source": "pubmed",
                 "params": {"query": QueryBuilder.iron_aging_mechanisms().query, "max_results": 30}},
                {"name": "natural_compounds", "source": "pubmed",
                 "params": {"query": QueryBuilder.natural_compounds_iron_aging().query, "max_results": 30}},
                {"name": "gnn_dti", "source": "pubmed",
                 "params": {"query": QueryBuilder.gnn_dti_prediction().query, "max_results": 30}},
                {"name": "ferroptosis", "source": "pubmed",
                 "params": {"query": QueryBuilder.ferroptosis_iron_aging().query, "max_results": 30}},
                {"name": "github_gnn_dti", "source": "github",
                 "params": {"query": "drug target interaction GNN", "per_page": 20}},
            ]

        results: dict[str, dict] = {}
        errors: dict[str, str] = {}

        for q in queries:
            name = q.get("name", f"query_{len(results)}")
            source = q.get("source", "pubmed")
            params = q.get("params", {})

            try:
                if source == "pubmed":
                    query = PubMedQuery(**params)
                    results[name] = self.pubmed.search(query)
                elif source == "github":
                    query = GitHubQuery(**params)
                    results[name] = self.github.search_repositories(query)
                else:
                    raise ValueError(f"不支持的检索源: {source!r}")
            except Exception as e:
                logger.error(f"批量预取查询 {name} 失败: {e}")
                errors[name] = str(e)

        logger.info(
            f"批量预取完成: {len(results)}/{len(queries)} 查询成功, "
            f"{len(errors)} 失败"
        )
        return {
            "results": results,
            "errors": errors,
            "stats": self.get_stats() if self.cache else {},
        }

    def get_stats(self) -> dict:
        """返回检索统计信息，包括缓存命中率、总查询次数、API 调用次数等。"""
        if self.cache:
            return self.cache.get_stats()
        return {
            "total_queries": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "cached_pmids_count": 0,
            "hit_rate": 0.0,
        }