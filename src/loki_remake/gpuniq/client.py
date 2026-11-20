"""Typed synchronous HTTP client for the GPUniq API."""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GPUniqAPIError(RuntimeError):
    """Error raised when GPUniq returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id


class GPUniqClient:
    """Thin typed client around GPUniq Compute, Storage, Image Jobs, and OpenAI-compat LLM APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.gpuniq.com",
        timeout: float = 60.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": api_key,
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GPUniqClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry_on_error: bool = True,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        max_attempts = self.max_retries if retry_on_error else 1
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._client.request(
                    method,
                    path,
                    json=json,
                    params=params,
                    timeout=timeout,
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    delay = 1.5 * attempt
                    logger.warning(
                        "Transport error on {} {} ({}), retrying in {}s", method, path, exc, delay
                    )
                    time.sleep(delay)
                    continue
                raise GPUniqAPIError(f"Transport error on {method} {path}: {exc}") from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_attempts:
                retry_after = resp.headers.get("Retry-After")
                is_digits = bool(retry_after and retry_after.isdigit())
                delay = float(retry_after) if is_digits else 1.5 * attempt
                logger.warning(
                    "HTTP {} on {} {}, retrying in {}s", resp.status_code, method, path, delay
                )
                time.sleep(delay)
                continue

            return self._parse_response(resp, method, path)

        raise GPUniqAPIError(f"Request {method} {path} failed after retries: {last_exc}")

    @staticmethod
    def _parse_response(resp: httpx.Response, method: str, path: str) -> dict[str, Any]:
        try:
            payload = resp.json()
        except Exception as exc:
            if resp.is_error:
                raise GPUniqAPIError(
                    f"HTTP {resp.status_code} on {method} {path}: {resp.text[:300]}",
                    status_code=resp.status_code,
                ) from exc
            raise GPUniqAPIError(
                f"Invalid JSON response on {method} {path}",
                status_code=resp.status_code,
            ) from exc

        if isinstance(payload, dict) and "error" in payload and isinstance(payload["error"], dict):
            err = payload["error"]
            raise GPUniqAPIError(
                err.get("message") or f"API error on {method} {path}",
                status_code=payload.get("status_code") or resp.status_code,
                error_code=err.get("code"),
                request_id=payload.get("request_id"),
            )

        if isinstance(payload, dict) and payload.get("exception", 0) != 0:
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            msg = data.get("error") or payload.get("message") or f"GPUniq error on {method} {path}"
            raise GPUniqAPIError(
                str(msg),
                status_code=resp.status_code,
                error_code=data.get("error_code"),
                request_id=meta.get("request_id"),
            )

        if resp.is_error:
            msg = payload.get("error") if isinstance(payload, dict) else resp.text[:300]
            raise GPUniqAPIError(
                f"HTTP {resp.status_code} on {method} {path}: {msg}",
                status_code=resp.status_code,
                request_id=payload.get("request_id") if isinstance(payload, dict) else None,
            )

        return payload

    # ---- Account & Compute ----

    def get_balances(self) -> dict[str, Any]:
        return self._request("GET", "/v1/workspaces/balances")["data"]

    def list_cloud_gpus(
        self,
        *,
        min_vram_gb: int | None = None,
        min_ram_gb: int | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if min_vram_gb is not None:
            params["min_vram_gb"] = min_vram_gb
        if min_ram_gb is not None:
            params["min_ram_gb"] = min_ram_gb
        if search:
            params["search"] = search
        return self._request("GET", "/v1/gpu-cloud/instances", params=params)["data"]

    def list_instances(self, *, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v1/instances/my",
            params={"page": page, "page_size": page_size},
        )["data"]

    def get_instance(self, task_id: int) -> dict[str, Any]:
        return self._request("GET", f"/v1/instances/{task_id}")["data"]

    def get_instance_logs(self, task_id: int) -> dict[str, Any]:
        return self._request("GET", f"/v1/instances/{task_id}/logs")["data"]

    def list_marketplace(
        self,
        *,
        gpu_model: list[str] | None = None,
        min_vram_gb: float | None = None,
        min_ram_gb: float | None = None,
        min_disk_gb: float | None = None,
        min_inet_speed_mbps: float | None = None,
        max_price_per_hour: float | None = None,
        max_gpu_count: int | None = None,
        provider: str | None = None,
        sort_by: str = "price-low",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sort_by": sort_by,
            "page": page,
            "page_size": page_size,
        }
        if gpu_model:
            params["gpu_model"] = gpu_model
        if min_vram_gb is not None:
            params["min_vram_gb"] = min_vram_gb
        if min_ram_gb is not None:
            params["min_ram_gb"] = min_ram_gb
        if min_disk_gb is not None:
            params["min_disk_gb"] = min_disk_gb
        if min_inet_speed_mbps is not None:
            params["min_inet_speed_mbps"] = min_inet_speed_mbps
        if max_price_per_hour is not None:
            params["max_price_per_hour"] = max_price_per_hour
        if max_gpu_count is not None:
            params["max_gpu_count"] = max_gpu_count
        if provider:
            params["provider"] = provider
        return self._request("GET", "/v1/marketplace/list", params=params)["data"]

    def get_agent(self, agent_id: int) -> dict[str, Any]:
        return self._request("GET", f"/v1/marketplace/agent/{agent_id}")["data"]

    def create_marketplace_order(
        self,
        *,
        agent_id: int,
        gpu_required: int = 1,
        pricing_type: str = "minute",
        docker_image: str = "vastai/pytorch:cuda-12.9.1-auto",
        ssh_key_ids: list[int] | None = None,
        disk_gb: int = 250,
        volume_id: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "gpu_required": gpu_required,
            "pricing_type": pricing_type,
            "docker_image": docker_image,
            "disk_gb": disk_gb,
        }
        if ssh_key_ids:
            body["ssh_key_ids"] = ssh_key_ids
        if volume_id is not None:
            body["volume_id"] = volume_id
        return self._request(
            "POST",
            "/v1/marketplace/order",
            json=body,
            timeout=300.0,
            retry_on_error=False,
        )["data"]

    def list_volumes(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/volumes/")["data"]

    def list_ssh_keys(self) -> dict[str, Any]:
        return self._request("GET", "/v1/settings/ssh-keys")["data"]

    def create_ssh_key(self, *, key_name: str, public_key: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/settings/ssh-keys",
            json={"key_name": key_name, "public_key": public_key},
        )["data"]

    def deploy_cloud_gpu(
        self,
        *,
        gpu_name: str,
        gpu_count: int = 1,
        docker_image: str = "pytorch/pytorch:latest",
        disk_gb: int = 150,
        volume_id: int | None = None,
        secure_cloud: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "docker_image": docker_image,
            "disk_gb": disk_gb,
            "secure_cloud": secure_cloud,
        }
        if volume_id is not None:
            body["volume_id"] = volume_id
        return self._request(
            "POST",
            "/v1/gpu-cloud/deploy",
            json=body,
            timeout=300.0,
            retry_on_error=False,
        )["data"]

    def start_instance(self, task_id: int) -> dict[str, Any]:
        return self._request("POST", f"/v1/instances/{task_id}/start")["data"]

    def stop_instance(self, task_id: int) -> dict[str, Any]:
        return self._request("POST", f"/v1/instances/{task_id}/stop")["data"]

    def delete_instance(self, task_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/instances/{task_id}")["data"]

    # ---- Media & LLM API ----

    def create_image_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/llm/images/jobs", json=payload, timeout=90.0)["data"]

    def get_image_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/llm/images/jobs/{job_id}", timeout=60.0)["data"]

    def list_topaz_models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/llm/topaz/models", timeout=60.0)["data"]

    def create_topaz_video_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/llm/topaz/video/jobs", json=payload, timeout=120.0)[
            "data"
        ]

    def get_topaz_video_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/llm/topaz/video/jobs/{job_id}", timeout=60.0)["data"]

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/openai/chat/completions", json=payload, timeout=120.0)
