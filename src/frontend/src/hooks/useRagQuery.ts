import { useCallback, useMemo, useState } from "react";
import type { RagResponse } from "../types/rag";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001/api/query";

type QueryStatus = "idle" | "loading" | "success" | "error";

export function useRagQuery() {
  const [status, setStatus] = useState<QueryStatus>("idle");
  const [result, setResult] = useState<RagResponse | null>(null);
  const [error, setError] = useState<string>("");

  const ask = useCallback(async (query: string) => {
    const clean = query.trim();
    if (!clean) {
      return;
    }

    setStatus("loading");
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: clean })
      });

      const payload = (await response.json()) as Partial<RagResponse> & { error?: string };

      if (!response.ok) {
        throw new Error(payload.error ?? "请求失败");
      }

      setResult({
        query: payload.query ?? clean,
        llm_response: payload.llm_response ?? "",
        top3_nodes: payload.top3_nodes ?? [],
        directly_related_nodes: payload.directly_related_nodes ?? [],
        query_plan: payload.query_plan ?? {}
      });
      setStatus("success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "未知错误";
      setError(message);
      setStatus("error");
    }
  }, []);

  const statusText = useMemo(() => {
    if (status === "loading") {
      return "正在请求后端...";
    }
    if (status === "success") {
      return "已完成";
    }
    if (status === "error") {
      return "出错";
    }
    return "等待输入";
  }, [status]);

  return {
    ask,
    status,
    statusText,
    result,
    error,
    isLoading: status === "loading"
  };
}
