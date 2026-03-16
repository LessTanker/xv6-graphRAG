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
        throw new Error(payload.error ?? "request failure");
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
      const message = err instanceof Error ? err.message : "unknown error";
      setError(message);
      setStatus("error");
    }
  }, []);

  const statusText = useMemo(() => {
    if (status === "loading") {
      return "requesting backend";
    }
    if (status === "success") {
      return "done";
    }
    if (status === "error") {
      return "error";
    }
    return "wait for input";
  }, [status]);

  return {
    ask,
    statusText,
    result,
    error,
    isLoading: status === "loading"
  };
}
