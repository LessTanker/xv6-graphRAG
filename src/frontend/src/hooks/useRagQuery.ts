import { useCallback, useEffect, useMemo, useState } from "react";
import type { RagEdge, RagResponse } from "../types/rag";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001/api/query";
const NODES_API_URL = import.meta.env.VITE_NODES_API_URL ?? "http://127.0.0.1:8001/api/nodes";
const EDGES_API_URL = import.meta.env.VITE_EDGES_API_URL ?? "http://127.0.0.1:8001/api/edges";
const EXPERT_PATHS_API_URL = import.meta.env.VITE_EXPERT_PATHS_API_URL ?? "http://127.0.0.1:8001/api/expert-paths";

type QueryStatus = "idle" | "loading" | "success" | "error";

export function useRagQuery() {
  const [status, setStatus] = useState<QueryStatus>("idle");
  const [result, setResult] = useState<RagResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [allNodes, setAllNodes] = useState<RagResponse["directly_related_nodes"]>([]);
  const [allEdges, setAllEdges] = useState<RagEdge[]>([]);
  const [globalExpertPathIds, setGlobalExpertPathIds] = useState<number[]>([]);

  const refreshNodes = useCallback(async () => {
    try {
      const response = await fetch(NODES_API_URL, { method: "GET" });
      const nodesPayload = (await response.json()) as { nodes?: RagResponse["directly_related_nodes"]; error?: string };
      if (response.ok) {
        setAllNodes(Array.isArray(nodesPayload.nodes) ? nodesPayload.nodes : []);
      } else {
        setAllNodes([]);
      }
    } catch {
      setAllNodes([]);
    }
  }, []);

  useEffect(() => {
    let isCancelled = false;

    const loadGraph = async () => {
      const [nodesResult, edgesResult, expertResult] = await Promise.allSettled([
        fetch(NODES_API_URL, { method: "GET" }),
        fetch(EDGES_API_URL, { method: "GET" }),
        fetch(EXPERT_PATHS_API_URL, { method: "GET" })
      ]);

      if (isCancelled) {
        return;
      }

      if (nodesResult.status === "fulfilled") {
        try {
          const nodesPayload = (await nodesResult.value.json()) as { nodes?: RagResponse["directly_related_nodes"]; error?: string };
          if (nodesResult.value.ok) {
            setAllNodes(Array.isArray(nodesPayload.nodes) ? nodesPayload.nodes : []);
          } else {
            setAllNodes([]);
          }
        } catch {
          setAllNodes([]);
        }
      } else {
        setAllNodes([]);
      }

      if (edgesResult.status === "fulfilled") {
        try {
          const edgesPayload = (await edgesResult.value.json()) as { edges?: RagEdge[]; error?: string };
          if (edgesResult.value.ok) {
            setAllEdges(Array.isArray(edgesPayload.edges) ? edgesPayload.edges : []);
          } else {
            setAllEdges([]);
          }
        } catch {
          setAllEdges([]);
        }
      } else {
        setAllEdges([]);
      }

      if (expertResult.status === "fulfilled" && expertResult.value.ok) {
        try {
          const expertPayload = (await expertResult.value.json()) as { expert_paths?: { node_ids: number[] }[] };
          const allIds = (expertPayload.expert_paths ?? []).flatMap((p) => p.node_ids);
          setGlobalExpertPathIds(Array.from(new Set(allIds)));
        } catch {
          setGlobalExpertPathIds([]);
        }
      }
    };

    void loadGraph();

    return () => {
      isCancelled = true;
    };
  }, []);

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

      // The backend may enrich chunks_metadata during query; refresh graph nodes to stay in sync.
      await refreshNodes();

      setStatus("success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error";
      setError(message);
      setStatus("error");
    }
  }, [refreshNodes]);

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
    allNodes,
    allEdges,
    globalExpertPathIds,
    error,
    isLoading: status === "loading"
  };
}
