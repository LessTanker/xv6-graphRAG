import ForceGraph2D from "react-force-graph-2d";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RagEdge, RagNode } from "../types/rag";

type GraphRole = "top" | "related" | "ambient";

interface GraphNode {
  id: string;
  name: string;
  type: string;
  role: GraphRole;
  color: string;
  size: number;
  raw: RagNode;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string;
  target: string;
  kind: "top" | "related" | "ambient";
  edgeType?: string;
}

interface GraphViewerProps {
  allNodes: RagNode[];
  allEdges: RagEdge[];
  topNodes: RagNode[];
  relatedNodes: RagNode[];
  onNodeSelect?: (node: RagNode) => void;
  height?: number;
  borderless?: boolean;
}

export default function GraphViewer({ allNodes, allEdges, topNodes, relatedNodes, onNodeSelect, height = 280, borderless = false }: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [graphWidth, setGraphWidth] = useState(0);
  const [graphHeight, setGraphHeight] = useState(0);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      setGraphWidth(Math.floor(entry.contentRect.width));
      setGraphHeight(Math.floor(entry.contentRect.height));
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (allNodes.length === 0) {
      return { nodes: [] as GraphNode[], links: [] as GraphLink[] };
    }

    const makeNodeKey = (node: RagNode, fallbackIndex: number, label: string) => {
      if (typeof node.id === "number") {
        return `id:${node.id}`;
      }
      if (node.name) {
        return `name:${node.name}:${node.type ?? "unknown"}:${node.file ?? "nofile"}`;
      }
      return `${label}:${fallbackIndex}`;
    };

    const nodes = new Map<string, GraphNode>();
    const links: GraphLink[] = [];
    const keyToGraphId = new Map<string, string>();
    const idNumberToGraphId = new Map<number, string>();

    allNodes.forEach((node, index) => {
      const key = makeNodeKey(node, index, "all");
      const id = `node-${key}`;
      keyToGraphId.set(key, id);
      if (typeof node.id === "number") {
        idNumberToGraphId.set(node.id, id);
      }
      if (!nodes.has(id)) {
        nodes.set(id, {
          id,
          name: node.name ?? `Node ${index + 1}`,
          type: node.type ?? "unknown",
          role: "ambient",
          color: "#d8a277",
          size: 3.4,
          raw: node
        });
      }
    });

    const resolveGraphId = (node: RagNode, index: number, label: string) => {
      if (typeof node.id === "number") {
        return idNumberToGraphId.get(node.id);
      }
      const key = makeNodeKey(node, index, label);
      return keyToGraphId.get(key);
    };

    const relatedIds = new Set<string>();
    relatedNodes.forEach((node, index) => {
      const resolved = resolveGraphId(node, index, "related");
      if (resolved) {
        relatedIds.add(resolved);
      }
    });

    const topIds = new Set<string>();
    topNodes.forEach((node, index) => {
      const resolved = resolveGraphId(node, index, "top");
      if (resolved) {
        topIds.add(resolved);
      }
    });

    nodes.forEach((graphNode, id) => {
      if (topIds.has(id)) {
        graphNode.role = "top";
        graphNode.color = "#a72323";
        graphNode.size = 8.6;
      } else if (relatedIds.has(id)) {
        graphNode.role = "related";
        graphNode.color = "#b6462a";
        graphNode.size = 7.2;
      }
    });

    const centerIds = Array.from(new Set([...Array.from(topIds), ...Array.from(relatedIds)]));

    centerIds.forEach((id, index) => {
      const graphNode = nodes.get(id);
      if (!graphNode) {
        return;
      }
      const angle = (index / Math.max(centerIds.length, 1)) * Math.PI * 2;
      const radius = graphNode.role === "top" ? 30 : 62;
      graphNode.x = Math.cos(angle) * radius;
      graphNode.y = Math.sin(angle) * radius;
    });

    const outerIds = Array.from(nodes.keys()).filter((id) => !centerIds.includes(id));
    outerIds.forEach((id, index) => {
      const graphNode = nodes.get(id);
      if (!graphNode) {
        return;
      }
      const angle = (index / Math.max(outerIds.length, 1)) * Math.PI * 2;
      const orbit = 150 + (index % 11) * 7;
      graphNode.x = Math.cos(angle) * orbit;
      graphNode.y = Math.sin(angle) * orbit;
    });

    const topSet = new Set(topIds);
    const relatedSet = new Set(relatedIds);
    const seenEdge = new Set<string>();

    allEdges.forEach((edge) => {
      if (typeof edge.from !== "number" || typeof edge.to !== "number") {
        return;
      }

      const source = idNumberToGraphId.get(edge.from);
      const target = idNumberToGraphId.get(edge.to);
      if (!source || !target) {
        return;
      }

      const dedupeKey = `${source}->${target}:${edge.type ?? ""}`;
      if (seenEdge.has(dedupeKey)) {
        return;
      }
      seenEdge.add(dedupeKey);

      const sourceIsTop = topSet.has(source);
      const targetIsTop = topSet.has(target);
      const sourceIsRelated = relatedSet.has(source);
      const targetIsRelated = relatedSet.has(target);

      let kind: GraphLink["kind"] = "ambient";
      if (sourceIsTop && targetIsTop) {
        kind = "top";
      } else if ((sourceIsTop && targetIsRelated) || (sourceIsRelated && targetIsTop) || (sourceIsRelated && targetIsRelated)) {
        kind = "related";
      }

      links.push({ source, target, kind, edgeType: edge.type });
    });

    return {
      nodes: Array.from(nodes.values()),
      links
    };
  }, [allEdges, allNodes, relatedNodes, topNodes]);

  const isEmpty = allNodes.length === 0;
  const renderHeight = graphHeight || height;

  const drawGlowNode = useCallback((node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const graphNode = node as GraphNode;

    const x = (node as { x?: number }).x ?? 0;
    const y = (node as { y?: number }).y ?? 0;

    let coreRadius = 1.6;
    let glowRadius = 8;
    let glowInnerColor = "rgba(92, 170, 255, 0.85)";
    let glowMidColor = "rgba(92, 170, 255, 0.28)";

    if (graphNode.role === "top") {
      coreRadius = 2.9;
      glowRadius = 17;
      glowInnerColor = "rgba(255, 45, 45, 1.0)";
      glowMidColor = "rgba(255, 45, 45, 0.46)";
    } else if (graphNode.role === "related") {
      coreRadius = 2.4;
      glowRadius = 14;
      glowInnerColor = "rgba(255, 80, 80, 0.95)";
      glowMidColor = "rgba(255, 80, 80, 0.35)";
    } else {
      coreRadius = 1.5;
      glowRadius = 9;
      glowInnerColor = "rgba(255, 170, 80, 0.8)";
      glowMidColor = "rgba(255, 170, 80, 0.28)";
    }

    // Keep glow stable at different zoom levels while preserving tiny core points.
    const zoomSafeCore = coreRadius / Math.sqrt(globalScale);
    const zoomSafeGlow = glowRadius / Math.sqrt(globalScale);

    // Subtle pulsation to create a tiny twinkle effect.
    const now = Date.now() * 0.003;
    const phase = graphNode.id.length * 0.7;
    const pulse = 0.96 + 0.04 * Math.sin(now + phase);
    const pulseGlow = zoomSafeGlow * pulse;
    const pulseCore = zoomSafeCore * (0.985 + 0.02 * Math.sin(now * 1.3 + phase));

    ctx.save();

    const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseGlow);
    gradient.addColorStop(0, glowInnerColor);
    gradient.addColorStop(0.38, glowMidColor);
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, pulseGlow, 0, 2 * Math.PI, false);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, pulseCore, 0, 2 * Math.PI, false);
    ctx.fill();

    ctx.restore();
  }, []);

  return (
    <section className="flex h-full flex-col p-4 sm:p-5">
      <h3 className="mb-2 text-base font-semibold text-ink">Knowledge Graph</h3>

      <div className={`relative mt-2 flex-1 rounded-md bg-white ${borderless ? "" : "border border-border"}`} ref={containerRef}>
        {isEmpty ? (
          <div className="flex items-center justify-center text-sm text-ink/70" style={{ height: renderHeight }}>
            暂无图谱数据，请先提交查询。
          </div>
        ) : (
          <>
            <ForceGraph2D
              graphData={graphData}
              width={graphWidth || 300}
              height={renderHeight}
              backgroundColor="#ffffff"
              nodeCanvasObjectMode={() => "replace"}
              nodeCanvasObject={drawGlowNode}
              autoPauseRedraw={false}
              enableNodeDrag={false}
              nodeLabel={() => ""}
              onNodeHover={(node) => setHoveredNode((node as GraphNode | null) ?? null)}
              onNodeClick={(node) => {
                const clickedNode = node as GraphNode | null;
                if (clickedNode && onNodeSelect) {
                  onNodeSelect(clickedNode.raw);
                }
              }}
              linkColor={(link) => {
                const graphLink = link as GraphLink;
                if (graphLink.kind === "top") {
                  return "rgba(255, 60, 60, 0.92)";
                }
                if (graphLink.kind === "related") {
                  return "rgba(255, 92, 92, 0.72)";
                }
                return "rgba(120, 120, 120, 0.16)";
              }}
              linkWidth={(link) => {
                const graphLink = link as GraphLink;
                if (graphLink.kind === "top") {
                  return 1.9;
                }
                if (graphLink.kind === "related") {
                  return 1.35;
                }
                return 0.45;
              }}
              linkDirectionalParticles={1}
              linkDirectionalParticleWidth={0.8}
              linkDirectionalParticleSpeed={0.0015}
              linkDirectionalParticleColor={(link) => {
                const graphLink = link as GraphLink;
                if (graphLink.kind === "top") {
                  return "rgba(255, 50, 50, 0.9)";
                }
                if (graphLink.kind === "related") {
                  return "rgba(255, 78, 78, 0.74)";
                }
                return "rgba(50, 50, 50, 0.2)";
              }}
              cooldownTicks={110}
            />

            {hoveredNode ? (
              <div className="pointer-events-none absolute left-3 top-3 rounded border border-border bg-white/90 px-2 py-1 text-xs text-ink shadow-lg backdrop-blur-sm">
                <div className="font-semibold">{hoveredNode.name}</div>
                <div className="text-ink/70">{hoveredNode.type}</div>
              </div>
            ) : null}
          </>
        )}
      </div>
    </section>
  );
}
