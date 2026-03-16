import type { RagNode, RagQueryPlan } from "../types/rag";

interface GraphViewerProps {
  topNodes: RagNode[];
  relatedNodes: RagNode[];
  plan: RagQueryPlan | null;
}

export default function GraphViewer({ topNodes, relatedNodes, plan }: GraphViewerProps) {
  return (
    <section className="border-t border-dashed border-border px-4 py-4 sm:px-6">
      <h3 className="mb-2 text-base font-semibold">检索概览</h3>
      <p className="text-sm text-ink/80">
        Top 节点: <span className="font-semibold">{topNodes.length}</span>，关联节点:
        <span className="font-semibold"> {relatedNodes.length}</span>
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-white p-3">
          <h4 className="mb-2 text-sm font-semibold">Query Plan</h4>
          <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-ink/80">{JSON.stringify(plan ?? {}, null, 2)}</pre>
        </div>
        <div className="rounded-lg border border-border bg-white p-3">
          <h4 className="mb-2 text-sm font-semibold">Top Nodes</h4>
          <ul className="space-y-1 text-sm text-ink/80">
            {topNodes.slice(0, 5).map((node, idx) => (
              <li key={`${node.id ?? idx}-${node.name ?? "node"}`}>
                {node.name ?? "unknown"} {node.file ? `(${node.file})` : ""}
              </li>
            ))}
            {topNodes.length === 0 ? <li>暂无</li> : null}
          </ul>
        </div>
      </div>
    </section>
  );
}
