import GraphViewer from "./components/GraphViewer";
import QueryInput from "./components/QueryInput";
import ResultDisplay from "./components/ResultDisplay";
import { useRagQuery } from "./hooks/useRagQuery";
import DOMPurify from "dompurify";
import hljs from "highlight.js";
import { useMemo, useState } from "react";
import type { RagNode } from "./types/rag";

export default function App() {
  const { ask, statusText, result, allNodes, allEdges, error, isLoading } = useRagQuery();
  const [selectedNode, setSelectedNode] = useState<RagNode | null>(null);

  const highlightedCodeHtml = useMemo(() => {
    const rawCode = typeof selectedNode?.code === "string" ? selectedNode.code.trim() : "";
    if (!rawCode) {
      return "<pre><code class=\"hljs\">N/A</code></pre>";
    }

    const preferredLang = selectedNode?.file?.endsWith(".h") || selectedNode?.file?.endsWith(".c") ? "c" : "";
    const highlighted = preferredLang && hljs.getLanguage(preferredLang)
      ? hljs.highlight(rawCode, { language: preferredLang }).value
      : hljs.highlightAuto(rawCode).value;

    return DOMPurify.sanitize(`<pre><code class=\"hljs\">${highlighted}</code></pre>`);
  }, [selectedNode]);

  return (
    <main className="min-h-screen px-4 py-6 lg:px-6">
      <div className="mx-auto flex max-w-[1400px] flex-col gap-4 lg:flex-row lg:items-start">
        <section className="order-2 space-y-4 lg:order-1 lg:w-1/3">
          <div className="overflow-hidden rounded-2xl bg-panel">
            <GraphViewer
              allNodes={allNodes}
              allEdges={allEdges}
              topNodes={result?.top3_nodes ?? []}
              relatedNodes={result?.directly_related_nodes ?? []}
              onNodeSelect={setSelectedNode}
              height={260}
              borderless
            />
          </div>

          <div className="overflow-hidden rounded-2xl border border-border bg-panel shadow-card">
            <section className="h-full px-4 py-4 sm:px-6">
              <h3 className="mb-3 text-base font-semibold">Node Details</h3>
              {selectedNode ? (
                <div className="space-y-3 text-sm text-ink/85">
                  <div className="grid grid-cols-[72px_1fr] gap-y-1">
                    <span className="font-semibold">ID:</span>
                    <span>{selectedNode.id ?? "N/A"}</span>
                    <span className="font-semibold">Name:</span>
                    <span>{selectedNode.name ?? "unknown"}</span>
                    <span className="font-semibold">Type:</span>
                    <span>{selectedNode.type ?? "unknown"}</span>
                    <span className="font-semibold">File:</span>
                    <span>{selectedNode.file ?? "N/A"}</span>
                  </div>

                  <div>
                    <p className="mb-1 font-semibold">Summary</p>
                    <p className="rounded border border-border bg-white/80 px-2 py-1 text-ink/80">{selectedNode.summary?.trim() ? selectedNode.summary : "N/A"}</p>
                  </div>

                  <div>
                    <p className="mb-1 font-semibold">Keywords</p>
                    <p className="rounded border border-border bg-white/80 px-2 py-1 text-ink/80">{Array.isArray(selectedNode.keywords) && selectedNode.keywords.length > 0 ? selectedNode.keywords.join(", ") : "N/A"}</p>
                  </div>

                  <div>
                    <p className="mb-1 font-semibold">Code</p>
                    <div
                      className="markdown-body rounded border border-border bg-white p-2 text-xs text-ink/90"
                      dangerouslySetInnerHTML={{ __html: highlightedCodeHtml }}
                    />
                  </div>
                </div>
              ) : (
                <p className="text-sm text-ink/65">点击上方图谱节点查看详细信息。</p>
              )}
            </section>
          </div>

          <div className="overflow-hidden rounded-2xl border border-border bg-panel shadow-card">
            <section className="h-full px-4 py-4 sm:px-6">
              <h3 className="mb-3 text-base font-semibold">Query Plan</h3>
              <pre className="max-h-[58vh] overflow-auto whitespace-pre-wrap rounded-lg border border-border bg-white p-3 text-xs text-ink/80 lg:h-[calc(100%-2.25rem)] lg:max-h-none">
                {JSON.stringify(result?.query_plan ?? {}, null, 2)}
              </pre>
            </section>
          </div>
        </section>

        <section className="order-1 lg:order-2 lg:w-2/3">
          <div className="overflow-hidden rounded-2xl border border-border bg-panel shadow-card">
            <header className="border-b border-border bg-gradient-to-br from-[#f4cfb8] to-[#fff2e8] px-4 py-5 sm:px-6">
              <h1 className="text-2xl font-semibold tracking-wide">xv6-riscv graphRAG </h1>
            </header>

            <QueryInput disabled={isLoading} onSubmit={ask} statusText={statusText} />

            <section className="border-t border-dashed border-border px-4 py-4 sm:px-6">
              <h2 className="mb-3 text-lg font-semibold">Output</h2>
              <ResultDisplay markdown={error ? `请求失败: ${error}` : result?.llm_response ?? ""} />
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}
