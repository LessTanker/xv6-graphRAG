import GraphViewer from "./components/GraphViewer";
import QueryInput from "./components/QueryInput";
import ResultDisplay from "./components/ResultDisplay";
import { useRagQuery } from "./hooks/useRagQuery";

export default function App() {
  const { ask, statusText, result, allNodes, allEdges, error, isLoading } = useRagQuery();

  return (
    <main className="min-h-screen px-4 py-6 lg:px-6">
      <div className="mx-auto flex max-w-[1400px] flex-col gap-4 lg:flex-row lg:items-start">
        <section className="order-2 lg:order-1 lg:h-[calc(100vh-3rem)] lg:w-1/3 lg:grid lg:grid-rows-[1fr_1fr] lg:gap-4">
          <div className="overflow-hidden rounded-2xl bg-panel">
            <GraphViewer
              allNodes={allNodes}
              allEdges={allEdges}
              topNodes={result?.top3_nodes ?? []}
              relatedNodes={result?.directly_related_nodes ?? []}
              height={260}
              borderless
            />
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
