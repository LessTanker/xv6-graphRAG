import GraphViewer from "./components/GraphViewer";
import QueryInput from "./components/QueryInput";
import ResultDisplay from "./components/ResultDisplay";
import { useRagQuery } from "./hooks/useRagQuery";

export default function App() {
  const { ask, statusText, result, error, isLoading } = useRagQuery();

  return (
    <main className="flex min-h-screen">
      <aside className="hidden w-1/3 lg:block" />
      <section className="mx-auto my-8 flex-1 max-w-5xl px-4">
        <div className="overflow-hidden rounded-2xl border border-border bg-panel shadow-card">
          <header className="border-b border-border bg-gradient-to-br from-[#f4cfb8] to-[#fff2e8] px-4 py-5 sm:px-6">
            <h1 className="text-2xl font-semibold tracking-wide">xv6-riscv graphRAG </h1>
          </header>

          <QueryInput disabled={isLoading} onSubmit={ask} statusText={statusText} />

          <section className="border-t border-dashed border-border px-4 py-4 sm:px-6">
            <h2 className="mb-3 text-lg font-semibold">Output</h2>
            <ResultDisplay markdown={error ? `请求失败: ${error}` : result?.llm_response ?? ""} />
          </section>

          <GraphViewer topNodes={result?.top3_nodes ?? []} relatedNodes={result?.directly_related_nodes ?? []} plan={result?.query_plan ?? null} />
        </div>
      </section>
    </main>
  );
}
