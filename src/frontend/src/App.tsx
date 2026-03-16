import GraphViewer from "./components/GraphViewer";
import QueryInput from "./components/QueryInput";
import ResultDisplay from "./components/ResultDisplay";
import { useRagQuery } from "./hooks/useRagQuery";

export default function App() {
  const { ask, statusText, result, error, isLoading } = useRagQuery();

  return (
    <main className="mx-auto my-8 max-w-5xl px-4">
      <section className="overflow-hidden rounded-2xl border border-border bg-panel shadow-card">
        <header className="border-b border-border bg-gradient-to-br from-[#f4cfb8] to-[#fff2e8] px-4 py-5 sm:px-6">
          <h1 className="text-2xl font-semibold tracking-wide">xv6 GraphRAG 网页问答</h1>
          <p className="mt-2 text-sm text-ink/80">输入问题并提交，回答会渲染为 Markdown，代码块支持高亮。</p>
        </header>

        <QueryInput disabled={isLoading} onSubmit={ask} />

        <div className="flex items-center border-t border-border px-4 py-2 text-sm text-ink/80 sm:px-6">
          <span>{statusText}</span>
        </div>

        <section className="border-t border-dashed border-border px-4 py-4 sm:px-6">
          <h2 className="mb-3 text-lg font-semibold">回答</h2>
          <ResultDisplay markdown={error ? `请求失败: ${error}` : result?.llm_response ?? ""} />
        </section>

        <GraphViewer topNodes={result?.top3_nodes ?? []} relatedNodes={result?.directly_related_nodes ?? []} plan={result?.query_plan ?? null} />
      </section>
    </main>
  );
}
