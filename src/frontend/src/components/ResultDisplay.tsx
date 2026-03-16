import DOMPurify from "dompurify";
import hljs from "highlight.js";
import "highlight.js/styles/github.css";
import { marked } from "marked";
import { useMemo } from "react";

marked.setOptions({
  gfm: true,
  breaks: true
});

interface ResultDisplayProps {
  markdown: string;
}

export default function ResultDisplay({ markdown }: ResultDisplayProps) {

  const safeHtml = useMemo(() => {
    const source = (markdown || "").trim() || "Answers shown here";
    const renderer = new marked.Renderer();
    renderer.code = (text: string, lang?: string) => {
      const normalizedLang = (lang ?? "").trim();
      const highlighted = normalizedLang && hljs.getLanguage(normalizedLang)
        ? hljs.highlight(text, { language: normalizedLang }).value
        : hljs.highlightAuto(text).value;
      const className = normalizedLang ? `language-${normalizedLang}` : "";
      return `<pre><code class="hljs ${className}">${highlighted}</code></pre>`;
    };

    const html = marked.parse(source, { renderer });
    return DOMPurify.sanitize(typeof html === "string" ? html : "");
  }, [markdown]);

  return <div className="markdown-body text-ink" dangerouslySetInnerHTML={{ __html: safeHtml }} />;
}
