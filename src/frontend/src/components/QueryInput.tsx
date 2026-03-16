import { FormEvent, useState } from "react";

interface QueryInputProps {
  disabled: boolean;
  onSubmit: (query: string) => Promise<void>;
}

export default function QueryInput({ disabled, onSubmit }: QueryInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onSubmit(query);
  };

  return (
    <form className="grid gap-3 p-4 sm:p-6" onSubmit={handleSubmit}>
      <textarea
        className="w-full min-h-[120px] resize-y rounded-xl border border-border bg-white p-4 outline-none transition focus:border-accent"
        placeholder="例如：trap path 是如何到达 usertrap 的？"
        required
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div className="flex items-center gap-3">
        <button
          type="submit"
          className="rounded-full bg-accent px-5 py-2 font-bold text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={disabled}
        >
          提交问题
        </button>
      </div>
    </form>
  );
}
