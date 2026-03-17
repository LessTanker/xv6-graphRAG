export interface RagNode {
  id?: number;
  type?: string;
  name?: string;
  file?: string;
  summary?: string;
  keywords?: string[];
  code?: string;
  similarity?: number;
  distance?: number;
  [key: string]: unknown;
}

export interface RagEdge {
  from: number;
  to: number;
  type?: string;
}

export interface RagQueryPlan {
  query_type?: string;
  traversal_strategy?: string;
  seed_selection_hint?: string;
  reasoning?: string;
  target_entity?: string;
  restricted_community_id?: number | null;
}

export interface RagResponse {
  query: string;
  llm_response: string;
  top3_nodes: RagNode[];
  directly_related_nodes: RagNode[];
  query_plan: RagQueryPlan;
}

export interface ExpertPath {
  theme: string;
  path_summary: string;
  node_ids: number[];
  source?: string;
}

export interface ExpertPathsResponse {
  paths: ExpertPath[];
  path_count: number;
}
