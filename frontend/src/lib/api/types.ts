// API Types for MLX Model Manager

export interface ServerProfile {
  id: number;
  name: string;
  description: string | null;
  model_path: string;
  model_type:
    | "lm"
    | "multimodal"
    | "whisper"
    | "embeddings"
    | "image-generation"
    | "image-edit";
  port: number;
  host: string;
  context_length: number | null;
  max_concurrency: number;
  queue_timeout: number;
  queue_size: number;
  tool_call_parser: string | null;
  reasoning_parser: string | null;
  enable_auto_tool_choice: boolean;
  trust_remote_code: boolean;
  chat_template_file: string | null;
  log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  log_file: string | null;
  no_log_file: boolean;
  auto_start: boolean;
  launchd_installed: boolean;
  created_at: string;
  updated_at: string;
}

export interface ServerProfileCreate {
  name: string;
  description?: string;
  model_path: string;
  model_type?: string;
  port: number;
  host?: string;
  context_length?: number;
  max_concurrency?: number;
  queue_timeout?: number;
  queue_size?: number;
  tool_call_parser?: string;
  reasoning_parser?: string;
  enable_auto_tool_choice?: boolean;
  trust_remote_code?: boolean;
  chat_template_file?: string;
  log_level?: string;
  log_file?: string;
  no_log_file?: boolean;
  auto_start?: boolean;
}

export interface ServerProfileUpdate {
  name?: string;
  description?: string;
  model_path?: string;
  model_type?: string;
  port?: number;
  host?: string;
  context_length?: number | null;
  max_concurrency?: number;
  queue_timeout?: number;
  queue_size?: number;
  tool_call_parser?: string | null;
  reasoning_parser?: string | null;
  enable_auto_tool_choice?: boolean;
  trust_remote_code?: boolean;
  chat_template_file?: string | null;
  log_level?: string;
  log_file?: string | null;
  no_log_file?: boolean;
  auto_start?: boolean;
}

export interface RunningServer {
  profile_id: number;
  profile_name: string;
  pid: number;
  port: number;
  health_status: "starting" | "healthy" | "unhealthy" | "stopped";
  uptime_seconds: number;
  memory_mb: number;
}

export interface ModelSearchResult {
  model_id: string;
  author: string;
  downloads: number;
  likes: number;
  estimated_size_gb: number;
  tags: string[];
  is_downloaded: boolean;
  last_modified: string | null;
}

export interface LocalModel {
  model_id: string;
  local_path: string;
  size_bytes: number;
  size_gb: number;
}

export interface SystemMemory {
  total_gb: number;
  available_gb: number;
  used_gb: number;
  percent_used: number;
  mlx_recommended_gb: number;
}

export interface SystemInfo {
  os_version: string;
  chip: string;
  memory_gb: number;
  python_version: string;
  mlx_version: string | null;
  mlx_openai_server_version: string | null;
}

export interface DownloadProgress {
  status: "starting" | "downloading" | "completed" | "failed";
  model_id: string;
  progress: number;
  downloaded_bytes?: number;
  total_bytes?: number;
  speed_mbps?: number;
  error?: string;
}

export interface HealthStatus {
  status: "healthy" | "unhealthy" | "starting" | "stopped";
  response_time_ms?: number;
  model_loaded?: boolean;
  error?: string;
}

export interface LaunchdStatus {
  installed: boolean;
  running: boolean;
  pid?: number;
  label: string;
}

export interface ServerStatus {
  profile_id: number;
  running: boolean;
  pid?: number;
  exit_code?: number;
  failed: boolean;
  error_message?: string;
}
