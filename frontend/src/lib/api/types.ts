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
  message_converter: string | null;
  enable_auto_tool_choice: boolean;
  trust_remote_code: boolean;
  chat_template_file: string | null;
  log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR";
  log_file: string | null;
  no_log_file: boolean;
  auto_start: boolean;
  system_prompt: string | null;
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
  message_converter?: string;
  enable_auto_tool_choice?: boolean;
  trust_remote_code?: boolean;
  chat_template_file?: string;
  log_level?: string;
  log_file?: string;
  no_log_file?: boolean;
  auto_start?: boolean;
  system_prompt?: string;
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
  message_converter?: string | null;
  enable_auto_tool_choice?: boolean;
  trust_remote_code?: boolean;
  chat_template_file?: string | null;
  log_level?: string;
  log_file?: string | null;
  no_log_file?: boolean;
  auto_start?: boolean;
  system_prompt?: string | null;
}

export interface RunningServer {
  profile_id: number;
  profile_name: string;
  pid: number;
  port: number;
  health_status: "starting" | "healthy" | "unhealthy" | "stopped";
  uptime_seconds: number;
  memory_mb: number;
  memory_percent: number;
  memory_limit_percent: number;
  cpu_percent: number;
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

export interface ModelCharacteristics {
  model_type?: string;
  architecture_family?: string;
  max_position_embeddings?: number;
  num_hidden_layers?: number;
  hidden_size?: number;
  vocab_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  quantization_bits?: number;
  quantization_group_size?: number;
  is_multimodal?: boolean;
  multimodal_type?: string;
  use_cache?: boolean;
  is_tool_use?: boolean;
}

export interface LocalModel {
  model_id: string;
  local_path: string;
  size_bytes: number;
  size_gb: number;
  characteristics?: ModelCharacteristics;
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

export interface ModelDetectionInfo {
  model_family: string | null;
  recommended_options: {
    tool_call_parser?: string;
    reasoning_parser?: string;
    message_converter?: string;
  };
  is_downloaded: boolean;
}

export interface ParserOptions {
  tool_call_parsers: string[];
  reasoning_parsers: string[];
  message_converters: string[];
}

// Auth types
export type UserStatus = "pending" | "approved" | "disabled";

export interface User {
  id: number;
  email: string;
  is_admin: boolean;
  status: UserStatus;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
}

export interface UserCreate {
  email: string;
  password: string;
}

export interface PasswordReset {
  password: string;
}

export interface UserUpdate {
  email?: string;
  is_admin?: boolean;
  status?: UserStatus;
}

// Chat message types for multimodal support
export interface ContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: {
    url: string; // "data:image/png;base64,..." or HTTP URL
  };
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string | ContentPart[];
}

export interface Attachment {
  file: File;
  preview: string; // Object URL for thumbnail, or filename for text
  type: "image" | "video" | "text";
}

// Tool-use types for MCP integration
export interface ToolCall {
  id: string;
  index?: number;
  function: {
    name: string;
    arguments: string; // JSON-encoded string
  };
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// Settings API Types

// API protocol types for cloud providers
export type ApiType = "openai" | "anthropic";

// Backend types for routing and providers
export type BackendType =
  | "local"
  | "openai"
  | "anthropic"
  | "openai_compatible"
  | "anthropic_compatible"
  | "together"
  | "groq"
  | "fireworks"
  | "mistral"
  | "deepseek";

export type PatternType = "exact" | "prefix" | "regex";
export type EvictionPolicy = "lru" | "lfu" | "ttl";
export type MemoryLimitMode = "percent" | "gb";

// Cloud provider credentials
export interface CloudCredential {
  id: number;
  backend_type: BackendType;
  api_type: ApiType;
  name: string;
  base_url: string | null;
  created_at: string;
}

export interface CloudCredentialCreate {
  backend_type: BackendType;
  api_type?: ApiType;
  name?: string;
  api_key: string;
  base_url?: string;
}

// Provider defaults from API
export interface ProviderDefault {
  backend_type: BackendType;
  base_url: string;
  api_type: ApiType;
}

// Backend routing rules
export interface BackendMapping {
  id: number;
  model_pattern: string;
  pattern_type: PatternType;
  backend_type: BackendType;
  backend_model: string | null;
  fallback_backend: BackendType | null;
  priority: number;
  enabled: boolean;
}

export interface BackendMappingCreate {
  model_pattern: string;
  pattern_type?: PatternType;
  backend_type: BackendType;
  backend_model?: string;
  fallback_backend?: BackendType;
  priority?: number;
}

export interface BackendMappingUpdate {
  model_pattern?: string;
  pattern_type?: PatternType;
  backend_type?: BackendType;
  backend_model?: string | null;
  fallback_backend?: BackendType | null;
  priority?: number;
  enabled?: boolean;
}

// Server pool configuration
export interface ServerPoolConfig {
  memory_limit_mode: MemoryLimitMode;
  memory_limit_value: number;
  eviction_policy: EvictionPolicy;
  preload_models: string[];
}

export interface ServerPoolConfigUpdate {
  memory_limit_mode?: MemoryLimitMode;
  memory_limit_value?: number;
  eviction_policy?: EvictionPolicy;
  preload_models?: string[];
}

// Rule test result
export interface RuleTestResult {
  matched_rule_id: number | null;
  backend_type: BackendType;
}

// Timeout settings
export interface TimeoutSettings {
  chat_seconds: number;
  completions_seconds: number;
  embeddings_seconds: number;
}

export interface TimeoutSettingsUpdate {
  chat_seconds?: number;
  completions_seconds?: number;
  embeddings_seconds?: number;
}

// Audit log types
export interface AuditLog {
  id: number;
  request_id: string;
  timestamp: string;
  model: string;
  backend_type: "local" | "openai" | "anthropic";
  endpoint: string;
  duration_ms: number;
  status: "success" | "error" | "timeout";
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  error_type: string | null;
  error_message: string | null;
}

export interface AuditLogFilter {
  model?: string;
  backend_type?: string;
  status?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}

export interface AuditStats {
  total_requests: number;
  by_status: Record<string, number>;
  by_backend: Record<string, number>;
  unique_models: number;
}
