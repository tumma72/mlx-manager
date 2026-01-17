# Phase 2: Server Panel Redesign - Research

**Researched:** 2026-01-17
**Domain:** Real-time dashboard UI, macOS subprocess metrics, Svelte 5 patterns
**Confidence:** MEDIUM (some areas verified with official docs, others require validation)

## Summary

This phase transforms the server panel from a profile-centric list into a metrics-first dashboard. The primary challenges are:

1. **Metrics Collection**: Getting memory/CPU metrics is straightforward with psutil (already in use). GPU metrics on Apple Silicon require additional tooling since psutil doesn't expose GPU data. MLX-specific metrics (tokens/sec) require parsing OpenAI API response usage fields or server logs.

2. **UI Components**: Graphical gauges should use pure SVG/CSS, not external charting libraries. The existing bits-ui library supports the searchable dropdown pattern via its Combobox component.

3. **Scroll Stability**: The project already has sophisticated infrastructure (reconcileArray, keyed each blocks, polling coordinator) that minimizes DOM churn. The current double-RAF scroll restoration approach works but is fragile.

**Primary recommendation:** Extend existing backend metrics with uptime tracking, use SVG-based gauges styled with Tailwind (no new dependencies), implement bits-ui Combobox for profile selection, and enhance the existing reconciliation pattern for scroll stability.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already In Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| bits-ui | ^1.0.0-next | Headless UI primitives | Already used for AlertDialog, has Combobox |
| psutil | 7.x | Process CPU/memory metrics | Cross-platform, already in use |
| lucide-svelte | ^0.469 | Icons | Already in use |
| tailwind-variants | ^0.3 | Component styling | Already in use |

### Supporting (May Need to Add)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| macmon | latest | Apple Silicon GPU/power metrics | Optional: only if GPU metrics are critical |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SVG gauges | Chart.js/svelte-speedometer | External dep, overkill for simple gauges |
| bits-ui Combobox | Native select | No search/filter capability |
| macmon | powermetrics | Requires sudo, more complex |

**Installation:**
No new dependencies required for core functionality. If GPU metrics become critical:
```bash
brew install macmon  # For system-wide Apple Silicon metrics
```

## Architecture Patterns

### Recommended Component Structure
```
src/lib/components/
├── servers/
│   ├── ServerDashboard.svelte      # Main dashboard layout
│   ├── ServerTile.svelte           # Individual running server card
│   ├── ProfileSelector.svelte      # Searchable dropdown + Start button
│   └── MetricGauge.svelte          # Reusable SVG gauge component
└── ui/
    └── combobox.svelte             # bits-ui Combobox wrapper (new)
```

### Pattern 1: Metric Gauge Component (SVG-based)
**What:** Pure SVG circular progress gauge with Tailwind styling
**When to use:** Memory percentage, CPU usage, any 0-100% value
**Example:**
```svelte
<!-- MetricGauge.svelte -->
<script lang="ts">
  interface Props {
    value: number;      // 0-100
    max?: number;       // For non-percentage values
    label: string;
    size?: 'sm' | 'md' | 'lg';
    color?: 'default' | 'warning' | 'danger';
  }

  let { value, max = 100, label, size = 'md', color = 'default' }: Props = $props();

  const percentage = $derived(Math.min(100, Math.max(0, (value / max) * 100)));
  const circumference = 2 * Math.PI * 46; // r=46
  const offset = $derived(circumference - (percentage / 100) * circumference);

  const sizes = { sm: 64, md: 80, lg: 100 };
  const strokeWidths = { sm: 6, md: 8, lg: 10 };

  const colorClass = $derived(
    percentage > 90 ? 'text-red-500' :
    percentage > 75 ? 'text-yellow-500' :
    'text-green-500'
  );
</script>

<div class="relative flex items-center justify-center">
  <svg
    width={sizes[size]}
    height={sizes[size]}
    viewBox="0 0 100 100"
    class="transform -rotate-90"
  >
    <!-- Background circle -->
    <circle
      cx="50" cy="50" r="46"
      stroke="currentColor"
      stroke-width={strokeWidths[size]}
      fill="none"
      class="text-gray-200 dark:text-gray-700"
    />
    <!-- Progress circle -->
    <circle
      cx="50" cy="50" r="46"
      stroke="currentColor"
      stroke-width={strokeWidths[size]}
      fill="none"
      stroke-dasharray={circumference}
      stroke-dashoffset={offset}
      stroke-linecap="round"
      class="{colorClass} transition-all duration-300"
    />
  </svg>
  <div class="absolute text-center">
    <span class="text-sm font-semibold">{value.toFixed(0)}</span>
    <span class="text-xs text-muted-foreground block">{label}</span>
  </div>
</div>
```

### Pattern 2: bits-ui Combobox for Profile Selection
**What:** Searchable dropdown using bits-ui Combobox primitive
**When to use:** Profile selection with search/filter capability
**Example:**
```svelte
<!-- ProfileSelector.svelte -->
<script lang="ts">
  import { Combobox } from 'bits-ui';
  import type { ServerProfile } from '$api';

  interface Props {
    profiles: ServerProfile[];
    onStart: (profile: ServerProfile) => void;
  }

  let { profiles, onStart }: Props = $props();

  let selectedValue = $state<string>('');
  let searchValue = $state('');
  let open = $state(false);

  const filteredProfiles = $derived(
    searchValue === ''
      ? profiles
      : profiles.filter(p =>
          p.name.toLowerCase().includes(searchValue.toLowerCase()) ||
          p.model_path.toLowerCase().includes(searchValue.toLowerCase())
        )
  );

  const selectedProfile = $derived(
    profiles.find(p => p.id?.toString() === selectedValue)
  );

  function handleStart() {
    if (selectedProfile) {
      onStart(selectedProfile);
      selectedValue = '';
      searchValue = '';
    }
  }
</script>

<div class="flex gap-2">
  <Combobox.Root
    type="single"
    bind:value={selectedValue}
    bind:open={open}
    onOpenChangeComplete={(o) => { if (!o) searchValue = ''; }}
  >
    <div class="relative flex-1">
      <Combobox.Input
        class="w-full h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
        placeholder="Select profile to start..."
        oninput={(e) => searchValue = e.currentTarget.value}
      />
      <Combobox.Trigger class="absolute right-2 top-1/2 -translate-y-1/2">
        <!-- chevron icon -->
      </Combobox.Trigger>
    </div>

    <Combobox.Portal>
      <Combobox.Content class="bg-popover border rounded-md shadow-md p-1 max-h-60 overflow-auto">
        <Combobox.Viewport>
          {#each filteredProfiles as profile (profile.id)}
            <Combobox.Item
              value={profile.id?.toString() ?? ''}
              label={profile.name}
              class="px-2 py-1.5 text-sm rounded cursor-pointer data-[highlighted]:bg-accent"
            >
              <span class="font-medium">{profile.name}</span>
              <span class="text-xs text-muted-foreground block truncate">{profile.model_path}</span>
            </Combobox.Item>
          {/each}
          {#if filteredProfiles.length === 0}
            <div class="px-2 py-4 text-sm text-muted-foreground text-center">
              No profiles found
            </div>
          {/if}
        </Combobox.Viewport>
      </Combobox.Content>
    </Combobox.Portal>
  </Combobox.Root>

  <Button onclick={handleStart} disabled={!selectedProfile}>
    <Play class="w-4 h-4 mr-1" />
    Start
  </Button>
</div>
```

### Pattern 3: Scroll Preservation with Key-Based Identity
**What:** Maintain scroll position during polling updates
**When to use:** Any list that updates via polling
**Example:**
```svelte
<script lang="ts">
  // The existing reconcileArray pattern already handles this well.
  // Key insight: Svelte 5's fine-grained reactivity + keyed each blocks
  // + in-place mutation via reconcileArray minimizes DOM churn.

  // Additional safeguard: track container scroll position
  let container: HTMLElement;
  let savedScrollTop = 0;

  // Save scroll before update
  $effect.pre(() => {
    void serverStore.servers.length; // track dependency
    if (container) {
      savedScrollTop = container.scrollTop;
    }
  });

  // Restore scroll after update (only if significantly different)
  $effect(() => {
    void serverStore.servers; // track dependency
    if (container && Math.abs(container.scrollTop - savedScrollTop) > 10) {
      container.scrollTop = savedScrollTop;
    }
  });
</script>

<div bind:this={container} class="overflow-auto">
  {#each serverStore.servers as server (server.profile_id)}
    <ServerTile {server} />
  {/each}
</div>
```

### Anti-Patterns to Avoid
- **Re-creating arrays on every poll:** Use reconcileArray to mutate in-place
- **Missing keyed each blocks:** Always use `(item.id)` syntax for identity
- **External charting libraries for simple gauges:** SVG + Tailwind is simpler and lighter
- **Polling without visibility check:** Already handled by pollingCoordinator

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Searchable dropdown | Custom input + list | bits-ui Combobox | Keyboard nav, a11y, portal handling |
| Circular gauge | Canvas/charting lib | SVG with stroke-dasharray | Simpler, no deps, Tailwind compatible |
| Process metrics | Shell commands | psutil | Cross-platform, type-safe, already installed |
| Polling coordination | setInterval per component | pollingCoordinator | Dedup, throttle, visibility handling |
| Array reconciliation | Replace array on update | reconcileArray | Preserves DOM identity, minimizes re-renders |

**Key insight:** The project already has sophisticated infrastructure for polling and state reconciliation. The main work is extending the backend to provide additional metrics and building the UI components.

## Common Pitfalls

### Pitfall 1: GPU Metrics Unavailability
**What goes wrong:** Assuming psutil provides GPU metrics on macOS
**Why it happens:** psutil doesn't expose GPU data; Apple Silicon uses unified memory
**How to avoid:**
- For basic monitoring: RSS memory from psutil is sufficient (it includes unified memory allocation)
- For detailed GPU metrics: Use macmon CLI tool and parse JSON output
- Accept limitation: Per-process GPU usage is not easily available on macOS
**Warning signs:** Looking for `gpu_percent` or similar in psutil docs

### Pitfall 2: Tokens/Second Not Available from Server
**What goes wrong:** Expecting mlx-openai-server to expose throughput metrics
**Why it happens:** mlx-openai-server returns standard OpenAI response format with token counts, not rates
**How to avoid:**
- Calculate client-side: Track request duration and completion_tokens to compute tokens/sec
- Alternative: Parse server logs for generation timing info
- Or accept showing total_tokens instead of tokens/sec
**Warning signs:** Searching for dedicated metrics endpoints

### Pitfall 3: Scroll Jump Despite Keyed Each
**What goes wrong:** Scroll position resets even with keyed each blocks
**Why it happens:** Parent container re-renders, or items change height during update
**How to avoid:**
- Use `$effect.pre` to capture scroll position before update
- Use CSS `contain: layout` on items to prevent layout thrashing
- Ensure item heights are stable (avoid content that loads async)
**Warning signs:** Scroll jumps only on certain updates, not all

### Pitfall 4: Combobox Input Value Stale After Selection
**What goes wrong:** Programmatically changing value doesn't update input text
**Why it happens:** Known issue in bits-ui (reported March 2025)
**How to avoid:**
- Clear searchValue in onOpenChangeComplete callback
- Don't rely on two-way binding for displayed text after selection
- Test selection flow thoroughly
**Warning signs:** Selection shows in dropdown but input shows old search text

### Pitfall 5: Memory Leak with Multiple Polling Loops
**What goes wrong:** Each component instance creates its own polling interval
**Why it happens:** Not using centralized polling coordinator
**How to avoid:**
- Always use pollingCoordinator.register() and pollingCoordinator.start()
- Clean up in component onDestroy/cleanup function
- Check isPolling() before starting new polls
**Warning signs:** Memory growth over time, duplicate API calls in network tab

## Code Examples

Verified patterns from official sources:

### OpenAI Response Usage Parsing (for tokens tracking)
```typescript
// Source: OpenAI API documentation
// The response includes usage statistics that can be accumulated

interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// Track cumulative tokens per server
class ServerMetrics {
  private totalTokens = new Map<number, number>();
  private requestCount = new Map<number, number>();

  recordCompletion(profileId: number, usage: ChatCompletionUsage) {
    const current = this.totalTokens.get(profileId) ?? 0;
    this.totalTokens.set(profileId, current + usage.completion_tokens);

    const count = this.requestCount.get(profileId) ?? 0;
    this.requestCount.set(profileId, count + 1);
  }

  getTotalTokens(profileId: number): number {
    return this.totalTokens.get(profileId) ?? 0;
  }
}
```

### Extended Server Stats Type
```python
# Source: Extension of existing types.py pattern
class ExtendedServerStats(TypedDict):
    """Extended statistics for server dashboard."""
    pid: int
    memory_mb: float
    memory_percent: float  # NEW: percentage of system RAM
    cpu_percent: float
    status: str
    create_time: float
    uptime_seconds: float  # NEW: computed from create_time
    # Token tracking (if implemented)
    total_tokens: int | None
    requests_count: int | None
```

### SVG Gauge Color Thresholds
```typescript
// Source: Standard UX pattern for resource monitoring
function getGaugeColor(value: number, metric: 'memory' | 'cpu'): string {
  // Memory: concern starts earlier (models can OOM)
  if (metric === 'memory') {
    if (value > 85) return 'text-red-500';
    if (value > 70) return 'text-yellow-500';
    return 'text-green-500';
  }

  // CPU: higher thresholds acceptable for inference
  if (value > 95) return 'text-red-500';
  if (value > 80) return 'text-yellow-500';
  return 'text-green-500';
}
```

## State of the Art (2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Chart.js for all charts | SVG + CSS for simple gauges | 2024+ | Smaller bundles, better Svelte integration |
| beforeUpdate/afterUpdate | $effect.pre/$effect | Svelte 5 (2024) | More precise dependency tracking |
| Custom dropdown components | bits-ui Combobox | bits-ui 1.0 (2025) | First-class searchable select primitive |
| window scroll tracking | Container-scoped scroll | Current best practice | More reliable, works with virtualization |

**New tools/patterns to consider:**
- **bits-ui 1.x:** Now has dedicated Combobox component (not just Select)
- **Svelte 5 SvelteMap/SvelteSet:** For reactive collections (already in use)
- **macmon:** Sudoless Apple Silicon metrics including GPU

**Deprecated/outdated:**
- **beforeUpdate/afterUpdate:** Replaced by $effect.pre/$effect in Svelte 5
- **Store subscriptions for polling:** Use polling coordinator pattern instead

## Open Questions

Things that couldn't be fully resolved:

1. **Per-process GPU memory on Apple Silicon**
   - What we know: macmon provides system-wide GPU metrics, psutil provides process RSS
   - What's unclear: How to attribute GPU memory to specific process on unified memory architecture
   - Recommendation: Show system GPU usage OR show only RSS memory per server (which includes unified memory allocation)

2. **Tokens/second metric source**
   - What we know: mlx-openai-server returns standard usage object (prompt_tokens, completion_tokens)
   - What's unclear: Whether server logs contain generation timing info
   - Recommendation: Calculate client-side by timing completion requests, or omit tokens/sec and show total_tokens instead

3. **bits-ui Combobox input sync issue**
   - What we know: Issue #1317 reports input text not updating on programmatic value change
   - What's unclear: Whether fixed in current version (1.8.0)
   - Recommendation: Test thoroughly; may need workaround to manually sync input value

## Sources

### Primary (HIGH confidence)
- bits-ui Combobox documentation: https://www.bits-ui.com/docs/components/combobox
- Svelte 5 $effect documentation: https://svelte.dev/docs/svelte/$effect
- psutil documentation: https://psutil.readthedocs.io/
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat

### Secondary (MEDIUM confidence)
- macmon GitHub: https://github.com/vladkens/macmon - JSON output format for Apple Silicon metrics
- mlx-lm SERVER.md: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md - Response format
- Tailwind radial progress patterns: https://tailwindflex.com/@shariful-islam/radial-progress-indicator-with-tailwindcss
- daisyUI radial progress: https://daisyui.com/components/radial-progress/

### Tertiary (LOW confidence - needs validation)
- bits-ui issue #1317 regarding input sync: https://github.com/huntabyte/bits-ui/issues/1317 - May be resolved
- Apple Silicon GPU monitoring limitations: Based on community discussions, not official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project or well-documented
- Architecture patterns: MEDIUM - Patterns verified with docs, but specific Svelte 5 + bits-ui combo needs testing
- Pitfalls: MEDIUM - Based on issue trackers and community reports
- Metrics collection: MEDIUM - psutil verified, GPU metrics require validation

**Research date:** 2026-01-17
**Valid until:** 30 days (stable domain, bits-ui may have updates)
